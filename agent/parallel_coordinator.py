"""
parallel_coordinator.py
並列ワーカー / コーディネーター パターン

複数のワーカースレッドが画面変化を並列に観察（LLM Vision解析）し、
1つのコーディネータースレッドが報告を集約してアクションを決定・実行する。

アーキテクチャ:
  Main Thread     : キャプチャ → 変化検知 → FrameDispatcher に委任
  Worker Pool     : LLM観察（description + changes のみ、アクション提案なし）
  Coordinator     : 報告を集約 → LLMで判断 → SafetyCheck → ActionExecutor

ワーカーは「目」、コーディネーターは「頭と手」。
"""

from __future__ import annotations

import json
import time
import queue
import threading
import signal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Optional

from agent.llm_vision import LLMVision, ActionSuggestion
from agent.action_executor import ActionExecutor
from agent.session_logger import SessionLogger
from agent.safety_rules import SafetyRuleChecker, SafetyDecision


@dataclass
class WorkerReport:
    """ワーカーからの観察報告。アクション情報は含まない。"""
    frame_index: int
    captured_at: float       # フレームキャプチャ時刻
    reported_at: float       # ワーカー解析完了時刻
    description: str
    changes: str
    diff_score: float
    worker_id: int


class FrameDispatcher:
    """変化検知されたフレームをワーカープールに振り分ける。"""

    def __init__(
        self,
        num_workers: int,
        vision_worker: LLMVision,
        report_queue: queue.Queue,
        screen_size: tuple = (1920, 1080),
        on_log=None,
    ):
        self.vision = vision_worker
        self.report_queue = report_queue
        self.screen_size = screen_size
        self._on_log = on_log
        self._pool = ThreadPoolExecutor(max_workers=num_workers)
        self._worker_counter = 0
        self._dropped = 0

    def dispatch(self, frame_b64: str, frame_index: int, diff_score: float, captured_at: float):
        """フレームをワーカーに委任する。"""
        worker_id = self._worker_counter
        self._worker_counter += 1
        self._pool.submit(self._worker_task, frame_b64, frame_index, diff_score, captured_at, worker_id)

    def _worker_task(self, frame_b64: str, frame_index: int, diff_score: float, captured_at: float, worker_id: int):
        """ワーカースレッドで実行される観察タスク。"""
        try:
            result = self.vision.analyze_observation_only(
                current_frame_b64=frame_b64,
                diff_score=diff_score,
                screen_size=self.screen_size,
            )
            report = WorkerReport(
                frame_index=frame_index,
                captured_at=captured_at,
                reported_at=time.time(),
                description=result.get("description", ""),
                changes=result.get("changes", ""),
                diff_score=diff_score,
                worker_id=worker_id,
            )
            self.report_queue.put(report)
        except Exception as e:
            if self._on_log:
                self._on_log(f"  Worker-{worker_id} エラー: {e}")

    def shutdown(self):
        self._pool.shutdown(wait=False)

    @property
    def dropped_count(self) -> int:
        return self._dropped


class ActionCoordinator(threading.Thread):
    """
    ワーカーからの報告を集約し、アクションを決定・実行するスレッド。
    """

    def __init__(
        self,
        report_queue: queue.Queue,
        vision_coordinator: LLMVision,
        executor: ActionExecutor,
        safety_checker: Optional[SafetyRuleChecker],
        logger: SessionLogger,
        goal: str,
        coordinator_interval: float = 1.0,
        staleness_threshold: float = 5.0,
        latest_frame_ref: Optional[list] = None,
        frame_lock: Optional[threading.Lock] = None,
        screen_size: tuple = (1920, 1080),
        dry_run: bool = False,
        on_log=None,
        on_event=None,
    ):
        super().__init__(daemon=True)
        self.report_queue = report_queue
        self.vision = vision_coordinator
        self.executor = executor
        self.safety_checker = safety_checker
        self.logger = logger
        self.goal = goal
        self.coordinator_interval = coordinator_interval
        self.staleness_threshold = staleness_threshold
        self.latest_frame_ref = latest_frame_ref or [None]
        self.frame_lock = frame_lock or threading.Lock()
        self.screen_size = screen_size
        self.dry_run = dry_run
        self._on_log = on_log
        self._on_event = on_event
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.coordinator_interval)
            if self._stop_event.is_set():
                break
            self._process_reports()

    def _process_reports(self):
        """キューから報告を取り出し、判断して実行する。"""
        reports: list[WorkerReport] = []
        now = time.time()

        # キューを全てドレイン
        while True:
            try:
                report = self.report_queue.get_nowait()
                # 古い報告を破棄
                if now - report.captured_at <= self.staleness_threshold:
                    reports.append(report)
            except queue.Empty:
                break

        if not reports:
            return

        # 最新の報告順にソート
        reports.sort(key=lambda r: r.captured_at, reverse=True)

        if self._on_log:
            self._on_log(f"  [Coordinator] {len(reports)} 件の報告を処理")

        # 報告をテキストに変換
        report_dicts = [{"description": r.description, "changes": r.changes} for r in reports]

        # 最新スクリーンショットを取得
        with self.frame_lock:
            latest_frame = self.latest_frame_ref[0]

        # コーディネーターLLMでアクション決定
        try:
            suggestion = self.vision.decide_action(
                reports=report_dicts,
                goal=self.goal,
                current_frame_b64=latest_frame,
                screen_size=self.screen_size,
            )
        except Exception as e:
            if self._on_log:
                self._on_log(f"  [Coordinator] LLMエラー: {e}")
            return

        if self._on_log:
            self._on_log(f"  [Coordinator] 判断: [{suggestion.action_type}] {suggestion.target} ({suggestion.confidence:.0%})")

        if self._on_event:
            self._on_event("coordinator_decision", {
                "action_type": suggestion.action_type,
                "target": suggestion.target,
                "confidence": suggestion.confidence,
                "reasoning": suggestion.reasoning,
                "reports_count": len(reports),
            })

        # アクション実行
        if suggestion.action_type != "none" and suggestion.confidence >= 0.5:
            # 安全ルールチェック
            if self.safety_checker is not None:
                verdict = self.safety_checker.check(suggestion)
                if verdict.decision == SafetyDecision.DENY:
                    if self._on_log:
                        self._on_log(f"  [Coordinator] [BLOCKED] {verdict.message}")
                    return
                elif verdict.decision == SafetyDecision.CONFIRM:
                    if self._on_log:
                        self._on_log(f"  [Coordinator] [CONFIRM] {verdict.message} — 並列モードではスキップ")
                    return

            result = self.executor.execute_suggestion(suggestion)
            if self._on_log:
                prefix = "[DRY-RUN] " if self.dry_run else ""
                if result.success:
                    self._on_log(f"  [Coordinator] {prefix}実行: {result.description}")
                elif result.cancelled:
                    self._on_log(f"  [Coordinator] キャンセル: {result.description}")
                else:
                    self._on_log(f"  [Coordinator] 失敗: {result.error}")

        # ログ記録（最新の報告のframe_indexとtimestampを使用）
        latest_report = reports[0]
        self.logger.log_frame(
            diff_score=latest_report.diff_score,
            changed=True,
            description=latest_report.description,
            changes=latest_report.changes,
            action_type=suggestion.action_type,
            action_target=suggestion.target,
            action_value=suggestion.value,
            action_x=suggestion.x,
            action_y=suggestion.y,
            action_success=True,
            action_reasoning=suggestion.reasoning,
            frame_index=latest_report.frame_index,
            timestamp=latest_report.captured_at,
        )


class ParallelAgent:
    """
    並列ワーカー/コーディネーターパターンのエージェント。

    Main Thread: キャプチャ → 変化検知 → ワーカーに委任
    Worker Pool: 画面観察（LLM Vision、観察のみ）
    Coordinator: 報告集約 → アクション決定 → 実行
    """

    def __init__(
        self,
        goal: str,
        worker_model: str,
        coordinator_model: str,
        num_workers: int,
        coordinator_interval: float,
        staleness_threshold: float,
        executor: ActionExecutor,
        capture,
        detector,
        logger: SessionLogger,
        safety_checker: Optional[SafetyRuleChecker],
        language: str,
        screen_size: tuple,
        dry_run: bool,
        verbose: bool,
        on_event=None,
    ):
        self.goal = goal
        self.capture = capture
        self.detector = detector
        self.logger = logger
        self.verbose = verbose
        self.on_event = on_event
        self._running = True

        self._report_queue: queue.Queue[WorkerReport] = queue.Queue()
        self._latest_frame_ref: list = [None]
        self._frame_lock = threading.Lock()

        def log(msg):
            if verbose:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] {msg}", flush=True)
            if on_event:
                on_event("log", {"message": msg})

        self._log = log

        # ワーカー用Vision（軽量モデル）
        try:
            vision_worker = LLMVision(model=worker_model, language=language)
            log(f"Worker Vision初期化: {worker_model} x{num_workers}")
        except ValueError as e:
            log(f"Worker Vision初期化失敗: {e}")
            vision_worker = None

        # コーディネーター用Vision
        try:
            vision_coordinator = LLMVision(model=coordinator_model, language=language)
            log(f"Coordinator Vision初期化: {coordinator_model}")
        except ValueError as e:
            log(f"Coordinator Vision初期化失敗: {e}")
            vision_coordinator = None

        # FrameDispatcher
        self._dispatcher = None
        if vision_worker:
            self._dispatcher = FrameDispatcher(
                num_workers=num_workers,
                vision_worker=vision_worker,
                report_queue=self._report_queue,
                screen_size=screen_size,
                on_log=log,
            )

        # ActionCoordinator
        self._coordinator = None
        if vision_coordinator:
            self._coordinator = ActionCoordinator(
                report_queue=self._report_queue,
                vision_coordinator=vision_coordinator,
                executor=executor,
                safety_checker=safety_checker,
                logger=logger,
                goal=goal,
                coordinator_interval=coordinator_interval,
                staleness_threshold=staleness_threshold,
                latest_frame_ref=self._latest_frame_ref,
                frame_lock=self._frame_lock,
                screen_size=screen_size,
                dry_run=dry_run,
                on_log=log,
                on_event=on_event,
            )

    def run(self, max_steps: int = 0, interval: float = 1.0) -> dict:
        """
        並列エージェントループを実行する。

        Returns
        -------
        dict
            セッションサマリー
        """
        # Ctrl+C ハンドラ
        def _on_sigint(sig, frame):
            self._log("\n中断シグナル受信。並列エージェントを停止します...")
            self._running = False
        signal.signal(signal.SIGINT, _on_sigint)

        # コーディネータースレッド開始
        if self._coordinator:
            self._coordinator.start()
            self._log("Coordinator スレッド開始")

        step = 0
        try:
            with self.capture:
                screen_size = self.capture.get_screen_size()
                self._log(f"並列モード開始: スクリーン {screen_size[0]}x{screen_size[1]}")

                while self._running:
                    if max_steps > 0 and step >= max_steps:
                        self._log(f"最大ステップ数 ({max_steps}) に到達")
                        break

                    loop_start = time.time()
                    step += 1

                    # キャプチャ
                    try:
                        frame = self.capture.capture()
                    except Exception as e:
                        self._log(f"キャプチャエラー: {e}")
                        time.sleep(interval)
                        continue

                    # 変化検知
                    changed, diff_score = self.detector.is_changed(frame)

                    if self.verbose and step % 10 == 0:
                        self._log(f"Step {step}: score={diff_score:.4f} changed={changed}")

                    # 変化があればワーカーに委任
                    if changed and self._dispatcher:
                        captured_at = time.time()
                        frame_b64 = self.capture.capture_as_base64(quality=75)

                        # 最新フレームを更新（コーディネーター用）
                        with self._frame_lock:
                            self._latest_frame_ref[0] = frame_b64

                        self._dispatcher.dispatch(frame_b64, step, diff_score, captured_at)
                        self._log(f"Step {step}: 変化検知 score={diff_score:.4f} → Worker に委任")

                    # インターバル
                    elapsed = time.time() - loop_start
                    sleep_time = max(0.0, interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            self._log("キーボード割り込み")
        finally:
            if self._coordinator:
                self._coordinator.stop()
                self._coordinator.join(timeout=5)
            if self._dispatcher:
                self._dispatcher.shutdown()

            summary = self.logger.end_session()
            self._log(f"\n=== 並列エージェント終了 ===")
            self._log(f"総ステップ数: {summary.total_frames}")
            self._log(f"変化検知数: {summary.changed_frames}")
            self._log(f"実行アクション数: {summary.actions_taken}")

            summary_dict = {
                "session_id": summary.session_id,
                "goal": summary.goal,
                "started_at": summary.started_at,
                "ended_at": summary.ended_at,
                "duration_seconds": summary.duration_seconds,
                "total_frames": summary.total_frames,
                "changed_frames": summary.changed_frames,
                "actions_taken": summary.actions_taken,
                "action_breakdown": summary.action_breakdown,
                "model": summary.model,
                "method": summary.method,
                "threshold": summary.threshold,
            }
            if self.on_event:
                self.on_event("session_end", summary_dict)
            return summary_dict
