"""
agent_loop.py
Video2AI 自律デスクトップエージェント - メインループ

「画面変化検知 → LLM Vision解析 → マウス/キーボード操作」を
自律的に繰り返すエージェントループ。

使用例:
    # ドライランモード（操作なし、ログのみ）
    python agent_loop.py --goal "ファイルを保存する" --dry-run

    # 実際に操作する（注意: 実際のマウス/キーボードが動きます）
    python agent_loop.py --goal "Chromeでgoogle.comを開く" --max-steps 10

    # 設定を指定して実行
    python agent_loop.py \\
        --goal "テキストエディタでHello Worldと入力して保存する" \\
        --method ssim \\
        --threshold 0.05 \\
        --interval 1.0 \\
        --model gemini-2.5-flash \\
        --max-steps 20 \\
        --display :99
"""
import os
import sys
import time
import uuid
import signal
import argparse
import json
from pathlib import Path
from typing import Optional

# video2aiルートをパスに追加
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.screen_capture import ScreenCapture
from agent.change_detector import ChangeDetector
from agent.llm_vision import LLMVision
from agent.action_executor import ActionExecutor
from agent.session_logger import SessionLogger
from agent.safety_rules import SafetyRuleChecker, SafetyDecision


def run_agent_loop(
    goal: str = "",
    method: str = "diff",
    threshold: Optional[float] = None,
    interval: float = 1.0,
    model: str = "gemini-2.5-flash",
    max_steps: int = 0,
    dry_run: bool = False,
    display: str = ":99",
    output_dir: str = "/tmp/video2ai_agent_sessions",
    resize_capture: int = 0,
    language: str = "ja",
    verbose: bool = True,
    on_event=None,  # コールバック関数（Web UI統合用）
    safety_rules_path: Optional[str] = None,
    preview_mode: bool = False,
    preview_delay: float = 3.0,
    parallel: bool = False,
    num_workers: int = 3,
    worker_model: Optional[str] = None,
    coordinator_model: Optional[str] = None,
    coordinator_interval: float = 1.0,
    staleness_threshold: float = 5.0,
) -> dict:
    """
    エージェントループを実行する。
    
    Parameters
    ----------
    goal : str
        エージェントの目標（例: "ファイルを保存する"）
    method : str
        変化検知手法（"diff", "ssim", "optical-flow"）
    threshold : float, optional
        変化検知閾値（Noneで手法ごとのデフォルト値を使用）
    interval : float
        キャプチャ間隔（秒）
    model : str
        LLMモデル名
    max_steps : int
        最大ステップ数（0で無制限）
    dry_run : bool
        Trueの場合、実際の操作を行わずログのみ
    display : str
        X11ディスプレイ番号
    output_dir : str
        セッションログ出力ディレクトリ
    resize_capture : int
        キャプチャリサイズ幅（0でリサイズなし）
    language : str
        LLM応答言語（"ja" or "en"）
    verbose : bool
        詳細ログを表示するか
    on_event : callable, optional
        各フレーム処理後に呼ばれるコールバック（Web UI統合用）
        引数: (event_type: str, data: dict)
    safety_rules_path : str, optional
        安全ルールJSONファイルのパス（DENY/CONFIRM判定用）
    preview_mode : bool
        Trueの場合、操作前にオーバーレイで予告表示しEscでキャンセル可能
    preview_delay : float
        プレビューモード時の待機時間（秒）
    parallel : bool
        Trueの場合、並列ワーカー/コーディネーターモードで実行
    num_workers : int
        並列モードのワーカー数（デフォルト: 3）
    worker_model : str, optional
        ワーカー用LLMモデル（Noneでmodelと同じ）
    coordinator_model : str, optional
        コーディネーター用LLMモデル（Noneでmodelと同じ）
    coordinator_interval : float
        コーディネーターの判断間隔（秒）
    staleness_threshold : float
        報告の有効期限（秒）

    Returns
    -------
    dict
        セッションサマリー
    """
    session_id = str(uuid.uuid4())[:8]

    def log(msg: str):
        if verbose:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {msg}", flush=True)
        if on_event:
            on_event("log", {"message": msg, "timestamp": time.time()})

    def emit(event_type: str, data: dict):
        """JSON Lines形式でイベントを標準出力に出力（Web UI統合用）。"""
        entry = {"type": event_type, "timestamp": time.time(), **data}
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if on_event:
            on_event(event_type, data)

    log(f"=== Video2AI Agent Loop 開始 ===")
    log(f"セッションID: {session_id}")
    log(f"目標: {goal or '(未設定)'}")
    log(f"変化検知: {method} / 閾値: {threshold or 'auto'}")
    log(f"LLMモデル: {model}")
    log(f"ドライラン: {dry_run}")
    log(f"ディスプレイ: {display}")

    # 環境変数設定
    os.environ["DISPLAY"] = display

    # 各モジュールを初期化
    # 安全ルールチェッカー
    safety_checker = SafetyRuleChecker(safety_rules_path) if safety_rules_path else None
    if safety_checker and safety_checker.rules:
        log(f"安全ルール: {len(safety_checker.rules)} 件読み込み ({safety_rules_path})")

    capture = ScreenCapture(resize_width=resize_capture)
    detector = ChangeDetector(method=method, threshold=threshold)
    executor = ActionExecutor(
        dry_run=dry_run,
        preview_mode=preview_mode,
        preview_delay=preview_delay,
        display=display,
    )
    logger = SessionLogger(
        session_id=session_id,
        output_dir=output_dir,
        goal=goal,
        model=model,
        method=method,
        threshold=detector.threshold,
    )

    # LLM Visionは初回変化検知時に初期化（API keyチェック）
    vision: Optional[LLMVision] = None
    try:
        vision = LLMVision(model=model, language=language)
        log(f"LLM Vision初期化完了: {model}")
    except ValueError as e:
        log(f"警告: LLM Vision初期化失敗 ({e}). 変化検知のみ実行します。")

    # Ctrl+C で安全に終了
    _running = [True]
    def _on_sigint(sig, frame):
        log("\n中断シグナルを受信。セッションを終了します...")
        _running[0] = False
    signal.signal(signal.SIGINT, _on_sigint)

    step = 0
    prev_analysis_text = ""
    screen_size = (1920, 1080)

    emit("session_start", {
        "session_id": session_id,
        "goal": goal,
        "model": model,
        "method": method,
        "threshold": detector.threshold,
        "dry_run": dry_run,
        "parallel": parallel,
    })

    # 並列モード: Worker/Coordinator パターンで実行
    if parallel:
        from agent.parallel_coordinator import ParallelAgent
        log("=== 並列ワーカーモード ===")
        with capture:
            screen_size = capture.get_screen_size()
        parallel_agent = ParallelAgent(
            goal=goal,
            worker_model=worker_model or model,
            coordinator_model=coordinator_model or model,
            num_workers=num_workers,
            coordinator_interval=coordinator_interval,
            staleness_threshold=staleness_threshold,
            executor=executor,
            capture=capture,
            detector=detector,
            logger=logger,
            safety_checker=safety_checker,
            language=language,
            screen_size=screen_size,
            dry_run=dry_run,
            verbose=verbose,
            on_event=on_event,
        )
        return parallel_agent.run(max_steps=max_steps, interval=interval)

    try:
        with capture:
            screen_size = capture.get_screen_size()
            log(f"スクリーンサイズ: {screen_size[0]}x{screen_size[1]}")

            while _running[0]:
                if max_steps > 0 and step >= max_steps:
                    log(f"最大ステップ数 ({max_steps}) に達しました。終了します。")
                    break

                loop_start = time.time()
                step += 1

                # 1. 画面キャプチャ
                try:
                    frame = capture.capture()
                except Exception as e:
                    log(f"キャプチャエラー: {e}")
                    time.sleep(interval)
                    continue

                # 2. 変化検知
                changed, diff_score = detector.is_changed(frame)

                if verbose and step % 10 == 0:
                    log(f"Step {step}: score={diff_score:.4f} changed={changed} | {detector.stats}")

                # 3. 変化があった場合のみLLM Vision解析
                description = ""
                changes_text = ""
                action_type = "none"
                action_target = ""
                action_value = ""
                action_x = None
                action_y = None
                action_success = True
                action_reasoning = ""

                if changed and vision is not None:
                    log(f"Step {step}: 変化検知! score={diff_score:.4f} → LLM Vision解析中...")

                    try:
                        frame_b64 = capture.capture_as_base64(quality=75)
                        analysis = vision.analyze(
                            current_frame_b64=frame_b64,
                            goal=goal,
                            prev_analysis=prev_analysis_text,
                            diff_score=diff_score,
                            screen_size=screen_size,
                        )

                        description = analysis.description
                        changes_text = analysis.changes
                        suggestion = analysis.suggested_action
                        action_type = suggestion.action_type
                        action_target = suggestion.target
                        action_value = suggestion.value
                        action_x = suggestion.x
                        action_y = suggestion.y
                        action_reasoning = suggestion.reasoning
                        prev_analysis_text = description

                        log(f"  画面: {description[:80]}...")
                        log(f"  変化: {changes_text[:60]}...")
                        log(f"  提案: [{action_type}] {action_target} ({suggestion.confidence:.0%})")
                        log(f"  理由: {action_reasoning[:80]}...")

                        emit("change_detected", {
                            "step": step,
                            "diff_score": diff_score,
                            "description": description,
                            "changes": changes_text,
                            "action_type": action_type,
                            "action_target": action_target,
                            "action_value": action_value,
                            "action_x": action_x,
                            "action_y": action_y,
                            "action_reasoning": action_reasoning,
                            "confidence": suggestion.confidence,
                        })

                        # 4. アクション実行（安全ルールチェック → 実行）
                        if action_type != "none" and suggestion.confidence >= 0.5:
                            # 安全ルールチェック
                            if safety_checker is not None:
                                verdict = safety_checker.check(suggestion)
                                if verdict.decision == SafetyDecision.DENY:
                                    log(f"  [BLOCKED] {verdict.message} (rule: {verdict.matched_rule_id})")
                                    emit("action_blocked", {
                                        "step": step, "rule_id": verdict.matched_rule_id,
                                        "message": verdict.message, "action_type": action_type,
                                    })
                                    continue
                                elif verdict.decision == SafetyDecision.CONFIRM:
                                    log(f"  [CONFIRM] {verdict.message}")
                                    emit("safety_confirm", {
                                        "step": step, "rule_id": verdict.matched_rule_id,
                                        "message": verdict.message, "action_type": action_type,
                                    })
                                    try:
                                        answer = input("  続行しますか? [y/N]: ").strip().lower()
                                    except EOFError:
                                        answer = "n"
                                    if answer != "y":
                                        log(f"  ユーザーが拒否しました")
                                        continue

                            result = executor.execute_suggestion(suggestion)
                            action_success = result.success
                            if dry_run:
                                log(f"  [DRY-RUN] {result.description}")
                            elif result.success:
                                log(f"  ✓ アクション実行: {result.description}")
                            else:
                                log(f"  ✗ アクション失敗: {result.error}")

                            emit("action_executed", {
                                "step": step,
                                "action_type": action_type,
                                "description": result.description,
                                "success": result.success,
                                "dry_run": dry_run,
                                "error": result.error,
                            })

                    except Exception as e:
                        log(f"  LLM Vision解析エラー: {e}")
                        description = f"解析エラー: {e}"

                elif changed:
                    # Vision無効時は変化検知のみ記録
                    log(f"Step {step}: 変化検知 score={diff_score:.4f} (Vision無効)")
                    emit("change_detected", {
                        "step": step,
                        "diff_score": diff_score,
                        "description": "変化を検知しました（Vision解析なし）",
                        "changes": "",
                        "action_type": "none",
                    })

                # 5. ログ記録
                logger.log_frame(
                    diff_score=diff_score,
                    changed=changed,
                    description=description,
                    changes=changes_text,
                    action_type=action_type,
                    action_target=action_target,
                    action_value=action_value,
                    action_x=action_x,
                    action_y=action_y,
                    action_success=action_success,
                    action_reasoning=action_reasoning,
                )

                # 6. インターバル待機
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        log("キーボード割り込みで終了します。")
    finally:
        # セッション終了
        summary = logger.end_session()
        log(f"\n=== セッション終了 ===")
        log(f"総フレーム数: {summary.total_frames}")
        log(f"変化検知数: {summary.changed_frames}")
        log(f"実行アクション数: {summary.actions_taken}")
        log(f"経過時間: {summary.duration_seconds:.1f}秒")
        log(f"ログ保存先: {logger.output_dir}")

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
        emit("session_end", summary_dict)
        return summary_dict


def main():
    parser = argparse.ArgumentParser(
        description="Video2AI 自律デスクトップエージェント",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ドライランモード（操作なし、ログのみ）
  python agent_loop.py --goal "ファイルを保存する" --dry-run

  # 実際に操作する（注意: 実際のマウス/キーボードが動きます）
  python agent_loop.py --goal "Chromeでgoogle.comを開く" --max-steps 10

  # SSIMで精度重視の変化検知
  python agent_loop.py --goal "エラーを検知して報告する" --method ssim --threshold 0.05
        """,
    )
    parser.add_argument("--goal", default="", help="エージェントの目標")
    parser.add_argument("--method", default="diff", choices=["diff", "ssim", "optical-flow"],
                        help="変化検知手法（デフォルト: diff）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="変化検知閾値（デフォルト: 手法ごとの推奨値）")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="キャプチャ間隔（秒、デフォルト: 1.0）")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="LLMモデル名（デフォルト: gemini-2.5-flash）")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="最大ステップ数（0で無制限）")
    parser.add_argument("--dry-run", action="store_true",
                        help="ドライランモード（操作なし）")
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":99"),
                        help="X11ディスプレイ番号（デフォルト: :99）")
    parser.add_argument("--output-dir", default="/tmp/video2ai_agent_sessions",
                        help="セッションログ出力ディレクトリ")
    parser.add_argument("--language", default="ja", choices=["ja", "en"],
                        help="LLM応答言語（デフォルト: ja）")
    parser.add_argument("--quiet", action="store_true",
                        help="詳細ログを抑制する")
    parser.add_argument("--safety-rules", default=None,
                        help="安全ルールJSONファイルのパス")
    parser.add_argument("--preview", action="store_true",
                        help="プレビューモード（実行前に視覚的確認、Escでキャンセル）")
    parser.add_argument("--preview-delay", type=float, default=3.0,
                        help="プレビュー待機時間（秒、デフォルト: 3.0）")
    parser.add_argument("--parallel", action="store_true",
                        help="並列ワーカーモード（複数ワーカーが観察、コーディネーターが判断）")
    parser.add_argument("--workers", type=int, default=3,
                        help="並列モードのワーカー数（デフォルト: 3）")
    parser.add_argument("--worker-model", default=None,
                        help="ワーカー用LLMモデル（デフォルト: --model と同じ）")
    parser.add_argument("--coordinator-model", default=None,
                        help="コーディネーター用LLMモデル（デフォルト: --model と同じ）")
    parser.add_argument("--coordinator-interval", type=float, default=1.0,
                        help="コーディネーター判断間隔（秒、デフォルト: 1.0）")
    parser.add_argument("--staleness-threshold", type=float, default=5.0,
                        help="ワーカー報告の有効期限（秒、デフォルト: 5.0）")

    args = parser.parse_args()

    run_agent_loop(
        goal=args.goal,
        method=args.method,
        threshold=args.threshold,
        interval=args.interval,
        model=args.model,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        display=args.display,
        output_dir=args.output_dir,
        language=args.language,
        verbose=not args.quiet,
        safety_rules_path=args.safety_rules,
        preview_mode=args.preview,
        preview_delay=args.preview_delay,
        parallel=args.parallel,
        num_workers=args.workers,
        worker_model=args.worker_model,
        coordinator_model=args.coordinator_model,
        coordinator_interval=args.coordinator_interval,
        staleness_threshold=args.staleness_threshold,
    )


if __name__ == "__main__":
    main()
