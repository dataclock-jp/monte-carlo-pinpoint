"""
session_logger.py
エージェントセッションのログ記録モジュール

各フレームの解析結果・アクション・タイムスタンプをJSONL形式で記録する。
後から再生・分析・Web UIへの表示に利用できる。
"""
import json
import time
import os
import threading
from dataclasses import dataclass, asdict, field
from typing import Optional, List
from pathlib import Path


@dataclass
class FrameLog:
    """1フレーム分のログエントリ。"""
    timestamp: float           # Unix timestamp
    frame_index: int           # フレーム番号
    diff_score: float          # 変化スコア
    changed: bool              # 変化検知フラグ
    description: str           # 画面状況説明
    changes: str               # 前フレームからの変化
    action_type: str           # 実行したアクション種別
    action_target: str         # 操作対象
    action_value: str          # 入力値
    action_x: Optional[int]    # 操作X座標
    action_y: Optional[int]    # 操作Y座標
    action_success: bool       # アクション成功フラグ
    action_reasoning: str      # アクション理由
    image_path: str = ""       # フレーム画像パス（オプション）


@dataclass
class SessionSummary:
    """セッション全体のサマリー。"""
    session_id: str
    goal: str
    started_at: float
    ended_at: float
    duration_seconds: float
    total_frames: int
    changed_frames: int
    actions_taken: int
    action_breakdown: dict
    model: str
    method: str
    threshold: float


class SessionLogger:
    """
    エージェントセッションをJSONL形式でログ記録するクラス。
    
    各フレームの解析結果とアクションをリアルタイムで記録し、
    セッション終了時にサマリーを生成する。
    """

    def __init__(
        self,
        session_id: str,
        output_dir: str = "/tmp/video2ai_agent_sessions",
        goal: str = "",
        model: str = "",
        method: str = "diff",
        threshold: float = 0.02,
    ):
        self.session_id = session_id
        self.goal = goal
        self.model = model
        self.method = method
        self.threshold = threshold
        self.started_at = time.time()

        # 出力ディレクトリ
        self.output_dir = Path(output_dir) / session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "events.jsonl"
        self.summary_path = self.output_dir / "summary.json"

        self._frame_count = 0
        self._changed_count = 0
        self._action_counts: dict = {}
        self._logs: List[FrameLog] = []
        self._lock = threading.Lock()

        # セッション開始ログ
        self._write_meta({
            "type": "session_start",
            "session_id": session_id,
            "goal": goal,
            "model": model,
            "method": method,
            "threshold": threshold,
            "started_at": self.started_at,
        })

    def _write_meta(self, data: dict):
        """メタデータをJSONLに書き込む。"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def log_frame(
        self,
        diff_score: float,
        changed: bool,
        description: str = "",
        changes: str = "",
        action_type: str = "none",
        action_target: str = "",
        action_value: str = "",
        action_x: Optional[int] = None,
        action_y: Optional[int] = None,
        action_success: bool = True,
        action_reasoning: str = "",
        image_path: str = "",
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> FrameLog:
        """
        1フレーム分のログを記録する。スレッドセーフ。

        Parameters
        ----------
        frame_index : int, optional
            明示的なフレーム番号（並列モード用）。Noneで自動インクリメント。
        timestamp : float, optional
            明示的なタイムスタンプ（並列モード用）。Noneで現在時刻。

        Returns
        -------
        FrameLog
            記録したログエントリ
        """
        with self._lock:
            log = FrameLog(
                timestamp=timestamp if timestamp is not None else time.time(),
                frame_index=frame_index if frame_index is not None else self._frame_count,
                diff_score=diff_score,
                changed=changed,
                description=description,
                changes=changes,
                action_type=action_type,
                action_target=action_target,
                action_value=action_value,
                action_x=action_x,
                action_y=action_y,
                action_success=action_success,
                action_reasoning=action_reasoning,
                image_path=image_path,
            )

            self._frame_count += 1
            if changed:
                self._changed_count += 1
            if action_type != "none":
                self._action_counts[action_type] = self._action_counts.get(action_type, 0) + 1

            self._logs.append(log)

            # JSONLに追記
            entry = {"type": "frame", **asdict(log)}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            return log

    def end_session(self) -> SessionSummary:
        """
        セッションを終了してサマリーを生成・保存する。
        
        Returns
        -------
        SessionSummary
            セッションサマリー
        """
        ended_at = time.time()
        summary = SessionSummary(
            session_id=self.session_id,
            goal=self.goal,
            started_at=self.started_at,
            ended_at=ended_at,
            duration_seconds=ended_at - self.started_at,
            total_frames=self._frame_count,
            changed_frames=self._changed_count,
            actions_taken=sum(self._action_counts.values()),
            action_breakdown=self._action_counts,
            model=self.model,
            method=self.method,
            threshold=self.threshold,
        )

        # サマリーをJSONで保存
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

        # セッション終了ログ
        self._write_meta({
            "type": "session_end",
            "session_id": self.session_id,
            "ended_at": ended_at,
            "summary": asdict(summary),
        })

        return summary

    def get_recent_logs(self, n: int = 5) -> List[FrameLog]:
        """直近n件のログを返す。"""
        return self._logs[-n:]

    @property
    def stats(self) -> dict:
        """現在の統計情報を返す。"""
        return {
            "session_id": self.session_id,
            "frame_count": self._frame_count,
            "changed_count": self._changed_count,
            "action_counts": self._action_counts,
            "elapsed_seconds": time.time() - self.started_at,
        }
