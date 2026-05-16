"""
workflow.py
自動化ワークフロー — 操作の記録・再生

MCP ツールで行った操作をワークフローとして記録し、
名前で呼び出して再生する。複数アプリ跨ぎの定型作業の自動化基盤。

使い方:
  1. workflow_record("save_clip") で記録開始
  2. 通常通り操作（click, keypress, type_text 等）
  3. workflow_stop() で記録停止・保存
  4. workflow_run("save_clip") で再生
"""
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Callable
from pathlib import Path


# 記録対象のツール（読み取り系は除外）
RECORDABLE_TOOLS = {
    "click", "right_click", "double_click", "drag", "scroll",
    "type_text", "keypress", "focus_window", "save_file",
    "move_to",
}


@dataclass
class WorkflowStep:
    """ワークフローの1ステップ。"""
    tool: str           # "click", "keypress" 等
    args: dict          # {"x": 100, "y": 200, "monitor": 3} 等
    delay: float = 0.5  # 前のステップからの待ち時間（秒）


@dataclass
class Workflow:
    """保存されたワークフロー。"""
    name: str
    description: str
    steps: List[WorkflowStep]
    created_at: float = 0.0
    last_run: float = 0.0
    run_count: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [{"tool": s.tool, "args": s.args, "delay": s.delay}
                      for s in self.steps],
            "created_at": self.created_at,
            "last_run": self.last_run,
            "run_count": self.run_count,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Workflow':
        steps = [WorkflowStep(tool=s["tool"], args=s["args"],
                              delay=s.get("delay", 0.5))
                 for s in d.get("steps", [])]
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            steps=steps,
            created_at=d.get("created_at", 0.0),
            last_run=d.get("last_run", 0.0),
            run_count=d.get("run_count", 0),
            tags=d.get("tags", []),
        )


class WorkflowManager:
    """ワークフローの記録・保存・再生を管理する。"""

    def __init__(self, storage_dir: str = ""):
        if storage_dir:
            self._path = Path(storage_dir) / "workflows.json"
        else:
            self._path = (Path.home() / "video2ai_agent_sessions"
                          / "workflows.json")
        self._workflows: Dict[str, Workflow] = {}
        self._recording_name: Optional[str] = None
        self._recording_steps: List[WorkflowStep] = []
        self._recording_description: str = ""
        self._recording_tags: List[str] = []
        self._last_step_time: float = 0.0
        self._load()

    @property
    def is_recording(self) -> bool:
        return self._recording_name is not None

    @property
    def recording_name(self) -> Optional[str]:
        return self._recording_name

    def start_recording(self, name: str, description: str = "",
                        tags: Optional[List[str]] = None) -> str:
        """記録を開始する。"""
        if self.is_recording:
            return f"既に '{self._recording_name}' を記録中です。先に workflow_stop してください。"
        self._recording_name = name
        self._recording_steps = []
        self._recording_description = description
        self._recording_tags = tags or []
        self._last_step_time = time.time()
        return f"ワークフロー '{name}' の記録を開始しました。操作してください。"

    def record_step(self, tool: str, args: dict) -> None:
        """操作を記録する（_log_action から呼ばれる）。"""
        if not self.is_recording:
            return
        if tool not in RECORDABLE_TOOLS:
            return

        now = time.time()
        delay = min(now - self._last_step_time, 5.0)  # 最大5秒にキャップ
        self._last_step_time = now

        self._recording_steps.append(WorkflowStep(
            tool=tool,
            args=args,
            delay=round(delay, 2),
        ))

    def stop_recording(self) -> str:
        """記録を停止してワークフローを保存する。"""
        if not self.is_recording:
            return "記録中のワークフローがありません。"

        name = self._recording_name
        if not self._recording_steps:
            self._recording_name = None
            return f"'{name}' の記録を中止しました（操作が0件）。"

        wf = Workflow(
            name=name,
            description=self._recording_description,
            steps=self._recording_steps,
            created_at=time.time(),
            tags=self._recording_tags,
        )
        self._workflows[name] = wf
        self._save()

        self._recording_name = None
        self._recording_steps = []
        return (f"ワークフロー '{name}' を保存しました "
                f"（{len(wf.steps)}ステップ）。")

    def cancel_recording(self) -> str:
        """記録を中止する。"""
        if not self.is_recording:
            return "記録中のワークフローがありません。"
        name = self._recording_name
        self._recording_name = None
        self._recording_steps = []
        return f"'{name}' の記録を中止しました。"

    def run(self, name: str, executor_fn: Callable) -> str:
        """ワークフローを再生する。

        executor_fn(tool: str, args: dict) -> str を受け取り、
        各ステップの実行結果文字列を返す関数。
        """
        wf = self._workflows.get(name)
        if not wf:
            return f"ワークフロー '{name}' が見つかりません。"

        results = []
        for i, step in enumerate(wf.steps, 1):
            if step.delay > 0:
                time.sleep(step.delay)
            try:
                result = executor_fn(step.tool, step.args)
                results.append(f"  {i}. {step.tool}: OK")
            except Exception as e:
                results.append(f"  {i}. {step.tool}: FAIL - {e}")
                # エラーでも続行（最後にまとめて報告）

        wf.last_run = time.time()
        wf.run_count += 1
        self._save()

        header = f"ワークフロー '{name}' 完了（{len(wf.steps)}ステップ）"
        return header + "\n" + "\n".join(results)

    def list_workflows(self, tag: str = "") -> List[dict]:
        """ワークフロー一覧を返す。"""
        result = []
        for wf in self._workflows.values():
            if tag and tag not in wf.tags:
                continue
            result.append({
                "name": wf.name,
                "description": wf.description,
                "steps": len(wf.steps),
                "run_count": wf.run_count,
                "tags": wf.tags,
            })
        return result

    def get(self, name: str) -> Optional[Workflow]:
        return self._workflows.get(name)

    def delete(self, name: str) -> str:
        if name not in self._workflows:
            return f"ワークフロー '{name}' が見つかりません。"
        del self._workflows[name]
        self._save()
        return f"ワークフロー '{name}' を削除しました。"

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {name: wf.to_dict()
                    for name, wf in self._workflows.items()}
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for name, d in data.items():
                    self._workflows[name] = Workflow.from_dict(d)
        except Exception:
            pass
