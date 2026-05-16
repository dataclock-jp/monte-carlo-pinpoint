"""
working_memory.py
作業記憶（ワーキングメモリ）モジュール

人間のワーキングメモリに着想を得た、エージェントの短期作業記憶。
直近Nステップの操作と結果を構造化して保持し、LLMプロンプトに
コンテキストとして注入する。

容量はデフォルト7チャンク（Miller の 7±2 の中央値）。
各チャンクは1ステップの操作＋結果＋画面状態の要約で構成される。
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryChunk:
    """作業記憶の1チャンク = 1ステップの操作記録。"""
    step: int
    action_type: str
    target: str
    value: str
    result: str            # "成功" / "失敗: ..." 等
    screen_state: str      # アクション後の画面状態の要約
    grid_cell: str = ""    # 操作対象のグリッドセル


class WorkingMemory:
    """
    リングバッファ方式の作業記憶。

    容量を超えると古いチャンクから自動的に忘却される。
    LLMプロンプトに含めるためのフォーマット済みテキストを生成する。
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._chunks: deque[MemoryChunk] = deque(maxlen=capacity)

    def store(self, chunk: MemoryChunk) -> None:
        """チャンクを記憶に追加する。容量超過時は最古のチャンクが消える。"""
        self._chunks.append(chunk)

    def recall(self) -> str:
        """
        記憶内容をLLMプロンプト用のテキストに整形して返す。
        空の場合は空文字列を返す。
        """
        if not self._chunks:
            return ""

        lines = []
        for chunk in self._chunks:
            parts = [f"Step {chunk.step}: {chunk.action_type}"]
            if chunk.target:
                parts.append(f"対象={chunk.target}")
            if chunk.value:
                parts.append(f"値={chunk.value[:30]}")
            if chunk.grid_cell:
                parts.append(f"位置={chunk.grid_cell}")
            parts.append(f"→ {chunk.result}")
            if chunk.screen_state:
                parts.append(f"画面: {chunk.screen_state[:60]}")
            lines.append(" | ".join(parts))

        return "\n".join(lines)

    def recall_as_prompt(self) -> str:
        """LLMシステムプロンプトに挿入する形式で返す。空なら空文字列。"""
        content = self.recall()
        if not content:
            return ""
        return f"\n■ 作業記憶（直近{len(self._chunks)}ステップ）\n{content}"

    def snapshot(self) -> 'WorkingMemory':
        """現在の状態のスナップショットを返す（バックグラウンド S2 用）。
        元の WorkingMemory とは独立したコピーなのでスレッドセーフ。"""
        wm = WorkingMemory(capacity=self.capacity)
        wm._chunks = deque(self._chunks, maxlen=self.capacity)
        return wm

    def clear(self) -> None:
        """全チャンクをクリアする。"""
        self._chunks.clear()

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return f"WorkingMemory({len(self._chunks)}/{self.capacity})"
