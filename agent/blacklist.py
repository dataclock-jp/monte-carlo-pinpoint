"""
agent/blacklist.py
操作ブラックリスト（禁止リスト）

MCP サーバー用の操作禁止ルールを管理する。
JSON 設定ファイルでユーザーがカスタマイズ可能。
組み込みデフォルト（危険キー、システムディレクトリ）は常に有効。

設定ファイル検索順:
  1. プロジェクト直下: blacklist.json
  2. ユーザーホーム: ~/.video2ai/blacklist.json
  初回起動時に ~/.video2ai/blacklist.json を自動生成。

JSON 形式:
  {
    "forbidden_keys": ["win+l", "win+r"],
    "forbidden_paths": ["C:/Users/*/AppData/**"],
    "forbidden_windows": ["*パスワード*"],
    "forbidden_text_patterns": ["format c:"]
  }

パターンは fnmatch（glob）形式: * ? ** をサポート。
"""

from __future__ import annotations

import json
import logging
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 組み込みデフォルト（常に有効、設定ファイルで無効化できない）
# ---------------------------------------------------------------------------

_BUILTIN_FORBIDDEN_KEYS: set[str] = {
    "win+d", "win+m", "alt+f4", "ctrl+w",
    "ctrl+alt+delete", "ctrl+alt+del",
}

_BUILTIN_FORBIDDEN_PATHS: list[str] = [
    "c:/windows/**",
    "c:/program files/**",
    "c:/program files (x86)/**",
]

# デフォルト設定ファイルの内容
_DEFAULT_CONFIG = {
    "_comment": "Video2AI 操作ブラックリスト。パターンは glob 形式 (*/?/**)。",
    "forbidden_keys": [
        "win+l",
        "win+r",
    ],
    "forbidden_paths": [
        "C:/Users/*/AppData/**",
        "**/.env",
        "**/.ssh/**",
        "**/credentials*",
    ],
    "forbidden_windows": [
        "*パスワード*",
        "*Password*",
        "*credential*",
    ],
    "forbidden_text_patterns": [
        "format c:",
        "del /s /q",
        "rm -rf /",
    ],
}


def _normalize_key(key: str) -> str:
    """キー文字列を正規化する（小文字、スペース除去）。"""
    return key.lower().replace(" ", "")


def _normalize_path(filepath: str) -> str:
    """パス文字列を正規化する（スラッシュ統一、小文字化）。"""
    return filepath.replace("\\", "/").lower()


class OperationBlacklist:
    """MCP ツール用操作ブラックリスト。"""

    def __init__(self, project_root: Optional[str] = None):
        """
        Parameters
        ----------
        project_root : str or None
            プロジェクトルートパス。None の場合、このファイルの親の親を使用。
        """
        if project_root is None:
            project_root = str(Path(__file__).resolve().parent.parent)
        self._project_root = Path(project_root)

        # ユーザー設定（JSONから読み込み）
        self._user_keys: set[str] = set()
        self._user_paths: list[str] = []
        self._user_windows: list[str] = []
        self._user_text_patterns: list[str] = []
        self._config_path: Optional[Path] = None

        self._load_config()

    def _find_config(self) -> Optional[Path]:
        """設定ファイルを検索する。見つからなければデフォルトを生成。"""
        # 1. プロジェクト直下
        project_config = self._project_root / "blacklist.json"
        if project_config.exists():
            return project_config

        # 2. ユーザーホーム
        user_config = Path.home() / ".video2ai" / "blacklist.json"
        if user_config.exists():
            return user_config

        # 3. デフォルト生成
        try:
            user_config.parent.mkdir(parents=True, exist_ok=True)
            user_config.write_text(
                json.dumps(_DEFAULT_CONFIG, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"デフォルト blacklist.json を作成: {user_config}")
            return user_config
        except Exception as e:
            logger.warning(f"blacklist.json の作成に失敗: {e}")
            return None

    def _load_config(self) -> None:
        """設定ファイルを読み込む。"""
        self._config_path = self._find_config()
        if self._config_path is None:
            return
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            self._user_keys = {
                _normalize_key(k) for k in data.get("forbidden_keys", [])
            }
            self._user_paths = [
                _normalize_path(p) for p in data.get("forbidden_paths", [])
            ]
            self._user_windows = [
                p.lower() for p in data.get("forbidden_windows", [])
            ]
            self._user_text_patterns = [
                p.lower() for p in data.get("forbidden_text_patterns", [])
            ]
            logger.info(
                f"blacklist 読み込み完了: {self._config_path} "
                f"(keys={len(self._user_keys)}, paths={len(self._user_paths)}, "
                f"windows={len(self._user_windows)}, text={len(self._user_text_patterns)})"
            )
        except Exception as e:
            logger.warning(f"blacklist.json の読み込みに失敗: {e}")

    def reload(self) -> str:
        """設定ファイルを再読み込みする。"""
        self._user_keys.clear()
        self._user_paths.clear()
        self._user_windows.clear()
        self._user_text_patterns.clear()
        self._load_config()
        return f"blacklist 再読み込み完了: {self._config_path}"

    # ------------------------------------------------------------------
    # 追加・削除（JSONファイルに永続化）
    # ------------------------------------------------------------------

    _CATEGORY_MAP = {
        "key": "forbidden_keys",
        "path": "forbidden_paths",
        "window": "forbidden_windows",
        "text": "forbidden_text_patterns",
    }

    def add(self, category: str, pattern: str) -> str:
        """ブラックリストにパターンを追加し、JSONファイルに保存する。

        Parameters
        ----------
        category : str
            "key", "path", "window", "text" のいずれか
        pattern : str
            追加するパターン（例: "win+l", "D:/secret/**", "*メール*"）

        Returns
        -------
        str
            結果メッセージ
        """
        json_key = self._CATEGORY_MAP.get(category)
        if not json_key:
            return f"エラー: category は {list(self._CATEGORY_MAP.keys())} のいずれかを指定してください"

        # メモリに追加
        if category == "key":
            normalized = _normalize_key(pattern)
            if normalized in self._user_keys:
                return f"既に登録済み: {pattern} (category={category})"
            self._user_keys.add(normalized)
        elif category == "path":
            normalized = _normalize_path(pattern)
            if normalized in self._user_paths:
                return f"既に登録済み: {pattern} (category={category})"
            self._user_paths.append(normalized)
        elif category == "window":
            normalized = pattern.lower()
            if normalized in self._user_windows:
                return f"既に登録済み: {pattern} (category={category})"
            self._user_windows.append(normalized)
        elif category == "text":
            normalized = pattern.lower()
            if normalized in self._user_text_patterns:
                return f"既に登録済み: {pattern} (category={category})"
            self._user_text_patterns.append(normalized)

        # JSONファイルに保存
        return self._save_config(f"追加: {pattern} → {json_key}")

    def remove(self, category: str, pattern: str) -> str:
        """ブラックリストからパターンを削除し、JSONファイルに保存する。

        Parameters
        ----------
        category : str
            "key", "path", "window", "text" のいずれか
        pattern : str
            削除するパターン

        Returns
        -------
        str
            結果メッセージ
        """
        json_key = self._CATEGORY_MAP.get(category)
        if not json_key:
            return f"エラー: category は {list(self._CATEGORY_MAP.keys())} のいずれかを指定してください"

        # 組み込みルールは削除不可
        if category == "key" and _normalize_key(pattern) in _BUILTIN_FORBIDDEN_KEYS:
            return f"エラー: '{pattern}' は組み込み禁止キーのため削除できません"
        if category == "path":
            normalized = _normalize_path(pattern)
            if normalized in _BUILTIN_FORBIDDEN_PATHS:
                return f"エラー: '{pattern}' は組み込み禁止パスのため削除できません"

        # メモリから削除
        found = False
        if category == "key":
            normalized = _normalize_key(pattern)
            if normalized in self._user_keys:
                self._user_keys.discard(normalized)
                found = True
        elif category == "path":
            normalized = _normalize_path(pattern)
            if normalized in self._user_paths:
                self._user_paths.remove(normalized)
                found = True
        elif category == "window":
            normalized = pattern.lower()
            if normalized in self._user_windows:
                self._user_windows.remove(normalized)
                found = True
        elif category == "text":
            normalized = pattern.lower()
            if normalized in self._user_text_patterns:
                self._user_text_patterns.remove(normalized)
                found = True

        if not found:
            return f"見つかりません: {pattern} (category={category})"

        # JSONファイルに保存
        return self._save_config(f"削除: {pattern} ← {json_key}")

    def _save_config(self, action_desc: str) -> str:
        """現在のメモリ状態をJSONファイルに書き出す。"""
        if self._config_path is None:
            return f"エラー: 設定ファイルが見つかりません"
        try:
            # 元のJSONを読み込んでコメント等を保持
            data = {}
            if self._config_path.exists():
                data = json.loads(self._config_path.read_text(encoding="utf-8"))

            # メモリの状態で上書き（正規化前の表示用にそのまま保存）
            data["forbidden_keys"] = sorted(self._user_keys)
            data["forbidden_paths"] = self._user_paths[:]
            data["forbidden_windows"] = self._user_windows[:]
            data["forbidden_text_patterns"] = self._user_text_patterns[:]

            self._config_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return f"{action_desc}\n保存先: {self._config_path}"
        except Exception as e:
            return f"エラー: 保存失敗 — {e}"

    # ------------------------------------------------------------------
    # チェックメソッド（None=許可、文字列=拒否メッセージ）
    # ------------------------------------------------------------------

    def check_key(self, key: str) -> Optional[str]:
        """キー操作が禁止されているか判定する。"""
        normalized = _normalize_key(key)

        # 組み込み禁止キー（常に有効）
        if normalized in _BUILTIN_FORBIDDEN_KEYS:
            return f"ブロック: '{key}' は安全上の理由で禁止されています"

        # ユーザー設定の禁止キー
        if normalized in self._user_keys:
            return f"ブロック: '{key}' は blacklist 設定で禁止されています"

        return None

    def check_filepath(self, filepath: str) -> Optional[str]:
        """ファイルパスが禁止されているか判定する。"""
        normalized = _normalize_path(filepath)

        # 組み込み禁止パス（常に有効）
        for pattern in _BUILTIN_FORBIDDEN_PATHS:
            if fnmatch(normalized, pattern):
                return f"ブロック: '{filepath}' はシステム保護パスです"

        # ユーザー設定の禁止パス
        for pattern in self._user_paths:
            if fnmatch(normalized, pattern):
                return f"ブロック: '{filepath}' は blacklist 設定で禁止されています"

        return None

    def check_window(self, keyword: str) -> Optional[str]:
        """ウィンドウタイトル検索キーワードが禁止されているか判定する。"""
        lower = keyword.lower()
        for pattern in self._user_windows:
            if fnmatch(lower, pattern):
                return f"ブロック: '{keyword}' は blacklist 設定で禁止されたウィンドウパターンです"
        return None

    def check_text(self, text: str) -> Optional[str]:
        """入力テキストに禁止パターンが含まれているか判定する。"""
        lower = text.lower()
        for pattern in self._user_text_patterns:
            if pattern in lower:
                return f"ブロック: 入力テキストに禁止パターン '{pattern}' が含まれています"
        return None

    # ------------------------------------------------------------------
    # 情報表示
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """現在のブラックリスト設定をテキストで返す。"""
        lines = []
        lines.append(f"設定ファイル: {self._config_path or '(なし)'}")
        lines.append("")

        # 禁止キー
        all_keys = sorted(_BUILTIN_FORBIDDEN_KEYS | self._user_keys)
        lines.append(f"禁止キー ({len(all_keys)}):")
        for k in all_keys:
            source = "組み込み" if k in _BUILTIN_FORBIDDEN_KEYS else "ユーザー設定"
            lines.append(f"  {k}  [{source}]")

        # 禁止パス
        all_paths = _BUILTIN_FORBIDDEN_PATHS + self._user_paths
        lines.append(f"\n禁止パス ({len(all_paths)}):")
        for i, p in enumerate(all_paths):
            source = "組み込み" if i < len(_BUILTIN_FORBIDDEN_PATHS) else "ユーザー設定"
            lines.append(f"  {p}  [{source}]")

        # 禁止ウィンドウ
        if self._user_windows:
            lines.append(f"\n禁止ウィンドウ ({len(self._user_windows)}):")
            for w in self._user_windows:
                lines.append(f"  {w}")

        # 禁止テキスト
        if self._user_text_patterns:
            lines.append(f"\n禁止テキストパターン ({len(self._user_text_patterns)}):")
            for t in self._user_text_patterns:
                lines.append(f"  {t}")

        return "\n".join(lines)
