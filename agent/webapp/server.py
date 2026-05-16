"""
Video2AI Desktop Agent - FastAPI Web Server

ローカルPC上でデスクトップアプリをLLM Vision + マウス/キーボードで操作する
スタンドアロンWebアプリのバックエンド。

エンドポイント:
    GET  /              - フロントエンド配信
    GET  /api/status    - エージェント状態
    POST /api/session/start  - 自律エージェント開始
    POST /api/session/stop   - 停止
    POST /api/session/pause  - 一時停止/再開
    POST /api/action/*       - 手動操作 (click/type/keypress/scroll)
    WS   /ws                 - リアルタイム配信
"""
import asyncio
import json
import time
import uuid
import os
import sys
import logging
import traceback
import threading
from pathlib import Path
from typing import Optional
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.screen_capture import ScreenCapture, draw_grid_overlay, grid_cell_to_image_coords
from agent.working_memory import WorkingMemory, MemoryChunk
from agent.long_term_memory import LongTermMemory
from agent.mental_canvas import MentalCanvas, CanvasManager
from agent.mental_image import MentalImage
from agent.ocr_windows import ocr_image
from agent.change_detector import ChangeDetector
from agent.action_executor import ActionExecutor, ActionResult
from agent.safety_rules import SafetyRuleChecker, SafetyDecision
from agent.session_logger import SessionLogger
from agent.dialectic import DialecticalReasoner
from agent.system2 import (Gatekeeper, System2Loop, System2Result,
                            ThinkingMode,
                            ExperienceCache, PredictionErrorMonitor,
                            ThinkingDepthController, PerformanceTracker)

logger = logging.getLogger("video2ai.webapp")


def _detect_primary_monitor() -> int:
    """Windowsのプライマリモニター(左上が0,0)のmssインデックスを返す。"""
    try:
        import mss
        with mss.mss() as sct:
            for i, m in enumerate(sct.monitors):
                if i == 0:
                    continue  # index 0 は全モニター結合
                if m["left"] == 0 and m["top"] == 0:
                    return i
            return 1  # fallback
    except Exception:
        return 1


def _minimize_browser():
    """Webアプリを表示しているブラウザウィンドウを最小化する（フォーカス奪取防止）。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import ctypes.wintypes
        EnumWindows = ctypes.windll.user32.EnumWindows
        GetWindowTextW = ctypes.windll.user32.GetWindowTextW
        ShowWindow = ctypes.windll.user32.ShowWindow
        SW_MINIMIZE = 6
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

        def callback(hwnd, _):
            title = ctypes.create_unicode_buffer(256)
            GetWindowTextW(hwnd, title, 256)
            t = title.value
            # Video2AI Desktop Agent のブラウザタブを検出
            if "Video2AI" in t and ("Edge" in t or "Chrome" in t or "Firefox" in t or "localhost" in t):
                ShowWindow(hwnd, SW_MINIMIZE)
            return True

        EnumWindows(WNDENUMPROC(callback), 0)
    except Exception as e:
        logger.warning(f"Failed to minimize browser: {e}")


def _get_foreground_hwnd() -> int:
    """現在のフォアグラウンドウィンドウのHWNDを返す（0=取得失敗）。"""
    if sys.platform != "win32":
        return 0
    try:
        import ctypes
        return ctypes.windll.user32.GetForegroundWindow() or 0
    except Exception:
        return 0


def _activate_hwnd(hwnd: int) -> bool:
    """指定したHWNDのウィンドウをフォアグラウンドにする。
    フォーカス取得後に検証し、失敗時はリトライする（最大3回）。
    """
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        if not user32.IsWindow(hwnd):
            return False
        KEYEVENTF_EXTENDEDKEY = 0x0001
        KEYEVENTF_KEYUP = 0x0002
        VK_MENU = 0x12

        for attempt in range(3):
            user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY, 0)
            user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.05)
            # 検証: 実際にフォーカスが取れたか
            fg = user32.GetForegroundWindow()
            if fg == hwnd:
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, title, 256)
                logger.info(f"Activated hwnd: {title.value} (attempt {attempt+1})")
                return True
            time.sleep(0.1)

        logger.warning(f"Failed to activate hwnd {hwnd} after 3 attempts (current FG={fg})")
        return False
    except Exception as e:
        logger.warning(f"Failed to activate hwnd {hwnd}: {e}")
        return False


def _list_visible_windows() -> list[str]:
    """表示中のウィンドウタイトル一覧を返す（LLMへのコンテキスト用）。"""
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

        exclude = {"", "program manager", "settings", "microsoft text input application",
                   "windows input experience", "msctfime ui", "default ime"}
        titles = []

        def callback(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            title = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title, 256)
            t = title.value.strip()
            if t and t.lower() not in exclude and len(t) >= 3:
                titles.append(t)
            return True

        user32.EnumWindows(WNDENUMPROC(callback), 0)
        return titles
    except Exception:
        return []


def _detect_window_monitor(hwnd: int) -> int:
    """ウィンドウが存在するモニターのmssインデックスを返す（0=不明）。"""
    if sys.platform != "win32" or not hwnd:
        return 0
    try:
        import ctypes
        import ctypes.wintypes
        import mss

        user32 = ctypes.windll.user32

        # ウィンドウが最小化されている場合は復元してから座標を取得
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            time.sleep(0.3)

        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        win_cx = (rect.left + rect.right) // 2
        win_cy = (rect.top + rect.bottom) // 2

        # 最小化ウィンドウの座標 (-32000, -32000) を検出
        if win_cx < -10000 or win_cy < -10000:
            return 0

        with mss.mss() as sct:
            for i, mon in enumerate(sct.monitors):
                if i == 0:
                    continue  # index 0 は全モニター結合
                if (mon["left"] <= win_cx < mon["left"] + mon["width"] and
                        mon["top"] <= win_cy < mon["top"] + mon["height"]):
                    return i
        return 0
    except Exception:
        return 0


def _screen_coords_to_monitor(x: int, y: int) -> int:
    """スクリーン絶対座標がどのモニターに属するかを返す（0=不明）。"""
    try:
        import mss
        with mss.mss() as sct:
            for i, mon in enumerate(sct.monitors):
                if i == 0:
                    continue
                if (mon["left"] <= x < mon["left"] + mon["width"] and
                        mon["top"] <= y < mon["top"] + mon["height"]):
                    return i
        return 0
    except Exception:
        return 0


def _activate_window_at(x: int, y: int) -> int:
    """指定スクリーン座標のウィンドウを特定し、そのトップレベルウィンドウをアクティブ化する。HWNDを返す（0=失敗）。"""
    if sys.platform != "win32":
        return 0
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        pt = ctypes.wintypes.POINT(x, y)
        child_hwnd = user32.WindowFromPoint(pt)
        if not child_hwnd:
            return 0
        # トップレベルの親ウィンドウを取得
        GA_ROOT = 2
        root_hwnd = user32.GetAncestor(child_hwnd, GA_ROOT) or child_hwnd
        _activate_hwnd(root_hwnd)
        title = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(root_hwnd, title, 256)
        logger.info(f"WindowFromPoint({x},{y}) -> HWND={root_hwnd} '{title.value}'")
        return root_hwnd
    except Exception as e:
        logger.warning(f"WindowFromPoint failed: {e}")
        return 0


def _activate_newest_window() -> bool:
    """最も最近作成された可視ウィンドウをフォアグラウンドにする（アプリ起動直後に使用）。"""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        GetForegroundWindow = user32.GetForegroundWindow
        EnumWindows = user32.EnumWindows
        GetWindowTextW = user32.GetWindowTextW
        IsWindowVisible = user32.IsWindowVisible
        GetWindowThreadProcessId = user32.GetWindowThreadProcessId
        SetForegroundWindow = user32.SetForegroundWindow
        ShowWindow = user32.ShowWindow
        SW_RESTORE = 9
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

        # 現在のフォアグラウンド（コンソール等）と、エージェントのブラウザを除外して
        # Z-orderで最初に見つかる可視ウィンドウをアクティブ化
        current_fg = GetForegroundWindow()
        exclude_titles = ["Video2AI", "PowerShell", "cmd.exe", "Windows PowerShell",
                          "python", "uvicorn", "Terminal"]

        target_hwnd = [None]
        def callback(hwnd, _):
            if hwnd == current_fg:
                return True
            if not IsWindowVisible(hwnd):
                return True
            title = ctypes.create_unicode_buffer(256)
            GetWindowTextW(hwnd, title, 256)
            t = title.value
            if not t or len(t) < 2:
                return True
            # コンソール系・ブラウザ系を除外
            for ex in exclude_titles:
                if ex.lower() in t.lower():
                    return True
            target_hwnd[0] = hwnd
            return False  # 最初に見つかったものを採用

        EnumWindows(WNDENUMPROC(callback), 0)

        if target_hwnd[0]:
            hwnd = target_hwnd[0]
            # Windows制限回避: Altキーを送ってフォアグラウンドロックを解除
            KEYEVENTF_EXTENDEDKEY = 0x0001
            KEYEVENTF_KEYUP = 0x0002
            VK_MENU = 0x12  # Alt key
            ctypes.windll.user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY, 0)
            ctypes.windll.user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
            ShowWindow(hwnd, SW_RESTORE)
            SetForegroundWindow(hwnd)
            # フォーカスも設定
            ctypes.windll.user32.SetFocus(hwnd)
            title = ctypes.create_unicode_buffer(256)
            GetWindowTextW(hwnd, title, 256)
            logger.info(f"Activated window: {title.value}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to activate newest window: {e}")
        return False


def _activate_window_by_title(keywords: list[str]) -> bool:
    """ウィンドウタイトルにキーワードを含むウィンドウをアクティブにする。"""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        import ctypes.wintypes
        EnumWindows = ctypes.windll.user32.EnumWindows
        GetWindowTextW = ctypes.windll.user32.GetWindowTextW
        IsWindowVisible = ctypes.windll.user32.IsWindowVisible
        SetForegroundWindow = ctypes.windll.user32.SetForegroundWindow
        ShowWindow = ctypes.windll.user32.ShowWindow
        SW_RESTORE = 9
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

        found = [False]
        def callback(hwnd, _):
            if not IsWindowVisible(hwnd):
                return True
            title = ctypes.create_unicode_buffer(256)
            GetWindowTextW(hwnd, title, 256)
            t = title.value
            for kw in keywords:
                if kw in t:
                    ShowWindow(hwnd, SW_RESTORE)
                    SetForegroundWindow(hwnd)
                    found[0] = True
                    return False  # stop enumeration
            return True

        EnumWindows(WNDENUMPROC(callback), 0)
        return found[0]
    except Exception as e:
        logger.warning(f"Failed to activate window: {e}")
        return False


def _restore_browser():
    """最小化したブラウザウィンドウを復元する。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import ctypes.wintypes
        EnumWindows = ctypes.windll.user32.EnumWindows
        GetWindowTextW = ctypes.windll.user32.GetWindowTextW
        ShowWindow = ctypes.windll.user32.ShowWindow
        SW_RESTORE = 9
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

        def callback(hwnd, _):
            title = ctypes.create_unicode_buffer(256)
            GetWindowTextW(hwnd, title, 256)
            t = title.value
            if "Video2AI" in t and ("Edge" in t or "Chrome" in t or "Firefox" in t or "localhost" in t):
                ShowWindow(hwnd, SW_RESTORE)
            return True

        EnumWindows(WNDENUMPROC(callback), 0)
    except Exception as e:
        logger.warning(f"Failed to restore browser: {e}")


def _is_app_running(exe_name: str) -> bool:
    """指定のアプリが実行中かどうかを確認する。"""
    import subprocess as sp
    base = exe_name.lower().replace(".exe", "")
    try:
        result = sp.run(["tasklist", "/FI", f"IMAGENAME eq {base}.exe"],
                        capture_output=True, text=True, timeout=5)
        return base.lower() in result.stdout.lower()
    except Exception:
        return False


def _activate_existing_window(exe_name: str):
    """既に起動中のアプリをAlt+Tabで前面に持ってくる。"""
    import pyautogui
    pyautogui.FAILSAFE = False
    # Alt+Tabの方が確実（Windows制約を回避）
    pyautogui.hotkey('alt', 'tab')
    time.sleep(0.5)


def _launch_app(command: str) -> ActionResult:
    """Win+R（ファイル名を指定して実行）でアプリを起動する。確実にフォアグラウンドで開く。"""
    desc = f"アプリ起動: {command}"
    app_map = {
        "notepad": "notepad",
        "メモ帳": "notepad",
        "calculator": "calc",
        "電卓": "calc",
        "explorer": "explorer",
        "paint": "mspaint",
        "ペイント": "mspaint",
        "cmd": "cmd",
        "terminal": "cmd",
        "powershell": "powershell",
    }
    cmd = app_map.get(command.lower().strip().replace(".exe", ""), command.replace(".exe", ""))
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        # Win+R → コマンド入力 → Enter
        pyautogui.hotkey('win', 'r')
        time.sleep(0.8)
        pyautogui.typewrite(cmd, interval=0.03)
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(1.5)
        return ActionResult(True, "launch", desc)
    except Exception as e:
        return ActionResult(False, "launch", desc, error=str(e))

# --- Pydantic Models ---

class SessionStartRequest(BaseModel):
    goal: str = ""
    model: str = "gemini-2.5-flash"
    api_key: str = ""
    api_url: str = ""
    method: str = "diff"
    threshold: Optional[float] = None
    interval: float = 1.0
    dry_run: bool = True
    max_steps: int = 0
    language: str = "ja"
    safety_rules_path: Optional[str] = None
    monitor: int = 0  # 0=auto(primary), 1,2,3...=specific monitor

class ClickRequest(BaseModel):
    x: int
    y: int
    button: int = 1

class TypeRequest(BaseModel):
    text: str

class KeypressRequest(BaseModel):
    key: str

class ScrollRequest(BaseModel):
    x: int = 960
    y: int = 540
    direction: str = "down"
    amount: int = 3

class ConfirmRequest(BaseModel):
    request_id: str
    approved: bool


# --- Agent Session ---

class AgentSession:
    """非同期エージェントセッション。画面キャプチャ→変化検知→LLM解析→操作実行のループ。"""

    def __init__(self):
        self.state: str = "stopped"  # stopped | running | paused
        self.capture: Optional[ScreenCapture] = None
        self.detector: Optional[ChangeDetector] = None
        self.executor: Optional[ActionExecutor] = None
        self.vision = None  # LLMVision (遅延import)
        self.safety: Optional[SafetyRuleChecker] = None
        self.logger: Optional[SessionLogger] = None
        self.goal: str = ""
        self.model: str = ""
        self.interval: float = 1.0
        self.max_steps: int = 0
        self.step: int = 0
        self._last_action_step: int = -1
        self._needs_hires_next: bool = False
        self._last_llm_call: float = 0.0
        self.monitor_index: int = 1
        self.screen_size: tuple = (1920, 1080)
        self.prev_analysis: str = ""
        self._task: Optional[asyncio.Task] = None
        self._confirm_futures: dict[str, asyncio.Future] = {}
        self._action_log: list[dict] = []
        self._last_target_hwnd: int = 0  # 直前のclick/keypressで取得したフォアグラウンドHWND
        self.working_memory: WorkingMemory = WorkingMemory(capacity=7)
        self.canvas_mgr: CanvasManager = CanvasManager()
        self.mental_image: MentalImage = MentalImage()
        self._last_click_no_change: bool = False  # 直前のクリックで画面変化がなかったか
        self._dialectic = DialecticalReasoner()
        self._pending_dialectic: str = ""  # 次回LLMコールに注入する弁証法コンテキスト
        self._gatekeeper = Gatekeeper()
        self._system2 = System2Loop(
            self.working_memory, self.canvas_mgr, self._dialectic)
        self._experience_cache = ExperienceCache()
        self._prediction_monitor = PredictionErrorMonitor()
        self._performance_tracker = PerformanceTracker()
        self._current_task_signature: str = ""  # 現在のタスク署名（経験キャッシュ用）
        # System 2 非同期実行用
        self._s2_task: Optional[asyncio.Task] = None
        self._s2_pending_result: Optional[System2Result] = None
        self._s2_trigger_step: int = 0
        self._ocr_enabled: bool = True  # OCR 補助を有効にするか
        self._last_ocr_text: str = ""   # 直近の OCR 結果
        self._long_term_memory = LongTermMemory()  # 長期記憶（操作経験の永続化）

    # ---- 長期記憶ヘルパー ----

    def _detect_current_app(self) -> str:
        """フォアグラウンドウィンドウのタイトルからアプリ名を推定する。"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = self._last_target_hwnd or user32.GetForegroundWindow()
            if not hwnd:
                return ""
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            title = buf.value
            if not title:
                return ""
            # タイトルから一般的なアプリ名を抽出
            # "document.txt - メモ帳" → "メモ帳"
            # "Sheet1 - Excel" → "Excel"
            parts = title.rsplit(" - ", 1)
            return parts[-1].strip() if len(parts) > 1 else title.strip()
        except Exception:
            return ""

    # ---- app_knowledge 自動読み込み ----

    _app_knowledge_cache: dict = {}  # app_name → (content, mtime) キャッシュ

    def _load_app_knowledge(self, app_name: str) -> str:
        """アプリ名に対応する app_knowledge ファイルを読み込む。

        D:\\ClaudeProject\\app_knowledge\\ 内の .md ファイルを検索し、
        アプリ名がファイル名またはファイル内容に含まれるものを返す。
        結果はキャッシュされ、ファイル変更時のみ再読み込みする。
        LLM トークン節約のため、最大2000文字に制限。
        """
        if not app_name:
            return ""
        app_lower = app_name.lower()

        # キャッシュチェック
        if app_lower in self._app_knowledge_cache:
            cached_content, cached_mtime, cached_path = self._app_knowledge_cache[app_lower]
            try:
                current_mtime = cached_path.stat().st_mtime
                if current_mtime == cached_mtime:
                    return cached_content
            except Exception:
                pass

        # app_knowledge ディレクトリを検索
        from pathlib import Path
        knowledge_dir = Path("D:/ClaudeProject/app_knowledge")
        if not knowledge_dir.exists():
            return ""

        # ファイル名 + ファイル先頭行（タイトル）でマッチング
        best_match = None
        for md_file in knowledge_dir.glob("*.md"):
            if md_file.name.lower() == "readme.md":
                continue
            fname = md_file.stem.lower().replace("_", " ").replace("-", " ")
            # ファイル名マッチ
            if (app_lower in fname or fname in app_lower or
                    any(word in fname for word in app_lower.split()
                        if len(word) > 2)):
                best_match = md_file
                break
            # ファイル先頭（タイトル行）にアプリ名が含まれるかチェック
            try:
                head = md_file.read_text(encoding="utf-8")[:200].lower()
                if app_lower in head:
                    best_match = md_file
                    break
            except Exception:
                continue

        if not best_match:
            return ""

        try:
            content = best_match.read_text(encoding="utf-8")
            # トークン節約: 最大2000文字
            if len(content) > 2000:
                content = content[:2000] + "\n...(省略)"
            formatted = f"\n\n■ アプリ操作ガイド（{app_name}）\n{content}"
            mtime = best_match.stat().st_mtime
            self._app_knowledge_cache[app_lower] = (formatted, mtime, best_match)
            return formatted
        except Exception:
            return ""

    # ---- モニター切替 ----

    async def _check_and_switch_monitor(self, hwnd: int, capture_request, capture_response, _broadcast):
        """
        ウィンドウが現在のキャプチャモニターと異なるモニターにある場合、
        キャプチャ先を切り替える。
        """
        if not hwnd:
            return
        win_mon = await asyncio.to_thread(_detect_window_monitor, hwnd)
        if win_mon > 0 and win_mon != self.monitor_index:
            capture_request.put(("switch_monitor", win_mon))
            try:
                resp = await asyncio.to_thread(capture_response.get, True, 10.0)
                if resp[0] == "screen_size":
                    old_mon = self.monitor_index
                    self.monitor_index = win_mon
                    self.screen_size = resp[1]
                    await _broadcast({"type": "log", "message":
                        f"キャプチャモニターを切替: Monitor {old_mon} → {win_mon} ({resp[1][0]}x{resp[1][1]})"})
                    self._needs_hires_next = True
                elif resp[0] == "error":
                    await _broadcast({"type": "log", "message": f"モニター切替失敗: {resp[1]}"})
            except Exception as e:
                await _broadcast({"type": "log", "message": f"モニター切替タイムアウト: {e}"})

    # ---- 弁証法的検証（自動） ----

    # ファイル操作系のキーパターン
    _RISKY_KEYS = {"ctrl+s", "ctrl+shift+s", "ctrl+n", "delete", "ctrl+x", "ctrl+z"}

    def _should_dialectic_check(self, suggestion) -> bool:
        """このアクションに弁証法的検証が必要かどうかを判定する。"""
        # 低確信度
        if suggestion.confidence < 0.7 and suggestion.action_type not in ("wait", "none"):
            return True
        # ファイル操作系のキー入力
        if suggestion.action_type == "keypress":
            key = suggestion.value.lower().replace(" ", "")
            if key in self._RISKY_KEYS:
                return True
        # ステップ1（初回アクション）
        if self.step == 1:
            return True
        return False

    def _generate_antithesis(self, suggestion, analysis_desc: str) -> str:
        """アクション提案に対する機械的なアンチテーゼを生成する。"""
        challenges = []

        # 1. 作業記憶との矛盾チェック
        wm_text = self.working_memory.recall()
        if wm_text:
            # 同じアクションの繰り返し検出
            recent_actions = [line for line in wm_text.split("\n") if suggestion.action_type in line]
            if len(recent_actions) >= 2:
                challenges.append(
                    f"同じ種類の操作（{suggestion.action_type}）が直近で{len(recent_actions)}回実行されている。"
                    f"繰り返しは効果がない可能性がある。別のアプローチを検討すべきではないか？")

            # 直前に失敗した操作の再試行
            if "失敗" in wm_text.split("\n")[-1] if wm_text.split("\n") else "":
                challenges.append(
                    "直前の操作が失敗している。同じ方法で再試行するのは適切か？"
                    "原因を特定してから別の手段を試すべきではないか？")

        # 2. 低確信度への疑問
        if suggestion.confidence < 0.7:
            challenges.append(
                f"確信度が{suggestion.confidence:.0%}と低い。"
                f"対象「{suggestion.target}」が本当に正しいか確認すべきではないか？")

        # 3. クリック系の座標リスク
        if suggestion.action_type in ("click", "double_click"):
            challenges.append(
                f"クリック対象「{suggestion.target}」({suggestion.grid_cell})は本当にその要素か？"
                f"キーボード操作（Tab, ショートカット）で代替できないか？")

        # 4. ファイル操作のリスク
        if suggestion.action_type == "keypress" and suggestion.value.lower().replace(" ", "") in self._RISKY_KEYS:
            challenges.append(
                f"ファイル操作キー「{suggestion.value}」を実行しようとしている。"
                f"保存先や対象ファイルは正しいか？意図しないファイルを上書きするリスクはないか？")

        if not challenges:
            challenges.append("このアクションが目標達成の最短経路か？別のより効率的な方法はないか？")

        return " また、".join(challenges)

    async def _dialectic_check(self, suggestion, analysis) -> None:
        """
        重要なアクションの前に弁証法的検証を実行する。
        テーゼ（LLM提案）とアンチテーゼ（機械生成）を記録し、
        次回の LLM コールのコンテキストに注入する。
        """
        if not self._should_dialectic_check(suggestion):
            self._pending_dialectic = ""
            return

        session_name = f"step_{self.step}"
        thesis = f"{suggestion.action_type}({suggestion.target}): {suggestion.reasoning}"
        antithesis = self._generate_antithesis(suggestion, analysis.description)

        # 弁証法セッションを記録
        self._dialectic.start(session_name, goal=self.goal, thesis=thesis)
        self._dialectic.challenge(session_name, antithesis)

        # 次回の LLM コールに注入するテキストを構築
        self._pending_dialectic = (
            f"\n■ 弁証法的検証（前ステップの提案に対する反論）\n"
            f"テーゼ: {thesis}\n"
            f"アンチテーゼ: {antithesis}\n"
            f"→ この反論を踏まえた上で、最適なアクションを判断してください。"
        )

        await _broadcast({"type": "log",
            "message": f"🔄 弁証法的検証: {antithesis[:100]}..."})

        # セッションをクリーンアップ（古いものを削除）
        if len(self._dialectic) > 10:
            sessions = self._dialectic.list_sessions()
            for s in sessions[:5]:
                self._dialectic.delete(s["name"])

    # ---- 操作計画の可視化 ----

    def _visualize_plan(self, planned_steps, current_action, step: int) -> Optional[bytes]:
        """
        操作計画を脳内キャンバスに可視化する。
        現在のアクション（赤マーカー）+ 将来のステップ（青マーカー＋矢印）を描画。
        """
        if not planned_steps and not (current_action and current_action.grid_cell):
            return None

        try:
            canvas = self.canvas_mgr.from_screenshot("_plan", monitor=self.monitor_index)

            # 座標リスト（矢印の描画用）
            points = []

            # 現在のアクション（赤）
            if current_action and current_action.grid_cell:
                cx, cy = grid_cell_to_image_coords(
                    current_action.grid_cell,
                    self.screen_size[0], self.screen_size[1])
                if cx is not None:
                    canvas.draw_marker(cx, cy, label=f"NOW: {current_action.action_type}",
                                       color="red", size=25)
                    points.append((cx, cy))

            # 計画ステップ（青）
            for ps in planned_steps:
                if ps.grid_cell:
                    gc = ps.grid_cell.strip().upper()
                    if len(gc) >= 2 and gc[0].isalpha():
                        px, py = grid_cell_to_image_coords(
                            gc, self.screen_size[0], self.screen_size[1])
                        if px is not None:
                            label = f"{ps.step}: {ps.action}"
                            if ps.value:
                                label += f"({ps.value[:15]})"
                            canvas.draw_marker(px, py, label=label, color="blue", size=20)
                            points.append((px, py))

                # テキスト/キー操作はグリッドセルがないので、テキストラベルで表示
                elif ps.action in ("type", "keypress"):
                    label = f"Step {ps.step}: {ps.action}({ps.value[:20]})"
                    canvas.draw_text(10, 30 + ps.step * 25, label, color="cyan", size=14)

            # 連続するクリック系操作間に矢印を描画
            for i in range(len(points) - 1):
                canvas.draw_arrow(points[i][0], points[i][1],
                                  points[i+1][0], points[i+1][1],
                                  color="yellow", width=2)

            plan_bytes = canvas.to_bytes(format="JPEG")
            self.canvas_mgr.delete("_plan")
            return plan_bytes
        except Exception:
            return None

    # ---- 脳内シミュレーション ----

    def _simulate_click_on_canvas(self, x: int, y: int, grid_cell: str, step: int) -> MentalCanvas:
        """
        クリック前の脳内シミュレーション。
        画面をキャプチャしてキャンバスに取り込み、クリック予定位置にマーカーを描画。
        スナップショットとして保存し、実行後の検証に使う。
        """
        canvas_name = f"click_sim_{step}"
        canvas = self.canvas_mgr.from_screenshot(canvas_name, monitor=self.monitor_index)
        label = grid_cell if grid_cell else f"({x},{y})"
        canvas.draw_marker(x, y, label=label, color="red", size=20)
        canvas.snapshot("before_click")
        return canvas

    def _verify_click_with_canvas(self, x: int, y: int, step: int) -> str:
        """
        クリック後の検証。
        1. カーソル位置が意図した座標に合っているか確認
        2. ずれていれば補正
        3. クリック前後の画面変化を比較
        """
        # 位置検証 + 補正
        vr = self.mental_image.verify_position(x, y)
        if not vr.position_ok:
            corrected = self.mental_image.correct_position(x, y)
            if corrected:
                vr.assessment += " → 補正済み"

        # 画面変化の検証（キャンバスのスナップショット vs 現在の画面）
        canvas_name = f"click_sim_{step}"
        canvas = self.canvas_mgr.get(canvas_name)
        context_info = ""
        if canvas:
            # 現在の画面をキャプチャして比較用キャンバスに
            after_canvas = self.canvas_mgr.from_screenshot(
                f"click_after_{step}", monitor=self.monitor_index)
            # マーカーなしの元画像で比較するため、before スナップショットを復元
            canvas.restore("before_click")
            result = canvas.compare_with(after_canvas)
            context_info = f" / {result.summary}"
            # 使い終わったキャンバスを削除
            self.canvas_mgr.delete(f"click_after_{step}")
            self.canvas_mgr.delete(canvas_name)

        return f"{vr.assessment}{context_info}"

    async def start(self, config: SessionStartRequest):
        if self.state == "running":
            raise RuntimeError("Session already running")

        session_id = str(uuid.uuid4())[:8]

        # API Key を環境変数にセット（LLMVisionが参照）
        if config.api_key:
            os.environ["OPENAI_API_KEY"] = config.api_key
        if config.api_url:
            os.environ["OPENAI_API_BASE"] = config.api_url

        # モニター選択（0=プライマリ自動検出）
        self.monitor_index = config.monitor if config.monitor > 0 else _detect_primary_monitor()

        # コンポーネント初期化
        self.capture = None  # キャプチャスレッド内で初期化
        self.detector = ChangeDetector(method=config.method, threshold=config.threshold)
        self.executor = ActionExecutor(dry_run=config.dry_run)
        self.goal = config.goal
        self.model = config.model
        self.interval = config.interval
        self.max_steps = config.max_steps
        self.step = 0
        self.prev_analysis = ""
        self._action_log = []
        self.working_memory.clear()
        self.canvas_mgr.clear_all()
        self._last_click_no_change = False

        # Safety rules
        if config.safety_rules_path:
            self.safety = SafetyRuleChecker(config.safety_rules_path)
        else:
            self.safety = None

        # LLM Vision (API keyが必要)
        try:
            from agent.llm_vision import LLMVision
            self.vision = LLMVision(
                model=config.model,
                api_key=config.api_key or None,
                api_url=config.api_url or None,
                language=config.language,
            )
        except Exception as e:
            self.vision = None
            await _broadcast({"type": "log", "message": f"LLM Vision初期化失敗: {e}. 画面配信のみ実行します。"})

        # Session Logger
        output_dir = str(Path.home() / "video2ai_agent_sessions")
        self.logger = SessionLogger(
            session_id=session_id,
            output_dir=output_dir,
            goal=config.goal,
            model=config.model,
            method=config.method,
            threshold=self.detector.threshold,
        )

        # Screen capture はスレッド内で初期化するため、ここでは開始しない
        # screen_size は _run_loop 内で取得
        self.state = "running"
        self._task = asyncio.create_task(self._run_loop())

        await _broadcast({"type": "status", "state": "running", "session_id": session_id,
                          "screen_size": list(self.screen_size)})
        await _broadcast({"type": "log", "message": f"セッション開始: goal='{config.goal}' model={config.model}"})


    async def stop(self):
        self.state = "stopped"
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._cleanup()
        await _broadcast({"type": "status", "state": "stopped"})

    def pause_toggle(self):
        if self.state == "running":
            self.state = "paused"
        elif self.state == "paused":
            self.state = "running"

    def _cleanup(self):
        self.capture = None
        summary = None
        if self.logger:
            summary = self.logger.end_session()
        self.state = "stopped"
        return summary

    async def _auto_launch_if_needed(self):
        """ゴールにアプリ起動指示が含まれる場合、自動でアプリを起動する。"""
        import re
        goal_lower = self.goal.lower()
        # アプリ名→コマンドのマッピング
        app_patterns = {
            r"メモ帳|notepad": "notepad.exe",
            r"電卓|calculator|calc": "calc.exe",
            r"ペイント|paint": "mspaint.exe",
            r"エクスプローラー|explorer": "explorer.exe",
            r"word": "winword.exe",
            r"excel": "excel.exe",
            r"powerpoint|パワーポイント": "powerpnt.exe",
            r"chrome": "chrome.exe",
            r"edge": "msedge.exe",
            r"firefox": "firefox.exe",
            r"cmd|コマンドプロンプト": "cmd.exe",
            r"powershell": "powershell.exe",
            r"terminal|ターミナル": "wt.exe",
        }
        # 「開く」「起動」「開いて」等のキーワードがあるか
        open_keywords = r"開[くいけ]|開いて|起動|launch|open|start"
        if not re.search(open_keywords, goal_lower):
            return
        for pattern, cmd in app_patterns.items():
            if re.search(pattern, goal_lower, re.IGNORECASE):
                # 既に起動中か確認
                if _is_app_running(cmd):
                    await _broadcast({"type": "log", "message": f"{cmd} は既に起動中。フォーカスを移動します。"})
                    await asyncio.to_thread(_activate_existing_window, cmd)
                    await asyncio.sleep(1.0)
                else:
                    await _broadcast({"type": "log", "message": f"アプリ自動起動: {cmd}"})
                    result = await asyncio.to_thread(_launch_app, cmd)
                    if result.success:
                        await _broadcast({"type": "action", "action_type": "launch",
                                          "description": result.description, "success": True,
                                          "timestamp": time.time()})
                        await asyncio.sleep(0.5)
                    else:
                        await _broadcast({"type": "error", "message": f"アプリ起動失敗: {result.error}"})
                return

    async def _run_loop(self):
        """エージェントメインループ。キャプチャは専用スレッドで実行。"""
        import queue as stdlib_queue

        # キャプチャ専用スレッド: mss はスレッドローカルなので同一スレッドで生成・使用する
        # リクエスト: "stop" | ("capture", llm_width) | ("switch_monitor", index)
        capture_request = stdlib_queue.Queue()
        # レスポンス: ("screen_size", sz) | ("frame", bgr, preview_b64, llm_b64) | ("error", msg)
        capture_response = stdlib_queue.Queue()

        def _capture_thread():
            import cv2
            for _ in range(20):
                try:
                    cap = ScreenCapture(monitor_index=self.monitor_index, resize_width=0)
                    cap.__enter__()
                    break
                except Exception as e:
                    logger.warning(f"Capture thread init retry: {e}")
                    time.sleep(0.5)
            else:
                capture_response.put(("error", "Failed to init ScreenCapture"))
                return
            try:
                sz = cap.get_screen_size()
                capture_response.put(("screen_size", sz))
                logger.info(f"Capture thread started: {sz}")
                while True:
                    req = capture_request.get()
                    if req == "stop":
                        break
                    if req[0] == "switch_monitor":
                        new_index = req[1]
                        old_cap = cap
                        try:
                            cap.__exit__(None, None, None)
                            cap = ScreenCapture(monitor_index=new_index, resize_width=0)
                            cap.__enter__()
                            new_sz = cap.get_screen_size()
                            capture_response.put(("screen_size", new_sz))
                            logger.info(f"Switched to monitor {new_index}: {new_sz}")
                        except Exception as e:
                            # 切替失敗: 元のモニターに戻す
                            logger.warning(f"Monitor switch failed, reverting: {e}")
                            try:
                                cap = ScreenCapture(monitor_index=self.monitor_index, resize_width=0)
                                cap.__enter__()
                            except Exception:
                                pass  # リカバリも失敗 → 次のキャプチャリクエストで再試行
                            capture_response.put(("error", f"Monitor switch failed: {e}"))
                        continue
                    _, llm_width = req  # ("capture", width)
                    try:
                        frame = cap.capture()  # フル解像度BGR

                        # プレビュー用: 960px, quality 40
                        h, w = frame.shape[:2]
                        pw = 960
                        ph = int(h * pw / w)
                        preview_rgb = cv2.cvtColor(
                            cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2RGB)
                        from PIL import Image
                        from io import BytesIO
                        import base64
                        buf = BytesIO()
                        Image.fromarray(preview_rgb).save(buf, format="JPEG", quality=40)
                        preview_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

                        # LLM用: 指定幅にリサイズ (0=フル解像度)
                        if llm_width > 0 and w > llm_width:
                            lh = int(h * llm_width / w)
                            llm_frame = cv2.resize(frame, (llm_width, lh), interpolation=cv2.INTER_AREA)
                        else:
                            llm_frame = frame.copy()

                        # マウスカーソル位置にマーカーを描画（LLMがカーソル位置を認識できるように）
                        try:
                            import pyautogui
                            mx, my = pyautogui.position()
                            # 実画面座標 → LLM画像座標に変換
                            lh_actual, lw_actual = llm_frame.shape[:2]
                            marker_x = int(mx * lw_actual / w)
                            marker_y = int(my * lh_actual / h)
                            # 赤い十字マーカー + 円を描画
                            color = (0, 0, 255)  # BGR: 赤
                            cv2.circle(llm_frame, (marker_x, marker_y), 12, color, 2)
                            cv2.line(llm_frame, (marker_x - 18, marker_y), (marker_x + 18, marker_y), color, 2)
                            cv2.line(llm_frame, (marker_x, marker_y - 18), (marker_x, marker_y + 18), color, 2)
                        except Exception:
                            pass  # カーソル位置取得失敗時はスキップ

                        # グリッドオーバーレイを描画（LLMがセル名で位置指定できるように）
                        llm_frame = draw_grid_overlay(llm_frame)

                        llm_rgb = cv2.cvtColor(llm_frame, cv2.COLOR_BGR2RGB)
                        buf2 = BytesIO()
                        Image.fromarray(llm_rgb).save(buf2, format="JPEG", quality=75)
                        llm_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf2.getvalue()).decode()}"

                        capture_response.put(("frame", frame, preview_b64, llm_b64))
                    except Exception as e:
                        logger.warning(f"Capture error: {e}")
                        capture_response.put(("error", str(e)))
            finally:
                cap.__exit__(None, None, None)
                logger.info("Capture thread stopped")

        cap_thread = threading.Thread(target=_capture_thread, daemon=True)
        cap_thread.start()

        # screen_size を受け取る
        try:
            resp = await asyncio.to_thread(capture_response.get, True, 15.0)
        except Exception as e:
            await _broadcast({"type": "error", "message": f"キャプチャスレッド起動失敗: {e}"})
            self.state = "stopped"
            return

        if resp[0] == "screen_size":
            self.screen_size = resp[1]
            await _broadcast({"type": "log", "message": f"画面キャプチャ開始: {resp[1][0]}x{resp[1][1]}"})
        elif resp[0] == "error":
            await _broadcast({"type": "error", "message": f"キャプチャ初期化エラー: {resp[1]}"})
            self.state = "stopped"
            return

        # 実行中ウィンドウ一覧をLLMコンテキストに追加（隠れたウィンドウも検出）
        window_titles = await asyncio.to_thread(_list_visible_windows)
        if window_titles:
            win_list = "\n".join(f"  - {t}" for t in window_titles[:20])
            self.prev_analysis = f"[システム情報] 現在実行中のウィンドウ:\n{win_list}\n※スクリーンショットに映っていなくても、上記ウィンドウは起動中です。focus_windowで切り替え可能です。"
            await _broadcast({"type": "log", "message": f"実行中ウィンドウ: {len(window_titles)}件検出"})

        # 注: アプリ起動はLLMが画面上で操作して行う（自動起動は別モニターに開く問題があるため廃止）

        # Escapeキー検知スレッド（緊急停止用）
        _escape_pressed = threading.Event()
        def _escape_listener():
            """Escapeキーが押されたらイベントをセットする。"""
            if sys.platform == "win32":
                try:
                    import msvcrt
                    while not _escape_pressed.is_set() and self.state == "running":
                        if msvcrt.kbhit():
                            key = msvcrt.getch()
                            if key == b'\x1b':  # Escape
                                _escape_pressed.set()
                                return
                        time.sleep(0.1)
                except Exception:
                    pass
        escape_thread = threading.Thread(target=_escape_listener, daemon=True)
        escape_thread.start()

        await _broadcast({"type": "log", "message": "⚠ エージェントがマウス・キーボードを制御中です。Escapeキーで緊急停止できます。"})

        try:
            while self.state != "stopped":
                # Escapeキー検知
                if _escape_pressed.is_set():
                    await _broadcast({"type": "log", "message": "Escapeキーが押されました。エージェントを停止します。"})
                    break

                if self.state == "paused":
                    await asyncio.sleep(0.3)
                    continue

                if self.max_steps > 0 and self.step >= self.max_steps:
                    await _broadcast({"type": "log", "message": f"最大ステップ数 ({self.max_steps}) に到達"})
                    break

                loop_start = time.time()
                self.step += 1

                if self.step <= 3 or self.step % 10 == 0:
                    await _broadcast({"type": "log", "message": f"Step {self.step}: キャプチャ中..."})

                # 1. キャプチャ（専用スレッドにリクエスト）
                # 通常1920px、精密モード時は2560px（フル解像度だとLLMの座標精度が落ちる）
                need_hires = self._needs_hires_next
                llm_width = 2560 if need_hires else 1920
                self._needs_hires_next = False

                capture_request.put(("capture", llm_width))
                try:
                    resp = await asyncio.to_thread(capture_response.get, True, 5.0)
                except Exception:
                    await _broadcast({"type": "error", "message": "キャプチャタイムアウト"})
                    await asyncio.sleep(self.interval)
                    continue

                if resp[0] == "error":
                    await _broadcast({"type": "error", "message": f"キャプチャエラー: {resp[1]}"})
                    await asyncio.sleep(self.interval)
                    continue

                _, frame, preview_b64, llm_b64 = resp

                # プレビュー配信（軽量960px）
                await _broadcast({
                    "type": "frame",
                    "data": preview_b64,
                    "width": self.screen_size[0],
                    "height": self.screen_size[1],
                    "step": self.step,
                })

                # 2. 変化検知
                # 初回 or 直前にアクション実行後は強制解析（操作結果を確認するため）
                changed, diff_score = self.detector.is_changed(frame)
                force_analyze = (self.step == 1) or \
                                self._last_action_step == self.step - 1 or \
                                self.step % 5 == 0  # 5ステップごとに強制再解析
                if not changed and not force_analyze:
                    elapsed = time.time() - loop_start
                    await asyncio.sleep(max(0.0, self.interval - elapsed))
                    continue
                # force_analyze時も実際のdiff_scoreをそのまま使う（LLMに正確な変化量を伝える）

                # 3. LLM Vision 解析
                if self.vision is None:
                    await _broadcast({"type": "change", "step": self.step,
                                      "diff_score": diff_score, "description": "変化検知 (Vision無効)"})
                    if self.logger:
                        self.logger.log_frame(diff_score=diff_score, changed=True)
                    elapsed = time.time() - loop_start
                    await asyncio.sleep(max(0.0, self.interval - elapsed))
                    continue

                # LLM API レート制限対策: 最低2秒のクールダウン
                # クールダウン中もプレビュー配信を続ける
                now = time.time()
                since_last_llm = now - self._last_llm_call
                if since_last_llm < 2.0:
                    cooldown_remaining = 2.0 - since_last_llm
                    await _broadcast({"type": "log", "message": f"LLMクールダウン中... {cooldown_remaining:.1f}秒"})
                    while cooldown_remaining > 0.5:
                        await asyncio.sleep(0.5)
                        # クールダウン中もフレーム配信
                        capture_request.put(("capture", llm_width))
                        try:
                            cd_resp = await asyncio.to_thread(capture_response.get, True, 3.0)
                            if cd_resp[0] == "frame":
                                await _broadcast({
                                    "type": "frame", "data": cd_resp[2],
                                    "width": self.screen_size[0], "height": self.screen_size[1],
                                    "step": self.step,
                                })
                        except Exception:
                            pass
                        cooldown_remaining -= 0.5

                res_label = "FULL" if llm_width == 0 else f"{llm_width}px"
                await _broadcast({"type": "log", "message": f"Step {self.step}: 変化検知 score={diff_score:.4f} → LLM解析中 ({res_label})..."})

                try:
                    # 429リトライ（最大3回、指数バックオフ）
                    analysis = None
                    for retry in range(3):
                        try:
                            self._last_llm_call = time.time()
                            # 直近のアクション履歴をコンテキストとして構築
                            recent_actions = self._action_log[-8:]  # 直近8件
                            history_lines = []
                            for a in recent_actions:
                                step_n = a.get("step", "?")
                                a_type = a.get("action_type", "")
                                a_desc = a.get("description", "")
                                a_ok = "成功" if a.get("success") else "失敗"
                                a_score = a.get("diff_score")
                                score_info = f" 変化={a_score:.4f}" if a_score is not None else ""
                                history_lines.append(f"Step{step_n}: {a_type} - {a_desc} [{a_ok}]{score_info}")
                            history_text = "\n".join(history_lines)

                            # 繰り返し検出 + 画面変化なし検出
                            repeat_warning = ""
                            if self._last_click_no_change:
                                repeat_warning = ("\n\n⚠️ 重要: 直前のクリックで画面に変化がありませんでした。"
                                                  "クリックは成功しフォーカスは取得済みです。"
                                                  "次は絶対にclickを選ばず、type（テキスト入力）または keypress（キー入力）を実行してください。")
                            elif len(self._action_log) >= 2:
                                last2 = [a.get("action_type") for a in self._action_log[-2:]]
                                if last2[0] == last2[1]:
                                    repeated = last2[0]
                                    if repeated == "click":
                                        repeat_warning = "\n\n⚠️ 警告: clickを2回繰り返しています。クリック先は既にフォーカス済みです。次は必ず type（テキスト入力）または keypress（キー入力）を実行してください。clickは絶対に選ばないでください。"
                                    elif repeated == "keypress":
                                        repeat_warning = "\n\n⚠️ 警告: 同じkeypressを2回繰り返しています。同じキー操作は効果がありません。別のアプローチを試してください（type, click, wait等）。"
                                    else:
                                        repeat_warning = f"\n\n⚠️ 警告: {repeated}を2回繰り返しています。別のアクションに切り替えてください。"

                            # OCR 補助: 画面上のテキストを認識してLLMに補助情報として渡す
                            ocr_text = ""
                            if self._ocr_enabled:
                                try:
                                    from PIL import Image as PILImage
                                    from io import BytesIO as _BytesIO
                                    import base64 as _b64
                                    # LLM送信用の画像からOCR（リサイズ済みなので高速）
                                    raw_b64 = llm_b64.split(",", 1)[1] if "," in llm_b64 else llm_b64
                                    ocr_pil = PILImage.open(_BytesIO(_b64.b64decode(raw_b64)))
                                    ocr_text = await asyncio.to_thread(ocr_image, ocr_pil, 10)
                                    if ocr_text:
                                        self._last_ocr_text = ocr_text
                                except Exception:
                                    pass  # OCR失敗は無視して続行

                            context = self.prev_analysis
                            if history_text:
                                context = f"{self.prev_analysis}\n\n--- 実行済みアクション履歴（直近{len(recent_actions)}件）---\n{history_text}{repeat_warning}"
                            if ocr_text:
                                context += f"\n\n--- 画面上のテキスト（OCR）---\n{ocr_text[:500]}"
                            # バックグラウンド System 2 の結果を消費
                            if self._s2_pending_result is not None:
                                s2r = self._s2_pending_result
                                self._s2_pending_result = None
                                staleness = self.step - self._s2_trigger_step
                                self._pending_dialectic = (
                                    f"\n■ System 2 熟慮結果（{staleness}ステップ前に起動）\n"
                                    f"確信度={s2r.confidence:.0%}, "
                                    f"{s2r.iterations}反復, {s2r.exit_reason}\n"
                                    f"{s2r.dialectic_summary}\n"
                                    f"結論: {s2r.reasoning}")
                            # 弁証法的検証テキストの注入（前ステップで生成された場合）
                            if self._pending_dialectic:
                                context += self._pending_dialectic

                            # LLMには画像サイズを座標系として伝える
                            if llm_width > 0:
                                img_h = int(self.screen_size[1] * llm_width / self.screen_size[0])
                                llm_screen_size = (llm_width, img_h)
                            else:
                                llm_screen_size = self.screen_size

                            # 長期記憶 + app_knowledge: フォアグラウンドアプリから関連知識を取得
                            ltm_app = self._detect_current_app()
                            ltm_text = self._long_term_memory.recall_as_prompt(
                                app=ltm_app, limit=3)
                            # app_knowledge: 静的ガイド（ショートカット等）を長期記憶に追記
                            app_knowledge_text = self._load_app_knowledge(ltm_app)
                            if app_knowledge_text:
                                ltm_text += app_knowledge_text

                            analysis = await asyncio.to_thread(
                                self.vision.analyze,
                                llm_b64,
                                self.goal,
                                context,
                                diff_score,
                                llm_screen_size,
                                self.working_memory.recall_as_prompt(),
                                ltm_text,
                            )
                            break
                        except Exception as api_err:
                            if "429" in str(api_err):
                                wait = (retry + 1) * 5  # 5, 10, 15秒
                                await _broadcast({"type": "log", "message": f"API制限中... {wait}秒後にリトライ ({retry+1}/3)"})
                                await asyncio.sleep(wait)
                            else:
                                raise
                    if analysis is None:
                        raise Exception("API rate limit: 3回リトライ失敗")
                    # 直近のアクション履歴を含めて次回のコンテキストにする
                    self.prev_analysis = analysis.description
                    suggestion = analysis.suggested_action

                    # LLMが返す座標は画像座標 → 実画面座標にスケーリング
                    if suggestion.x and suggestion.y and llm_width > 0:
                        scale = self.screen_size[0] / llm_width
                        orig_x, orig_y = suggestion.x, suggestion.y
                        suggestion.x = int(suggestion.x * scale)
                        suggestion.y = int(suggestion.y * scale)
                        # ドラッグ終了座標もスケーリング
                        if suggestion.end_x and suggestion.end_y:
                            suggestion.end_x = int(suggestion.end_x * scale)
                            suggestion.end_y = int(suggestion.end_y * scale)
                        grid_info = f" [grid: {suggestion.grid_cell}]" if suggestion.grid_cell else ""
                        if suggestion.grid_cell_end:
                            grid_info += f"→{suggestion.grid_cell_end}"
                        await _broadcast({"type": "log", "message": f"座標: ({orig_x},{orig_y}) in {llm_width}px → ({suggestion.x},{suggestion.y}) in {self.screen_size[0]}px{grid_info}"})

                    # 操作計画の可視化（planned_steps がある場合）
                    if hasattr(analysis, 'planned_steps') and analysis.planned_steps:
                        plan_bytes = self._visualize_plan(
                            analysis.planned_steps, suggestion, self.step)
                        if plan_bytes:
                            import base64 as b64mod
                            plan_b64 = b64mod.b64encode(plan_bytes).decode()
                            await _broadcast({
                                "type": "plan",
                                "step": self.step,
                                "image": f"data:image/jpeg;base64,{plan_b64}",
                                "steps": [
                                    {"step": ps.step, "action": ps.action,
                                     "target": ps.target, "value": ps.value}
                                    for ps in analysis.planned_steps
                                ],
                            })
                        # 計画テキストを作業記憶に注入
                        plan_text = " → ".join(
                            f"{ps.step}.{ps.action}({ps.target})"
                            for ps in analysis.planned_steps
                        )
                        await _broadcast({"type": "log",
                            "message": f"📋 操作計画: {plan_text}"})

                    change_msg = {
                        "type": "change",
                        "step": self.step,
                        "diff_score": diff_score,
                        "description": analysis.description,
                        "changes": analysis.changes,
                        "action_type": suggestion.action_type,
                        "target": suggestion.target,
                        "value": suggestion.value,
                        "x": suggestion.x,
                        "y": suggestion.y,
                        "grid_cell": suggestion.grid_cell,
                        "confidence": suggestion.confidence,
                        "reasoning": suggestion.reasoning,
                    }
                    await _broadcast(change_msg)

                    # 3.5a. 経験キャッシュルックアップ（S2→S1 降格済みパターン）
                    if not self._current_task_signature and self.goal:
                        self._current_task_signature = ExperienceCache.make_signature(
                            self.goal, suggestion.target)
                        self._experience_cache.begin_recording(self._current_task_signature)

                    cached = self._experience_cache.lookup(self._current_task_signature)
                    if cached:
                        await _broadcast({"type": "log",
                            "message": f"⚡ System 1 キャッシュヒット: "
                                       f"{len(cached.action_sequence)}ステップのパターン再利用 "
                                       f"(成功{cached.success_count}回)"})

                    # 経験キャッシュ: アクション記録
                    self._experience_cache.record_action(
                        self._current_task_signature, {
                            "action_type": suggestion.action_type,
                            "target": suggestion.target,
                            "value": suggestion.value,
                            "grid_cell": suggestion.grid_cell,
                        })

                    # 3.5b. 予測誤差モニター
                    self._prediction_monitor.predict(suggestion.action_type)

                    # 3.5c. System 1/2 ゲートキーパー判定 + System 2 思考ループ
                    gk_ctx = self._gatekeeper.build_context(
                        suggestion, self._action_log, self.working_memory,
                        self.step, diff_score)
                    thinking_mode = self._gatekeeper.judge(gk_ctx)

                    if thinking_mode == ThinkingMode.SYSTEM2:
                        # System 2 をバックグラウンドで起動（メインループをブロックしない）
                        await self._launch_s2_background(
                            suggestion, analysis.description, gk_ctx)
                        # System 1 の軽量チェックで行動継続
                        await self._dialectic_check(suggestion, analysis)
                        self._performance_tracker.record(
                            ThinkingMode.SYSTEM1, True, suggestion.confidence)
                    else:
                        # System 1: 従来の弁証法チェック（軽量版）
                        await self._dialectic_check(suggestion, analysis)
                        self._performance_tracker.record(
                            ThinkingMode.SYSTEM1, True, suggestion.confidence)

                    # 閾値の自動チューニング（20ステップごと）
                    if self.step % 20 == 0 and self.step > 0:
                        new_th = self._performance_tracker.suggest_threshold(
                            self._gatekeeper.CONFIDENCE_THRESHOLD)
                        if new_th != self._gatekeeper.CONFIDENCE_THRESHOLD:
                            old_th = self._gatekeeper.CONFIDENCE_THRESHOLD
                            self._gatekeeper.CONFIDENCE_THRESHOLD = new_th
                            await _broadcast({"type": "log",
                                "message": f"⚙ 閾値自動調整: {old_th:.2f} → {new_th:.2f} "
                                           f"(stats: {self._performance_tracker.get_stats()})"})

                    # 4. クリック繰り返し防止 + 安全チェック + アクション実行
                    # 直前のクリックで画面変化がなかったのにまたクリックを提案 → スキップ
                    if self._last_click_no_change and suggestion.action_type in ("click", "double_click"):
                        await _broadcast({"type": "log", "message":
                            f"⚠️ クリック繰り返し防止: 前回クリック時に画面変化なし → click をスキップ（次ステップで再解析）"})
                        self.working_memory.store(MemoryChunk(
                            step=self.step,
                            action_type="click_blocked",
                            target=suggestion.target,
                            value=suggestion.value,
                            result="スキップ: 前回クリック時に画面変化なし",
                            screen_state=analysis.description[:80],
                            grid_cell=suggestion.grid_cell,
                        ))
                        self._last_click_no_change = False  # リセットして次回は通す
                        elapsed = time.time() - loop_start
                        await asyncio.sleep(max(0.0, self.interval - elapsed))
                        continue

                    # waitは安全なので常に実行、それ以外はconfidence >= 0.5
                    should_act = (suggestion.action_type == "wait") or \
                                 (suggestion.action_type != "none" and suggestion.confidence >= 0.5)
                    if should_act:
                        blocked = False
                        if self.safety:
                            verdict = self.safety.check(suggestion)
                            if verdict.decision == SafetyDecision.DENY:
                                await _broadcast({"type": "blocked", "step": self.step,
                                                  "rule_id": verdict.matched_rule_id, "message": verdict.message})
                                blocked = True
                            elif verdict.decision == SafetyDecision.CONFIRM:
                                approved = await self._request_confirm(suggestion, verdict.message)
                                if not approved:
                                    blocked = True

                        if not blocked:
                            # click/double_click 実行直前: 脳内シミュレーション
                            _pre_click_canvas = None
                            if suggestion.action_type in ("click", "double_click") and suggestion.x and suggestion.y and not self.executor.dry_run:
                                try:
                                    # 1. 画面キャプチャ → キャンバスに取り込み
                                    _pre_click_canvas = await asyncio.to_thread(
                                        self._simulate_click_on_canvas,
                                        suggestion.x, suggestion.y, suggestion.grid_cell, self.step)
                                except Exception as e:
                                    await _broadcast({"type": "log", "message": f"脳内シミュレーション失敗（続行）: {e}"})

                            # click 実行直前: 座標のウィンドウを特定してアクティブ化 + HWND保存
                            if suggestion.action_type in ("click", "double_click") and suggestion.x and suggestion.y and not self.executor.dry_run:
                                hwnd = await asyncio.to_thread(_activate_window_at, suggestion.x, suggestion.y)
                                if hwnd:
                                    self._last_target_hwnd = hwnd
                                    await _broadcast({"type": "log", "message": f"WindowFromPoint: HWND={hwnd} をアクティブ化"})

                            # type/keypress 実行直前: 保存済みHWNDにフォーカスを復帰
                            if suggestion.action_type in ("type", "keypress") and not self.executor.dry_run:
                                if self._last_target_hwnd:
                                    ok = await asyncio.to_thread(_activate_hwnd, self._last_target_hwnd)
                                    if ok:
                                        await _broadcast({"type": "log", "message": f"フォーカス復帰: HWND={self._last_target_hwnd}"})
                                    else:
                                        await asyncio.to_thread(_activate_newest_window)
                                else:
                                    await asyncio.to_thread(_activate_newest_window)
                                await asyncio.sleep(0.15)

                            if suggestion.action_type.lower() == "launch":
                                result = await asyncio.to_thread(_launch_app, suggestion.value or suggestion.target)
                            else:
                                result = await asyncio.to_thread(self.executor.execute_suggestion, suggestion)
                            action_msg = {
                                "type": "action", "step": self.step,
                                "action_type": suggestion.action_type,
                                "target": suggestion.target,
                                "description": result.description,
                                "success": result.success, "dry_run": result.dry_run,
                                "error": result.error, "timestamp": time.time(),
                                "diff_score": diff_score,
                            }
                            await _broadcast(action_msg)
                            self._action_log.append(action_msg)

                            # click/double_click 実行後: 脳内画像で位置検証 + 補正
                            verification_text = ""
                            if _pre_click_canvas and suggestion.action_type in ("click", "double_click") and result.success:
                                try:
                                    vr = await asyncio.to_thread(
                                        self._verify_click_with_canvas,
                                        suggestion.x, suggestion.y, self.step)
                                    verification_text = vr
                                    await _broadcast({"type": "log", "message": f"検証: {vr}"})
                                    # 「変化なし」なら次のクリックをブロックするフラグを立てる
                                    self._last_click_no_change = "変化なし" in vr
                                except Exception as e:
                                    await _broadcast({"type": "log", "message": f"検証失敗（続行）: {e}"})
                            elif suggestion.action_type not in ("click", "double_click"):
                                # クリック以外のアクションが実行されたらフラグをリセット
                                self._last_click_no_change = False

                            # 作業記憶にチャンクを保存
                            result_text = "成功" if result.success else f"失敗: {result.error}"
                            if verification_text:
                                result_text += f" [{verification_text}]"
                            self.working_memory.store(MemoryChunk(
                                step=self.step,
                                action_type=suggestion.action_type,
                                target=suggestion.target,
                                value=suggestion.value,
                                result=result_text,
                                screen_state=analysis.description[:80],
                                grid_cell=suggestion.grid_cell,
                            ))

                            # Dry Run でない場合のみ、次ステップで強制再解析
                            if not result.dry_run:
                                self._last_action_step = self.step
                                if suggestion.action_type in ("click", "scroll"):
                                    self._needs_hires_next = True
                                # click/keypress/focus_window後: フォアグラウンドウィンドウを記憶（type直前の復帰用）
                                if suggestion.action_type in ("click", "keypress", "focus_window"):
                                    await asyncio.sleep(0.2)
                                    hwnd = await asyncio.to_thread(_get_foreground_hwnd)
                                    if hwnd:
                                        self._last_target_hwnd = hwnd
                                # focus_window/click後: ウィンドウが別モニターならキャプチャ先を切替
                                if suggestion.action_type in ("focus_window", "click", "double_click") and result.success and self._last_target_hwnd:
                                    await self._check_and_switch_monitor(
                                        self._last_target_hwnd, capture_request, capture_response, _broadcast)

                                # アプリ起動検出: 検索後にEnterを押した場合、
                                # 新しいウィンドウが背面に開くことがあるのでAlt+Tabでフォーカス移動
                                if suggestion.action_type == "keypress" and \
                                   suggestion.value and suggestion.value.lower() in ("return", "enter"):
                                    # 直前のアクションがtypeだった場合（検索→Enter のパターン）
                                    if len(self._action_log) >= 2 and self._action_log[-2].get("action_type") == "type":
                                        await asyncio.sleep(1.5)  # アプリ起動を待つ
                                        # 最後にアクティブになったウィンドウ（＝最新起動アプリ）をフォアグラウンドに
                                        activated = await asyncio.to_thread(_activate_newest_window)
                                        if activated:
                                            # 起動したアプリのHWNDを保存（以降のtype/keypressで使う）
                                            await asyncio.sleep(0.3)
                                            hwnd = await asyncio.to_thread(_get_foreground_hwnd)
                                            if hwnd:
                                                self._last_target_hwnd = hwnd
                                            await _broadcast({"type": "log", "message": f"アプリ起動検出: ウィンドウをアクティブ化しました (HWND={hwnd})"})
                                            # 起動したアプリが別モニターにある場合はキャプチャ先を切替
                                            await self._check_and_switch_monitor(
                                                hwnd, capture_request, capture_response, _broadcast)
                                        else:
                                            await _broadcast({"type": "log", "message": "アプリ起動検出: ウィンドウが見つかりません"})
                                        self._needs_hires_next = True

                                # アクション後、画面反映を待つ
                                await asyncio.sleep(0.5)

                            if self.logger:
                                self.logger.log_frame(
                                    diff_score=diff_score, changed=True,
                                    description=analysis.description, changes=analysis.changes,
                                    action_type=suggestion.action_type, action_target=suggestion.target,
                                    action_value=suggestion.value or "",
                                    action_x=suggestion.x, action_y=suggestion.y,
                                    action_success=result.success, action_reasoning=suggestion.reasoning,
                                )
                    else:
                        # action_type が "none" の場合、目標達成の可能性をチェック
                        if suggestion.action_type == "none":
                            self._none_count = getattr(self, '_none_count', 0) + 1
                            if self._none_count >= 2:
                                await _broadcast({"type": "log", "message": "目標達成を検出: LLMが2回連続でアクション不要と判断しました"})
                                # 経験キャッシュに成功を記録
                                if self._current_task_signature:
                                    promoted = self._experience_cache.complete_success(
                                        self._current_task_signature)
                                    if promoted:
                                        await _broadcast({"type": "log",
                                            "message": "⚡ 経験キャッシュ: パターン昇格 → 次回は System 1 で即座実行"})
                                break
                        else:
                            self._none_count = 0

                        if self.logger:
                            self.logger.log_frame(
                                diff_score=diff_score, changed=True,
                                description=analysis.description, changes=analysis.changes,
                            )

                except Exception as e:
                    await _broadcast({"type": "error", "message": f"LLM解析エラー: {e}"})

                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0.0, self.interval - elapsed))

        except asyncio.CancelledError:
            pass
        finally:
            # バックグラウンド System 2 タスクをキャンセル
            if self._s2_task and not self._s2_task.done():
                self._s2_task.cancel()
                self._s2_task = None
            capture_request.put("stop")
            cap_thread.join(timeout=3)
            summary = self._cleanup()
            if summary:
                await _broadcast({"type": "log", "message": f"セッション終了: {summary.total_frames}フレーム, {summary.actions_taken}アクション"})
            await _broadcast({"type": "status", "state": "stopped"})

    async def _launch_s2_background(self, suggestion, analysis_desc: str,
                                      gk_ctx) -> None:
        """System 2 思考ループをバックグラウンドで起動する。
        メインループはブロックされず、System 1 で行動を継続する。
        結果は _s2_pending_result に格納され、次の LLM コールで消費される。
        """
        if self._s2_task and not self._s2_task.done():
            return  # 既に実行中 → 多重起動しない

        # スレッドセーフなスナップショットを取得
        wm_snapshot = self.working_memory.snapshot()
        trigger_step = self.step
        # suggestion のコピー（元はメインループで変更される可能性がある）
        import copy
        suggestion_copy = copy.copy(suggestion)

        async def _run():
            try:
                # 独立した弁証法インスタンス（メインループとの競合回避）
                s2 = System2Loop(wm_snapshot, self.canvas_mgr,
                                 DialecticalReasoner())
                result = await asyncio.to_thread(
                    s2.run, suggestion_copy, analysis_desc,
                    None, self.monitor_index, gk_ctx)
                self._s2_pending_result = result
                self._s2_trigger_step = trigger_step
                self._performance_tracker.record(
                    ThinkingMode.SYSTEM2, True,
                    result.confidence, result.duration_seconds)
                await _broadcast({"type": "system2_complete",
                    "message": f"System 2 完了: {result.iterations}反復, "
                               f"{result.duration_seconds:.1f}秒, "
                               f"確信度={result.confidence:.0%}, "
                               f"終了={result.exit_reason}"})
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.warning(f"Background System 2 error: {e}")

        self._s2_task = asyncio.create_task(_run())
        await _broadcast({"type": "system2_start",
            "message": f"System 2 起動(背景): 確信度={suggestion.confidence:.0%}, "
                       f"失敗={gk_ctx.consecutive_failures}, step={self.step}"})

    async def _request_confirm(self, suggestion, message: str) -> bool:
        req_id = str(uuid.uuid4())[:8]
        future = asyncio.get_event_loop().create_future()
        self._confirm_futures[req_id] = future
        await _broadcast({
            "type": "confirm_req", "request_id": req_id, "message": message,
            "action_type": suggestion.action_type, "target": suggestion.target,
        })
        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            return False
        finally:
            self._confirm_futures.pop(req_id, None)

    def resolve_confirm(self, request_id: str, approved: bool):
        future = self._confirm_futures.get(request_id)
        if future and not future.done():
            future.set_result(approved)

    async def execute_manual_action(self, action_type: str, **kwargs) -> dict:
        """手動アクション実行（エージェント停止中でも使用可能）"""
        if self.executor is None:
            self.executor = ActionExecutor(dry_run=False)

        if action_type == "click":
            result = await asyncio.to_thread(self.executor.click, kwargs["x"], kwargs["y"], kwargs.get("button", 1))
        elif action_type == "type":
            result = await asyncio.to_thread(self.executor.type_text, kwargs["text"])
        elif action_type == "keypress":
            result = await asyncio.to_thread(self.executor.keypress, kwargs["key"])
        elif action_type == "scroll":
            result = await asyncio.to_thread(
                self.executor.scroll, kwargs.get("x", 960), kwargs.get("y", 540),
                kwargs.get("direction", "down"), kwargs.get("amount", 3))
        elif action_type == "launch":
            result = await asyncio.to_thread(_launch_app, kwargs.get("command", ""))
        else:
            return {"success": False, "error": f"Unknown action: {action_type}"}

        msg = {"type": "action", "action_type": action_type, "description": result.description,
               "success": result.success, "error": result.error, "timestamp": time.time(), "manual": True}
        await _broadcast(msg)
        self._action_log.append(msg)
        return {"success": result.success, "description": result.description, "error": result.error}

    def get_status(self) -> dict:
        stats = {}
        if self.logger:
            stats = self.logger.stats
        return {
            "state": self.state,
            "step": self.step,
            "goal": self.goal,
            "model": self.model,
            "screen_size": list(self.screen_size),
            "stats": stats,
        }


# --- App ---

app = FastAPI(title="Video2AI Desktop Agent")
session = AgentSession()

# WebSocketクライアント管理: 各クライアントにasyncio.Queueを持たせ、
# 送信を直列化して concurrent write を防ぐ
_ws_clients: dict[WebSocket, asyncio.Queue] = {}


async def _broadcast(message: dict):
    data = json.dumps(message, ensure_ascii=False, default=str)
    for ws, queue in list(_ws_clients.items()):
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            pass  # drop frame if client is too slow


# --- Static Files ---

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/manual.html")
async def manual():
    return FileResponse(_STATIC_DIR / "manual.html")


@app.get("/api/status")
async def get_status():
    return session.get_status()


@app.post("/api/session/start")
async def session_start(req: SessionStartRequest):
    try:
        await session.start(req)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/session/stop")
async def session_stop():
    await session.stop()
    return {"ok": True}


@app.post("/api/session/pause")
async def session_pause():
    session.pause_toggle()
    state = session.state
    await _broadcast({"type": "status", "state": state})
    return {"ok": True, "state": state}


# --- Manual Actions ---

@app.post("/api/action/click")
async def action_click(req: ClickRequest):
    return await session.execute_manual_action("click", x=req.x, y=req.y, button=req.button)


@app.post("/api/action/type")
async def action_type(req: TypeRequest):
    return await session.execute_manual_action("type", text=req.text)


@app.post("/api/action/keypress")
async def action_keypress(req: KeypressRequest):
    return await session.execute_manual_action("keypress", key=req.key)


@app.post("/api/action/scroll")
async def action_scroll(req: ScrollRequest):
    return await session.execute_manual_action("scroll", x=req.x, y=req.y,
                                                direction=req.direction, amount=req.amount)


@app.post("/api/safety/confirm")
async def safety_confirm(req: ConfirmRequest):
    session.resolve_confirm(req.request_id, req.approved)
    return {"ok": True}


@app.get("/api/history")
async def get_history(limit: int = 50):
    return session._action_log[-limit:]


# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    _ws_clients[ws] = queue

    # 送信タスク: queue からメッセージを取り出して直列に送信
    async def sender():
        try:
            while True:
                data = await queue.get()
                await ws.send_text(data)
        except Exception:
            pass

    # プレビュータスク: エージェント未起動時に画面を配信
    # mss は Windows 上でスレッドローカルなので、キャプチャ全体を1つのスレッドで完結させる
    # エージェント起動時はmssを解放してからスリープする
    _preview_stop = threading.Event()

    async def preview_loop():
        def _capture_worker():
            """専用スレッドでキャプチャを実行（mssスレッドローカル対策）"""
            while not _preview_stop.is_set():
                # エージェント起動中はmssを保持せずスリープ
                if session.state == "running":
                    time.sleep(1.0)
                    continue

                cap = ScreenCapture(monitor_index=_detect_primary_monitor(), resize_width=960)
                cap.__enter__()
                try:
                    screen_size = cap.get_screen_size()
                    while not _preview_stop.is_set() and session.state != "running":
                        try:
                            b64 = cap.capture_as_base64("JPEG", 40)
                            msg = json.dumps({
                                "type": "frame", "data": b64,
                                "width": screen_size[0], "height": screen_size[1],
                                "step": 0, "preview": True,
                            })
                            try:
                                queue.put_nowait(msg)
                            except asyncio.QueueFull:
                                pass
                        except Exception as e:
                            logger.warning(f"Preview capture error: {e}")
                            break
                        time.sleep(0.5)
                finally:
                    try:
                        cap.__exit__(None, None, None)
                    except Exception:
                        pass

        try:
            await asyncio.to_thread(_capture_worker)
        except asyncio.CancelledError:
            _preview_stop.set()

    sender_task = asyncio.create_task(sender())
    preview_task = asyncio.create_task(preview_loop())

    try:
        # 接続時にステータス送信
        queue.put_nowait(json.dumps(session.get_status()))

        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type", "")
                if msg_type == "click":
                    await session.execute_manual_action("click", x=msg["x"], y=msg["y"])
                elif msg_type == "pause":
                    session.pause_toggle()
                    await _broadcast({"type": "status", "state": session.state})
                elif msg_type == "stop":
                    await session.stop()
                elif msg_type == "confirm":
                    session.resolve_confirm(msg["request_id"], msg["approved"])
            except Exception as e:
                logger.warning(f"WS message handling error: {e}")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}\n{traceback.format_exc()}")
    finally:
        _ws_clients.pop(ws, None)
        sender_task.cancel()
        preview_task.cancel()
