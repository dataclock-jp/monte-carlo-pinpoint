"""
action_executor.py
マウス・キーボード操作実行モジュール

xdotool（Linux）とPyAutoGUIを使ってデスクトップ操作を実行する。

実行モード（優先順）:
  dry_run=True      : ログのみ出力、実際の操作は行わない
  preview_mode=True : 操作前にオーバーレイで予告表示、Escでキャンセル可能
  通常              : 即時実行

対応アクション:
  click     - 指定座標をクリック
  type      - テキストを入力
  keypress  - キーを押す（例: "Return", "ctrl+s", "Escape"）
  scroll    - スクロール
  wait      - 指定秒数待機
  none      - 何もしない
"""
import os
import sys
import time
import subprocess
import shutil
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class ActionResult:
    """アクション実行結果。"""
    success: bool
    action_type: str
    description: str
    error: str = ""
    dry_run: bool = False
    cancelled: bool = False
    verification: Optional[object] = None  # VerificationResult (mental_image)


class ActionPreview:
    """
    操作プレビューオーバーレイ。
    実行予定のアクションを画面上に表示し、カウントダウン後に実行する。
    カウントダウン中にEscキーが押されるとキャンセルする。
    """

    ACTION_LABELS = {
        "click": "クリック",
        "double_click": "ダブルクリック",
        "type": "テキスト入力",
        "keypress": "キー入力",
        "scroll": "スクロール",
        "wait": "待機",
    }

    def __init__(self, delay_seconds: float = 3.0):
        self.delay_seconds = delay_seconds

    def show_and_wait(self, suggestion) -> bool:
        """
        プレビューを表示し、delay秒後にTrueを返す。
        Escキーが押された場合はFalseを返す。

        Returns
        -------
        bool
            True=実行する, False=キャンセル
        """
        action_type = getattr(suggestion, "action_type", "none").lower()
        label = self.ACTION_LABELS.get(action_type, action_type)
        target = getattr(suggestion, "target", "")
        value = getattr(suggestion, "value", "")
        x = getattr(suggestion, "x", None)
        y = getattr(suggestion, "y", None)

        # 表示テキストの構築
        parts = [f"[PREVIEW] {label}"]
        if target:
            parts.append(f"対象: {target}")
        if value and action_type in ("type", "keypress"):
            parts.append(f"値: {value}")
        if x is not None and y is not None:
            parts.append(f"座標: ({x}, {y})")
        desc = " | ".join(parts)

        cancelled = [False]

        # tkinterオーバーレイを試みる（失敗時はコンソールフォールバック）
        try:
            self._show_overlay(suggestion, desc, cancelled)
        except Exception:
            self._show_console(desc, cancelled)

        return not cancelled[0]

    def _show_overlay(self, suggestion, desc: str, cancelled: list):
        """tkinterでオーバーレイウィンドウを表示する。"""
        import tkinter as tk

        action_type = getattr(suggestion, "action_type", "").lower()
        x = getattr(suggestion, "x", None)
        y = getattr(suggestion, "y", None)

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.9)
        root.configure(bg="#1a1a2e")

        # ウィンドウ位置: アクション座標の近くに配置（画面外にならないよう調整）
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        win_w, win_h = 360, 100
        if x is not None and y is not None:
            wx = min(max(x - win_w // 2, 10), screen_w - win_w - 10)
            wy = max(y - win_h - 30, 10)
        else:
            wx = (screen_w - win_w) // 2
            wy = (screen_h - win_h) // 2
        root.geometry(f"{win_w}x{win_h}+{wx}+{wy}")

        # ラベル
        label = self.ACTION_LABELS.get(action_type, action_type)
        target_text = getattr(suggestion, "target", "") or ""
        tk.Label(root, text=f"{label}: {target_text[:40]}", font=("Segoe UI", 11, "bold"),
                 fg="#e94560", bg="#1a1a2e").pack(pady=(10, 2))
        countdown_var = tk.StringVar(value=f"{self.delay_seconds:.1f}秒後に実行 | Escでキャンセル")
        tk.Label(root, textvariable=countdown_var, font=("Segoe UI", 9),
                 fg="#cccccc", bg="#1a1a2e").pack()

        # マーカーウィンドウ（クリック系のみ）
        marker = None
        if action_type in ("click", "double_click", "scroll") and x is not None and y is not None:
            marker = tk.Toplevel(root)
            marker.overrideredirect(True)
            marker.attributes("-topmost", True)
            marker.attributes("-alpha", 0.7)
            marker_size = 24
            marker.geometry(f"{marker_size}x{marker_size}+{x - marker_size // 2}+{y - marker_size // 2}")
            canvas = tk.Canvas(marker, width=marker_size, height=marker_size,
                               bg="#1a1a2e", highlightthickness=0)
            canvas.pack()
            canvas.create_oval(2, 2, marker_size - 2, marker_size - 2,
                               outline="#e94560", width=3)

        # Escキーでキャンセル
        def on_escape(event):
            cancelled[0] = True

        root.bind("<Escape>", on_escape)
        root.focus_force()

        # カウントダウンループ
        start = time.time()
        while time.time() - start < self.delay_seconds:
            if cancelled[0]:
                break
            remaining = self.delay_seconds - (time.time() - start)
            countdown_var.set(f"{remaining:.1f}秒後に実行 | Escでキャンセル")
            root.update()
            time.sleep(0.05)

        # msvcrt でもEscをチェック（ウィンドウにフォーカスがない場合の補助）
        if sys.platform == "win32" and not cancelled[0]:
            try:
                import msvcrt
                while msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\x1b':
                        cancelled[0] = True
            except Exception:
                pass

        if marker:
            marker.destroy()
        root.destroy()

    def _show_console(self, desc: str, cancelled: list):
        """コンソールフォールバック: テキストでプレビュー表示。"""
        print(f"\n  {desc}")
        print(f"  {self.delay_seconds:.0f}秒後に実行します... (Escでキャンセル)")

        start = time.time()
        while time.time() - start < self.delay_seconds:
            if sys.platform == "win32":
                try:
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'\x1b':
                            cancelled[0] = True
                            print("  [CANCELLED]")
                            return
                except Exception:
                    pass
            time.sleep(0.1)


class ActionExecutor:
    """
    マウス・キーボード操作を実行するクラス。
    
    Linux環境ではxdotoolを優先使用し、フォールバックとしてPyAutoGUIを使用。
    ドライランモードでは実際の操作を行わずログのみ出力。
    """

    def __init__(
        self,
        dry_run: bool = False,
        preview_mode: bool = False,
        preview_delay: float = 3.0,
        display: str = ":99",
        move_duration: float = 0.3,
        click_delay: float = 0.1,
        mental_image=None,
    ):
        """
        Parameters
        ----------
        dry_run : bool
            Trueの場合、実際の操作を行わずログのみ出力
        preview_mode : bool
            Trueの場合、操作前にオーバーレイで予告表示しEscでキャンセル可能
        preview_delay : float
            プレビューモード時の待機時間（秒）
        display : str
            X11ディスプレイ番号（例: ":99", ":0"）
        move_duration : float
            マウス移動にかける時間（秒）
        click_delay : float
            クリック後の待機時間（秒）
        mental_image : MentalImage, optional
            脳内画像によるクリック位置検証。指定時、click で verify=True が使える。
        """
        self.dry_run = dry_run
        self.preview_mode = preview_mode
        self.preview_delay = preview_delay
        self.display = display
        self.move_duration = move_duration
        self.click_delay = click_delay
        self.mental_image = mental_image
        self._pre_paste_focus_fn = None  # Ctrl+V 直前のフォーカス復帰関数（MCP用）
        self._has_xdotool = shutil.which("xdotool") is not None

        # 環境変数にDISPLAYを設定
        if display:
            os.environ["DISPLAY"] = display

    def _xdotool(self, *args) -> bool:
        """xdotoolコマンドを実行する。"""
        if not self._has_xdotool:
            return False
        try:
            env = os.environ.copy()
            env["DISPLAY"] = self.display
            result = subprocess.run(
                ["xdotool"] + list(args),
                env=env,
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    # _ensure_ime_off / _send_unicode_string は削除済み
    # Windows では常にクリップボード経由（Ctrl+V）で入力し、IME を完全にバイパスする

    def move_to(self, x: int, y: int) -> ActionResult:
        """マウスカーソルを指定座標に移動する（クリックはしない）。"""
        desc = f"マウス移動 ({x}, {y})"
        if self.dry_run:
            return ActionResult(True, "move", f"[DRY-RUN] {desc}", dry_run=True)
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.moveTo(x, y, duration=self.move_duration)
            return ActionResult(True, "move", desc)
        except Exception as e:
            return ActionResult(False, "move", desc, error=str(e))

    def click(self, x: int, y: int, button: int = 1, verify: bool = False) -> ActionResult:
        """
        指定座標をクリックする。

        Parameters
        ----------
        x, y : int
            クリック座標
        button : int
            マウスボタン（1=左, 2=中, 3=右）
        verify : bool
            True なら脳内画像で位置検証し、ずれていれば補正する
        """
        desc = f"クリック ({x}, {y}) ボタン{button}"
        if self.dry_run:
            return ActionResult(True, "click", f"[DRY-RUN] {desc}", dry_run=True)

        # 検証モード: クリック前のコンテキストをキャプチャ
        before_crop = None
        if verify and self.mental_image:
            before_crop = self.mental_image.capture_context(x, y)

        if self._has_xdotool:
            ok = self._xdotool("mousemove", str(x), str(y))
            time.sleep(self.move_duration)
            ok = ok and self._xdotool("click", str(button))
            time.sleep(self.click_delay)
            if ok:
                result = ActionResult(True, "click", desc)
                if before_crop and self.mental_image:
                    result.verification = self._verify_and_correct(x, y, before_crop)
                return result

        # フォールバック: PyAutoGUI
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.moveTo(x, y, duration=self.move_duration)
            pyautogui.click(button=["left", "middle", "right"][button - 1])
            time.sleep(self.click_delay)
            result = ActionResult(True, "click", desc)
            if before_crop and self.mental_image:
                result.verification = self._verify_and_correct(x, y, before_crop)
            return result
        except Exception as e:
            return ActionResult(False, "click", desc, error=str(e))

    def _verify_and_correct(self, x: int, y: int, before_crop) -> object:
        """クリック後の位置を検証し、ずれていれば補正する。"""
        vr = self.mental_image.verify_click(x, y, before_crop)
        if not vr.position_ok:
            corrected = self.mental_image.correct_position(x, y)
            if corrected:
                vr.corrected = True
                vr.assessment += " → 補正済み"
        return vr

    def double_click(self, x: int, y: int) -> ActionResult:
        """ダブルクリックする。"""
        desc = f"ダブルクリック ({x}, {y})"
        if self.dry_run:
            return ActionResult(True, "double_click", f"[DRY-RUN] {desc}", dry_run=True)

        if self._has_xdotool:
            self._xdotool("mousemove", str(x), str(y))
            time.sleep(self.move_duration)
            ok = self._xdotool("click", "--repeat", "2", "1")
            if ok:
                return ActionResult(True, "double_click", desc)

        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.doubleClick(x, y)
            return ActionResult(True, "double_click", desc)
        except Exception as e:
            return ActionResult(False, "double_click", desc, error=str(e))

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int,
             button: int = 1, duration: float = 0.5) -> ActionResult:
        """
        ドラッグ操作（始点から終点まで）。

        Parameters
        ----------
        start_x, start_y : int
            ドラッグ開始座標
        end_x, end_y : int
            ドラッグ終了座標
        button : int
            マウスボタン（1=左, 2=中, 3=右）
        duration : float
            ドラッグにかける時間（秒）
        """
        desc = f"ドラッグ ({start_x},{start_y})→({end_x},{end_y}) ボタン{button}"
        if self.dry_run:
            return ActionResult(True, "drag", f"[DRY-RUN] {desc}", dry_run=True)

        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            btn = ["left", "middle", "right"][button - 1]
            pyautogui.moveTo(start_x, start_y, duration=self.move_duration)
            time.sleep(0.05)
            pyautogui.mouseDown(button=btn)
            time.sleep(0.05)
            pyautogui.moveTo(end_x, end_y, duration=duration)
            time.sleep(0.05)
            pyautogui.mouseUp(button=btn)
            return ActionResult(True, "drag", desc)
        except Exception as e:
            return ActionResult(False, "drag", desc, error=str(e))

    def drag_path(self, points: list, button: int = 1,
                  step_duration: float = 0.02,
                  pause_between: float = 0.005) -> ActionResult:
        """
        複数のウェイポイントに沿ってドラッグする（マウスボタンを離さず連続移動）。
        ペイントソフトでの滑らかな曲線描画に使用。

        Parameters
        ----------
        points : list of (x, y) tuples
            ドラッグパスの座標列
        button : int
            マウスボタン（1=左, 2=中, 3=右）
        step_duration : float
            各ポイント間の移動時間（秒）
        pause_between : float
            各移動後の待ち時間（秒）
        """
        if len(points) < 2:
            return ActionResult(False, "drag_path", "2点以上の座標が必要です",
                                error="insufficient points")
        desc = f"パスドラッグ {len(points)}点 ({points[0]}→...→{points[-1]})"
        if self.dry_run:
            return ActionResult(True, "drag_path", f"[DRY-RUN] {desc}",
                                dry_run=True)
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            btn = ["left", "middle", "right"][button - 1]
            pyautogui.moveTo(points[0][0], points[0][1],
                             duration=self.move_duration)
            time.sleep(0.05)
            pyautogui.mouseDown(button=btn)
            time.sleep(0.05)
            for px, py in points[1:]:
                pyautogui.moveTo(px, py, duration=step_duration)
                if pause_between > 0:
                    time.sleep(pause_between)
            time.sleep(0.05)
            pyautogui.mouseUp(button=btn)
            return ActionResult(True, "drag_path", desc)
        except Exception as e:
            return ActionResult(False, "drag_path", desc, error=str(e))

    def type_text(self, text: str, clear_first: bool = False) -> ActionResult:
        """
        テキストを入力する。
        
        Parameters
        ----------
        text : str
            入力するテキスト
        clear_first : bool
            入力前に既存テキストをクリア（Ctrl+A → Delete）
        """
        desc = f"テキスト入力: '{text[:30]}{'...' if len(text) > 30 else ''}'"
        if self.dry_run:
            return ActionResult(True, "type", f"[DRY-RUN] {desc}", dry_run=True)

        if clear_first:
            self.keypress("ctrl+a")
            self.keypress("Delete")

        if self._has_xdotool:
            # xdotoolはUnicode対応
            ok = self._xdotool("type", "--clearmodifiers", "--delay", "30", text)
            if ok:
                return ActionResult(True, "type", desc)

        # Windows: SendInput + KEYEVENTF_UNICODE でクリップボード不要のテキスト入力
        # スペースは VK_SPACE 仮想キーコードで送信（IME干渉による文字化け回避）
        if sys.platform == "win32":
            try:
                ok = self._send_input_unicode(text)
                if ok:
                    return ActionResult(True, "type", desc)
            except Exception as e:
                logging.warning(f"SendInput Unicode failed, falling back to clipboard: {e}")

        # フォールバック: クリップボード経由（Linux or SendInput 失敗時）
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            import pyperclip
            old_clipboard = ""
            try:
                old_clipboard = pyperclip.paste()
            except Exception:
                pass
            pyperclip.copy(text)
            time.sleep(0.02)
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.05)
            try:
                pyperclip.copy(old_clipboard)
            except Exception:
                pass
            return ActionResult(True, "type", desc)
        except Exception as e:
            return ActionResult(False, "type", desc, error=str(e))

    @staticmethod
    def _send_input_unicode(text: str) -> bool:
        """SendInput + KEYEVENTF_UNICODE でテキストを入力する。
        クリップボードを使わず、Unicode 文字を直接キーボード入力として送信。
        フォーカス競合・クリップボード汚染を完全回避。
        """
        import ctypes
        from ctypes import wintypes

        KEYEVENTF_UNICODE = 0x0004
        KEYEVENTF_KEYUP = 0x0002
        INPUT_KEYBOARD = 1

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class INPUT_UNION(ctypes.Union):
            _fields_ = [
                ("mi", MOUSEINPUT),
                ("ki", KEYBDINPUT),
                ("hi", HARDWAREINPUT),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", wintypes.DWORD),
                ("union", INPUT_UNION),
            ]

        if not text:
            return True

        # 制御文字・特殊文字→仮想キーコードのマッピング
        # スペースを VK_SPACE で送ると IME 干渉による後続文字化けを回避できる
        CONTROL_CHAR_TO_VK = {
            '\n': 0x0D,  # VK_RETURN
            '\r': 0x0D,  # VK_RETURN
            '\t': 0x09,  # VK_TAB
            ' ':  0x20,  # VK_SPACE — KEYEVENTF_UNICODE だと IME が介入し後続文字が化ける
        }

        # 1文字ずつ SendInput（KeyDown + KeyUp）を送信
        import time as _t
        total_sent = 0
        for ch in text:
            pair = (INPUT * 2)()
            vk = CONTROL_CHAR_TO_VK.get(ch)
            if vk:
                # 制御文字・スペースは仮想キーコードで送信（IMEバイパス）
                pair[0].type = INPUT_KEYBOARD
                pair[0].union.ki.wVk = vk
                pair[1].type = INPUT_KEYBOARD
                pair[1].union.ki.wVk = vk
                pair[1].union.ki.dwFlags = KEYEVENTF_KEYUP
            else:
                # 通常文字は KEYEVENTF_UNICODE で送信
                code = ord(ch)
                pair[0].type = INPUT_KEYBOARD
                pair[0].union.ki.wScan = code
                pair[0].union.ki.dwFlags = KEYEVENTF_UNICODE
                pair[1].type = INPUT_KEYBOARD
                pair[1].union.ki.wScan = code
                pair[1].union.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

            sent = ctypes.windll.user32.SendInput(
                2, ctypes.byref(pair), ctypes.sizeof(INPUT))
            total_sent += sent

        return total_sent == len(text) * 2

    def keypress(self, key: str) -> ActionResult:
        """
        キーを押す。
        
        Parameters
        ----------
        key : str
            キー名（例: "Return", "Escape", "ctrl+s", "ctrl+c"）
            xdotoolのキー名形式に準拠
        """
        desc = f"キー入力: {key}"
        if self.dry_run:
            return ActionResult(True, "keypress", f"[DRY-RUN] {desc}", dry_run=True)

        if self._has_xdotool:
            ok = self._xdotool("key", "--clearmodifiers", key)
            if ok:
                return ActionResult(True, "keypress", desc)

        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            # xdotool形式/LLM出力をpyautogui形式に変換
            _KEY_MAP = {
                "return": "enter", "escape": "esc", "backspace": "backspace",
                "super": "win", "super_l": "winleft", "super_r": "winright",
                "win": "win", "windows": "win",
                "space": "space", "tab": "tab", "delete": "delete",
                "up": "up", "down": "down", "left": "left", "right": "right",
            }
            if "+" in key:
                parts = [_KEY_MAP.get(p.strip().lower(), p.strip().lower()) for p in key.split("+")]
                pyautogui.hotkey(*parts)
            else:
                mapped = _KEY_MAP.get(key.strip().lower(), key.strip().lower())
                pyautogui.press(mapped)
            return ActionResult(True, "keypress", desc)
        except Exception as e:
            return ActionResult(False, "keypress", desc, error=str(e))

    def scroll(self, x: int, y: int, direction: str = "down", amount: int = 3) -> ActionResult:
        """
        スクロールする。
        
        Parameters
        ----------
        x, y : int
            スクロール位置
        direction : str
            "up" or "down"
        amount : int
            スクロール量（クリック数）
        """
        desc = f"スクロール {direction} ({x}, {y}) x{amount}"
        if self.dry_run:
            return ActionResult(True, "scroll", f"[DRY-RUN] {desc}", dry_run=True)

        button = "4" if direction == "up" else "5"
        if self._has_xdotool:
            self._xdotool("mousemove", str(x), str(y))
            for _ in range(amount):
                self._xdotool("click", button)
            return ActionResult(True, "scroll", desc)

        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.scroll(amount if direction == "up" else -amount, x=x, y=y)
            return ActionResult(True, "scroll", desc)
        except Exception as e:
            return ActionResult(False, "scroll", desc, error=str(e))

    def focus_window(self, keyword: str) -> ActionResult:
        """
        ウィンドウタイトルにキーワードを含むウィンドウをフォアグラウンドにする。

        Parameters
        ----------
        keyword : str
            ウィンドウタイトルの検索キーワード（例: "メモ帳", "Notepad"）
        """
        desc = f"ウィンドウフォーカス: '{keyword}'"
        if self.dry_run:
            return ActionResult(True, "focus_window", f"[DRY-RUN] {desc}", dry_run=True)

        if sys.platform != "win32":
            return ActionResult(False, "focus_window", desc, error="Windows以外は未対応")

        try:
            import ctypes
            import ctypes.wintypes
            user32 = ctypes.windll.user32
            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

            found_hwnd = [None]
            keyword_lower = keyword.lower()

            def callback(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, title, 256)
                if keyword_lower in title.value.lower():
                    found_hwnd[0] = hwnd
                    return False  # stop
                return True

            user32.EnumWindows(WNDENUMPROC(callback), 0)

            if found_hwnd[0]:
                hwnd = found_hwnd[0]
                KEYEVENTF_EXTENDEDKEY = 0x0001
                KEYEVENTF_KEYUP = 0x0002
                VK_MENU = 0x12
                user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY, 0)
                user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.SetForegroundWindow(hwnd)
                import time as _t
                _t.sleep(0.05)
                # Alt キーでメニューバーがアクティブになるアプリ対策: Escape で解除
                VK_ESCAPE = 0x1B
                user32.keybd_event(VK_ESCAPE, 0, 0, 0)
                user32.keybd_event(VK_ESCAPE, 0, KEYEVENTF_KEYUP, 0)
                _t.sleep(0.02)
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, title, 256)
                return ActionResult(True, "focus_window", f"{desc} → '{title.value}' (HWND={hwnd})")
            else:
                return ActionResult(False, "focus_window", desc, error=f"'{keyword}' を含むウィンドウが見つかりません")
        except Exception as e:
            return ActionResult(False, "focus_window", desc, error=str(e))

    def wait(self, seconds: float) -> ActionResult:
        """指定秒数待機する。"""
        desc = f"待機 {seconds}秒"
        if self.dry_run:
            return ActionResult(True, "wait", f"[DRY-RUN] {desc}", dry_run=True)
        time.sleep(seconds)
        return ActionResult(True, "wait", desc)

    def execute_suggestion(self, suggestion) -> ActionResult:
        """
        ActionSuggestionオブジェクトを実行する。

        Parameters
        ----------
        suggestion : ActionSuggestion
            LLMVisionが提案したアクション
        """
        action_type = suggestion.action_type.lower()

        if action_type == "none":
            return ActionResult(True, "none", "アクションなし（監視継続）")

        # プレビューモード: オーバーレイ表示 + 遅延 + Escキャンセル
        if self.preview_mode and not self.dry_run:
            preview = ActionPreview(delay_seconds=self.preview_delay)
            proceed = preview.show_and_wait(suggestion)
            if not proceed:
                desc = f"[CANCELLED] {action_type} - ユーザーがキャンセルしました"
                return ActionResult(False, action_type, desc, cancelled=True)

        if action_type == "click":
            if suggestion.x and suggestion.y:
                return self.click(suggestion.x, suggestion.y)
            return ActionResult(False, "click", "座標が指定されていません", error="No coordinates")

        elif action_type == "double_click":
            if suggestion.x and suggestion.y:
                return self.double_click(suggestion.x, suggestion.y)
            return ActionResult(False, "double_click", "座標が指定されていません", error="No coordinates")

        elif action_type == "type":
            return self.type_text(suggestion.value)

        elif action_type == "keypress":
            return self.keypress(suggestion.value or "Return")

        elif action_type == "scroll":
            direction = "down"
            if suggestion.value and "up" in suggestion.value.lower():
                direction = "up"
            x = suggestion.x or 960
            y = suggestion.y or 540
            return self.scroll(x, y, direction)

        elif action_type == "drag":
            end_x = getattr(suggestion, "end_x", None)
            end_y = getattr(suggestion, "end_y", None)
            if suggestion.x and suggestion.y and end_x and end_y:
                return self.drag(suggestion.x, suggestion.y, end_x, end_y)
            return ActionResult(False, "drag", "始点・終点の座標が不足", error="Missing coordinates")

        elif action_type == "focus_window":
            return self.focus_window(suggestion.value or suggestion.target)

        elif action_type == "wait":
            try:
                secs = float(suggestion.value) if suggestion.value else 1.0
            except ValueError:
                secs = 1.0
            return self.wait(secs)

        else:
            return ActionResult(False, action_type, f"未知のアクション: {action_type}", error="Unknown action")

    def get_mouse_position(self) -> tuple[int, int]:
        """現在のマウス座標を返す。"""
        if self._has_xdotool:
            try:
                env = os.environ.copy()
                env["DISPLAY"] = self.display
                result = subprocess.run(
                    ["xdotool", "getmouselocation"],
                    env=env, capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    parts = result.stdout.split()
                    x = int(parts[0].split(":")[1])
                    y = int(parts[1].split(":")[1])
                    return x, y
            except Exception:
                pass
        return 0, 0
