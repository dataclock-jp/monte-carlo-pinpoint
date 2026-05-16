"""
Focus-Free Operation — フォーカスもカーソルも動かさずにアプリを操作する

ユーザーがPCを使っている最中でも、AIが裏でアプリを操作できる。
メニューが閉じたり、カーソルが飛んだりしない。

3層フォールバック:
  1. UI Automation パターン（Invoke, Value, Toggle 等）— 最も確実、フォーカス不要
  2. PostMessage（WM_LBUTTONDOWN 等）— 多くの Win32 アプリで動作、フォーカス不要
  3. SendInput — 最終手段、カーソルが動く（従来方式）

使い方:
    ff = FocusFreeOperator()

    # ボタンをクリック（フォーカス不要）
    result = ff.click_element("メモ帳", "設定")

    # テキストを設定（フォーカス不要）
    result = ff.set_value("メモ帳", "テキスト エディター", "Hello World")

    # 要素の情報を取得
    info = ff.get_element_info("メモ帳", "設定")
"""

import ctypes
import ctypes.wintypes
import logging
import time
from dataclasses import dataclass
from typing import Optional

import comtypes
import comtypes.client

logger = logging.getLogger(__name__)

# UI Automation COM モジュール（プロセスで1回ロード）
_uia_module = comtypes.client.GetModule("UIAutomationCore.dll")

# UI Automation パターン ID
UIA_InvokePatternId = 10000
UIA_ValuePatternId = 10002
UIA_ScrollPatternId = 10004
UIA_ExpandCollapsePatternId = 10005
UIA_SelectionItemPatternId = 10010
UIA_TogglePatternId = 10015

# Win32 メッセージ定数
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_SETTEXT = 0x000C
MK_LBUTTON = 0x0001

# Control Type ID → 名前
_CONTROL_TYPES = {
    50000: "Button", 50002: "CheckBox", 50003: "ComboBox",
    50004: "Edit", 50006: "Document", 50007: "MenuItem",
    50020: "Text", 50032: "Window", 50033: "Pane",
}


@dataclass
class FFResult:
    """フォーカスフリー操作の結果"""
    success: bool
    method: str  # "uia_invoke", "uia_value", "postmessage", "sendinput", "failed"
    message: str
    element_name: str = ""
    control_type: str = ""

    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"[{status} {self.method}] {self.message}"


@dataclass
class ElementInfo:
    """UI 要素の詳細情報"""
    name: str
    control_type: str
    rect: tuple[int, int, int, int]
    value: Optional[str] = None
    has_invoke: bool = False
    has_value: bool = False
    has_toggle: bool = False
    has_expand: bool = False
    has_selection: bool = False
    hwnd: int = 0

    @property
    def center_x(self) -> int:
        return (self.rect[0] + self.rect[2]) // 2

    @property
    def center_y(self) -> int:
        return (self.rect[1] + self.rect[3]) // 2

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]


def _create_uia():
    """スレッド用 IUIAutomation インスタンスを生成"""
    return comtypes.CoCreateInstance(
        _uia_module.CUIAutomation._reg_clsid_,
        interface=_uia_module.IUIAutomation,
    )


def _makelparam(x: int, y: int) -> int:
    """PostMessage 用の LPARAM を作成（クライアント座標）"""
    return (y << 16) | (x & 0xFFFF)


class FocusFreeOperator:
    """フォーカスもカーソルも動かさずにアプリを操作するクラス"""

    def __init__(self):
        self._uia = None

    def _ensure_uia(self):
        if self._uia is None:
            comtypes.CoInitialize()
            self._uia = _create_uia()
        return self._uia

    # === ウィンドウ・要素の検索 ===

    def _find_window(self, window_keyword: str):
        """ウィンドウタイトルのキーワードで検索して COM 要素を返す"""
        uia = self._ensure_uia()
        root = uia.GetRootElement()
        condition = uia.CreateTrueCondition()
        walker = uia.CreateTreeWalker(condition)
        child = walker.GetFirstChildElement(root)
        while child:
            try:
                name = child.CurrentName
                if name and window_keyword in name:
                    return child
            except Exception:
                pass
            try:
                child = walker.GetNextSiblingElement(child)
            except Exception:
                break
        return None

    def _find_element_in_tree(self, parent, name: str = "", control_type: str = "",
                               max_depth: int = 6, depth: int = 0):
        """UI ツリーを走査して要素を検索（部分一致）"""
        uia = self._ensure_uia()
        try:
            el_name = parent.CurrentName or ""
            ct_id = parent.CurrentControlType
            ct = _CONTROL_TYPES.get(ct_id, f"Unknown({ct_id})")

            name_match = not name or name in el_name
            type_match = not control_type or control_type == ct
            if name_match and type_match and el_name:
                return parent
        except Exception:
            pass

        if depth >= max_depth:
            return None

        try:
            condition = uia.CreateTrueCondition()
            walker = uia.CreateTreeWalker(condition)
            child = walker.GetFirstChildElement(parent)
            while child:
                result = self._find_element_in_tree(
                    child, name, control_type, max_depth, depth + 1)
                if result:
                    return result
                try:
                    child = walker.GetNextSiblingElement(child)
                except Exception:
                    break
        except Exception:
            pass
        return None

    def _find_all_elements(self, parent, name: str = "", control_type: str = "",
                            max_depth: int = 6, depth: int = 0, results: list = None):
        """UI ツリーを走査して要素を複数検索"""
        if results is None:
            results = []
        uia = self._ensure_uia()
        try:
            el_name = parent.CurrentName or ""
            ct_id = parent.CurrentControlType
            ct = _CONTROL_TYPES.get(ct_id, f"Unknown({ct_id})")

            name_match = not name or name in el_name
            type_match = not control_type or control_type == ct
            rect = parent.CurrentBoundingRectangle
            visible = (rect.right - rect.left) > 0 and (rect.bottom - rect.top) > 0

            if name_match and type_match and el_name and visible:
                results.append(parent)
        except Exception:
            pass

        if depth >= max_depth:
            return results

        try:
            condition = uia.CreateTrueCondition()
            walker = uia.CreateTreeWalker(condition)
            child = walker.GetFirstChildElement(parent)
            while child:
                self._find_all_elements(
                    child, name, control_type, max_depth, depth + 1, results)
                try:
                    child = walker.GetNextSiblingElement(child)
                except Exception:
                    break
        except Exception:
            pass
        return results

    def _get_element_info(self, com_element) -> Optional[ElementInfo]:
        """COM 要素から ElementInfo を生成"""
        try:
            name = com_element.CurrentName or ""
            ct_id = com_element.CurrentControlType
            ct = _CONTROL_TYPES.get(ct_id, f"Unknown({ct_id})")
            rect = com_element.CurrentBoundingRectangle
            rect_tuple = (rect.left, rect.top, rect.right, rect.bottom)

            # HWND 取得
            hwnd = 0
            try:
                hwnd = com_element.CurrentNativeWindowHandle
            except Exception:
                pass

            # 値の取得
            value = None
            try:
                vp = com_element.GetCurrentPattern(UIA_ValuePatternId)
                if vp:
                    vi = vp.QueryInterface(_uia_module.IUIAutomationValuePattern)
                    value = vi.CurrentValue
            except Exception:
                pass

            # パターンの有無を確認
            has_invoke = self._has_pattern(com_element, UIA_InvokePatternId)
            has_value = self._has_pattern(com_element, UIA_ValuePatternId)
            has_toggle = self._has_pattern(com_element, UIA_TogglePatternId)
            has_expand = self._has_pattern(com_element, UIA_ExpandCollapsePatternId)
            has_selection = self._has_pattern(com_element, UIA_SelectionItemPatternId)

            return ElementInfo(
                name=name[:200], control_type=ct, rect=rect_tuple,
                value=value[:500] if value else None,
                has_invoke=has_invoke, has_value=has_value,
                has_toggle=has_toggle, has_expand=has_expand,
                has_selection=has_selection, hwnd=hwnd,
            )
        except Exception:
            return None

    def _has_pattern(self, com_element, pattern_id: int) -> bool:
        """要素が指定パターンをサポートしているか"""
        try:
            p = com_element.GetCurrentPattern(pattern_id)
            return p is not None
        except Exception:
            return False

    # === 公開 API ===

    def get_element_info(self, window_keyword: str, element_name: str,
                          control_type: str = "") -> Optional[ElementInfo]:
        """ウィンドウ内の UI 要素の詳細情報を取得する"""
        win = self._find_window(window_keyword)
        if not win:
            return None
        el = self._find_element_in_tree(win, element_name, control_type)
        if not el:
            return None
        return self._get_element_info(el)

    def list_actionable(self, window_keyword: str) -> list[ElementInfo]:
        """操作可能な（パターンを持つ）UI 要素を一覧する"""
        win = self._find_window(window_keyword)
        if not win:
            return []
        all_elements = self._find_all_elements(win)
        result = []
        for el in all_elements:
            info = self._get_element_info(el)
            if info and (info.has_invoke or info.has_value or info.has_toggle
                         or info.has_expand or info.has_selection):
                result.append(info)
        return result

    def click_element(self, window_keyword: str, element_name: str,
                       control_type: str = "") -> FFResult:
        """要素をクリック/Invoke する（3層フォールバック）。

        1. UI Automation Invoke パターン（フォーカス不要）
        2. PostMessage WM_LBUTTONDOWN/UP（フォーカス不要）
        3. SendInput（カーソルが動く、最終手段）
        """
        win = self._find_window(window_keyword)
        if not win:
            return FFResult(False, "failed", f"ウィンドウ '{window_keyword}' が見つかりません")

        el = self._find_element_in_tree(win, element_name, control_type)
        if not el:
            return FFResult(False, "failed",
                            f"要素 '{element_name}' が見つかりません",
                            element_name=element_name)

        info = self._get_element_info(el)
        el_desc = f"'{info.name}' [{info.control_type}]" if info else element_name

        # --- 1. UI Automation Invoke ---
        if info and info.has_invoke:
            try:
                pattern = el.GetCurrentPattern(UIA_InvokePatternId)
                invoke = pattern.QueryInterface(_uia_module.IUIAutomationInvokePattern)
                invoke.Invoke()
                return FFResult(True, "uia_invoke",
                                f"{el_desc} を Invoke しました（フォーカス不要）",
                                element_name=info.name if info else element_name,
                                control_type=info.control_type if info else "")
            except Exception as e:
                logger.debug(f"UIA Invoke 失敗: {e}")

        # --- 2. PostMessage ---
        if info:
            try:
                # 要素の中心座標（スクリーン座標）→ ウィンドウのクライアント座標に変換
                target_hwnd = info.hwnd
                if not target_hwnd:
                    # 要素自身に HWND がない場合、ウィンドウの HWND を使う
                    try:
                        target_hwnd = win.CurrentNativeWindowHandle
                    except Exception:
                        pass

                if target_hwnd:
                    # スクリーン座標 → クライアント座標
                    point = ctypes.wintypes.POINT(info.center_x, info.center_y)
                    ctypes.windll.user32.ScreenToClient(target_hwnd, ctypes.byref(point))

                    # ChildWindowFromPoint で正確な子ウィンドウを取得
                    child_hwnd = ctypes.windll.user32.ChildWindowFromPoint(
                        target_hwnd, point)
                    if child_hwnd:
                        # 子ウィンドウのクライアント座標に変換
                        screen_pt = ctypes.wintypes.POINT(info.center_x, info.center_y)
                        ctypes.windll.user32.ScreenToClient(child_hwnd, ctypes.byref(screen_pt))
                        lparam = _makelparam(screen_pt.x, screen_pt.y)
                        target_hwnd = child_hwnd
                    else:
                        lparam = _makelparam(point.x, point.y)

                    user32 = ctypes.windll.user32
                    user32.PostMessageW(target_hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
                    time.sleep(0.02)
                    user32.PostMessageW(target_hwnd, WM_LBUTTONUP, 0, lparam)

                    return FFResult(True, "postmessage",
                                    f"{el_desc} に PostMessage しました（フォーカス不要）",
                                    element_name=info.name, control_type=info.control_type)
            except Exception as e:
                logger.debug(f"PostMessage 失敗: {e}")

        # --- 3. SendInput（最終手段） ---
        if info:
            try:
                import pyautogui
                # カーソル位置を保存
                orig_x, orig_y = pyautogui.position()
                pyautogui.click(info.center_x, info.center_y)
                # カーソルを元に戻す
                pyautogui.moveTo(orig_x, orig_y, duration=0)
                return FFResult(True, "sendinput",
                                f"{el_desc} を SendInput でクリック（カーソル移動あり）",
                                element_name=info.name, control_type=info.control_type)
            except Exception as e:
                logger.debug(f"SendInput 失敗: {e}")

        return FFResult(False, "failed", f"{el_desc} の操作に全て失敗しました")

    def set_value(self, window_keyword: str, element_name: str,
                   value: str, control_type: str = "") -> FFResult:
        """要素にテキスト値を設定する（フォーカス不要）。

        1. UI Automation ValuePattern.SetValue（フォーカス不要）
        2. SendMessage WM_SETTEXT（フォーカス不要）
        3. フォーカス + type_text（最終手段）
        """
        win = self._find_window(window_keyword)
        if not win:
            return FFResult(False, "failed", f"ウィンドウ '{window_keyword}' が見つかりません")

        el = self._find_element_in_tree(win, element_name, control_type)
        if not el:
            return FFResult(False, "failed", f"要素 '{element_name}' が見つかりません")

        info = self._get_element_info(el)
        el_desc = f"'{info.name}' [{info.control_type}]" if info else element_name

        # --- 1. UI Automation ValuePattern ---
        if info and info.has_value:
            try:
                pattern = el.GetCurrentPattern(UIA_ValuePatternId)
                vp = pattern.QueryInterface(_uia_module.IUIAutomationValuePattern)
                vp.SetValue(value)
                return FFResult(True, "uia_value",
                                f"{el_desc} に '{value[:30]}' を設定（フォーカス不要）",
                                element_name=info.name, control_type=info.control_type)
            except Exception as e:
                logger.debug(f"UIA ValuePattern.SetValue 失敗: {e}")

        # --- 2. SendMessage WM_SETTEXT ---
        if info and info.hwnd:
            try:
                ctypes.windll.user32.SendMessageW(
                    info.hwnd, WM_SETTEXT, 0,
                    ctypes.c_wchar_p(value))
                return FFResult(True, "sendmessage",
                                f"{el_desc} に WM_SETTEXT で '{value[:30]}' を設定（フォーカス不要）",
                                element_name=info.name, control_type=info.control_type)
            except Exception as e:
                logger.debug(f"SendMessage WM_SETTEXT 失敗: {e}")

        return FFResult(False, "failed", f"{el_desc} への値設定に失敗")

    def toggle_element(self, window_keyword: str, element_name: str) -> FFResult:
        """チェックボックス等をトグルする（フォーカス不要）"""
        win = self._find_window(window_keyword)
        if not win:
            return FFResult(False, "failed", f"ウィンドウ '{window_keyword}' が見つかりません")

        el = self._find_element_in_tree(win, element_name)
        if not el:
            return FFResult(False, "failed", f"要素 '{element_name}' が見つかりません")

        info = self._get_element_info(el)
        if info and info.has_toggle:
            try:
                pattern = el.GetCurrentPattern(UIA_TogglePatternId)
                tp = pattern.QueryInterface(_uia_module.IUIAutomationTogglePattern)
                tp.Toggle()
                return FFResult(True, "uia_toggle",
                                f"'{info.name}' をトグル（フォーカス不要）",
                                element_name=info.name, control_type=info.control_type)
            except Exception as e:
                logger.debug(f"UIA Toggle 失敗: {e}")

        # フォールバック: click_element
        return self.click_element(window_keyword, element_name)

    def expand_element(self, window_keyword: str, element_name: str,
                        expand: bool = True) -> FFResult:
        """メニューやコンボボックスを開閉する（フォーカス不要）"""
        win = self._find_window(window_keyword)
        if not win:
            return FFResult(False, "failed", f"ウィンドウ '{window_keyword}' が見つかりません")

        el = self._find_element_in_tree(win, element_name)
        if not el:
            return FFResult(False, "failed", f"要素 '{element_name}' が見つかりません")

        info = self._get_element_info(el)
        if info and info.has_expand:
            try:
                pattern = el.GetCurrentPattern(UIA_ExpandCollapsePatternId)
                ec = pattern.QueryInterface(
                    _uia_module.IUIAutomationExpandCollapsePattern)
                if expand:
                    ec.Expand()
                else:
                    ec.Collapse()
                action = "展開" if expand else "折畳"
                return FFResult(True, "uia_expand",
                                f"'{info.name}' を{action}（フォーカス不要）",
                                element_name=info.name, control_type=info.control_type)
            except Exception as e:
                logger.debug(f"UIA ExpandCollapse 失敗: {e}")

        return FFResult(False, "failed",
                        f"'{element_name}' は ExpandCollapse に対応していません")
