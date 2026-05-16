"""
Screen Watcher — UI Automation + OpenCV 統合画面監視ワーカー

潜水艦のソナー担当のように、2つの感覚器で画面を常時監視する:
  - UI Automation: 構造化情報（ボタン名、テキスト値、座標） — 163ms/回
  - OpenCV: 視覚情報（色、変化検知、パターン） — 17ms/回

操作ワーカー（艦長/魚雷担当）は、スクリーンショット+LLM解析なしで
リアルタイムに画面状態を把握しながら操作できる。

使い方:
    watcher = ScreenWatcher("メモ帳")
    watcher.start()

    # ソナー報告（構造+視覚の統合レポート）
    print(watcher.report())

    # UI要素の検索（UI Automation）
    btn = watcher.find_element(name="保存")
    print(f"保存ボタン: ({btn.center_x}, {btn.center_y})")

    # カーソル位置の色（OpenCV）
    color = watcher.color_at_cursor()
    print(f"カーソル下: {color['name']}")

    # 指定領域の変化を監視（OpenCV）
    watcher.watch_region("canvas", x=100, y=100, w=800, h=600)
    changes = watcher.get_region_changes("canvas")

    # テキスト値の変化を待つ（UI Automation）
    watcher.wait_for_value("テキスト エディター", "入力完了")

    watcher.stop()
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import sys as _sys
import os as _os

import numpy as np
import cv2
import mss
import ctypes
import ctypes.wintypes

# video2aiのルートディレクトリをパスに追加（screenshot.py の検出関数を使うため）
_VIDEO2AI_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _VIDEO2AI_ROOT not in _sys.path:
    _sys.path.insert(0, _VIDEO2AI_ROOT)

from screenshot import _diff_score, _ssim_score, _optical_flow_score
from agent.change_detector import ChangeDetector

logger = logging.getLogger(__name__)

# === UI Automation セットアップ ===

import comtypes
import comtypes.client

_uia_module = comtypes.client.GetModule("UIAutomationCore.dll")


def _create_uia():
    """スレッドごとに IUIAutomation インスタンスを生成"""
    return comtypes.CoCreateInstance(
        _uia_module.CUIAutomation._reg_clsid_,
        interface=_uia_module.IUIAutomation,
    )


# Control Type ID → 名前
_CONTROL_TYPES = {
    50000: "Button", 50001: "Calendar", 50002: "CheckBox",
    50003: "ComboBox", 50004: "Edit", 50005: "Hyperlink",
    50006: "Document", 50007: "MenuItem", 50008: "Menu",
    50009: "MenuBar", 50010: "Group", 50011: "MenuItem",
    50012: "Image", 50013: "List", 50014: "ListItem",
    50016: "ProgressBar", 50017: "RadioButton", 50018: "Tab",
    50019: "TabItem", 50020: "Text", 50021: "Thumb",
    50025: "ToolBar", 50026: "StatusBar", 50030: "RichEdit",
    50031: "TitleBar", 50032: "Window", 50033: "Pane",
    50037: "Header",
}


# === データクラス ===

@dataclass
class UIElement:
    """UI 要素"""
    name: str
    control_type: str
    rect: tuple[int, int, int, int]  # (left, top, right, bottom)
    value: Optional[str] = None

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

    @property
    def is_visible(self) -> bool:
        return self.width > 0 and self.height > 0 and self.rect[0] >= -10000


@dataclass
class UIChange:
    """UI の変化"""
    timestamp: float
    change_type: str  # "value_changed", "appeared", "disappeared"
    element_name: str
    control_type: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None

    def __str__(self):
        if self.change_type == "value_changed":
            old = (self.old_value or "")[:40]
            new = (self.new_value or "")[:40]
            return f'[UI] "{self.element_name}" 値変化: "{old}" → "{new}"'
        elif self.change_type == "appeared":
            return f'[UI] "{self.element_name}" 出現'
        elif self.change_type == "disappeared":
            return f'[UI] "{self.element_name}" 消失'
        return f'[UI] {self.change_type}: {self.element_name}'


@dataclass
class VisualChange:
    """視覚的な変化"""
    timestamp: float
    region_name: str
    score: float  # 0.0-1.0
    description: str = ""

    def __str__(self):
        return f'[CV] 領域"{self.region_name}" 変化 (score={self.score:.4f}) {self.description}'


@dataclass
class WatchRegion:
    """監視対象の領域"""
    name: str
    x: int
    y: int
    w: int
    h: int
    threshold: float = 0.01  # 変化判定の閾値
    prev_frame: Optional[np.ndarray] = field(default=None, repr=False)


# === OpenCV 関数 ===

def _capture_region(x: int, y: int, w: int, h: int) -> np.ndarray:
    """指定領域をキャプチャ"""
    with mss.mss() as sct:
        img = sct.grab({"left": x, "top": y, "width": w, "height": h})
        return np.array(img)[:, :, :3]


def _get_cursor_pos() -> tuple[int, int]:
    point = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return (point.x, point.y)


def _color_name(h: int, s: int, v: int) -> str:
    if v < 30: return "黒"
    if s < 30:
        return "白" if v > 200 else "灰"
    if h < 10 or h > 170: return "赤"
    if h < 25: return "オレンジ"
    if h < 35: return "黄"
    if h < 80: return "緑"
    if h < 100: return "シアン"
    if h < 130: return "青"
    if h < 160: return "紫"
    return "ピンク"


# === UI Automation 走査 ===

def _extract_element(uia, com_element) -> Optional[UIElement]:
    try:
        name = com_element.CurrentName or ""
        ct_id = com_element.CurrentControlType
        ct = _CONTROL_TYPES.get(ct_id, f"Unknown({ct_id})")
        rect = com_element.CurrentBoundingRectangle
        rect_tuple = (rect.left, rect.top, rect.right, rect.bottom)

        value = None
        try:
            vp = com_element.GetCurrentPattern(10002)
            if vp:
                vi = vp.QueryInterface(_uia_module.IUIAutomationValuePattern)
                value = vi.CurrentValue
        except Exception:
            pass

        return UIElement(
            name=name[:200], control_type=ct,
            rect=rect_tuple, value=value[:500] if value else None,
        )
    except Exception:
        return None


def _scan_tree(uia, element, results: list, max_depth: int = 4, depth: int = 0):
    el = _extract_element(uia, element)
    if el:
        results.append(el)
    if depth >= max_depth:
        return
    try:
        condition = uia.CreateTrueCondition()
        walker = uia.CreateTreeWalker(condition)
        child = walker.GetFirstChildElement(element)
        while child:
            _scan_tree(uia, child, results, max_depth, depth + 1)
            try:
                child = walker.GetNextSiblingElement(child)
            except Exception:
                break
    except Exception:
        pass


# === メインクラス ===

class ScreenWatcher:
    """UI Automation + OpenCV 統合画面監視ワーカー"""

    def __init__(self, window_keyword: str, ui_interval: float = 0.1,
                 cv_interval: float = 0.02, max_depth: int = 4):
        """
        Args:
            window_keyword: 監視するウィンドウタイトルのキーワード
            ui_interval: UI Automation のスキャン間隔（秒）
            cv_interval: OpenCV のスキャン間隔（秒）
            max_depth: UI ツリーの最大探索深度
        """
        self.window_keyword = window_keyword
        self.ui_interval = ui_interval
        self.cv_interval = cv_interval
        self.max_depth = max_depth

        # UI Automation 状態
        self._uia = None
        self._window_element = None
        self._ui_elements: list[UIElement] = []
        self._prev_ui_map: dict[str, UIElement] = {}
        self._ui_changes: list[UIChange] = []

        # OpenCV 状態
        self._watch_regions: dict[str, WatchRegion] = {}
        self._visual_changes: list[VisualChange] = []

        # 共通
        self._lock = threading.Lock()
        self._running = False
        self._ui_thread: Optional[threading.Thread] = None
        self._cv_thread: Optional[threading.Thread] = None
        self._changes_cursor_ui = 0
        self._changes_cursor_cv = 0

        # 統計
        self._ui_scan_count = 0
        self._ui_total_time = 0.0
        self._cv_scan_count = 0
        self._cv_total_time = 0.0

    # --- UI Automation スレッド ---

    def _ui_find_window(self) -> bool:
        root = self._uia.GetRootElement()
        condition = self._uia.CreateTrueCondition()
        walker = self._uia.CreateTreeWalker(condition)
        child = walker.GetFirstChildElement(root)
        while child:
            try:
                name = child.CurrentName
                if name and self.window_keyword in name:
                    self._window_element = child
                    return True
            except Exception:
                pass
            try:
                child = walker.GetNextSiblingElement(child)
            except Exception:
                break
        return False

    def _ui_element_key(self, el: UIElement) -> str:
        return f"{el.control_type}::{el.name}"

    def _ui_detect_changes(self, new_elements: list[UIElement]):
        new_map = {}
        for el in new_elements:
            key = self._ui_element_key(el)
            new_map[key] = el

        now = time.time()

        for key, old_el in self._prev_ui_map.items():
            if key not in new_map and old_el.is_visible:
                self._ui_changes.append(UIChange(
                    timestamp=now, change_type="disappeared",
                    element_name=old_el.name, control_type=old_el.control_type,
                ))

        for key, new_el in new_map.items():
            if not new_el.is_visible:
                continue
            if key not in self._prev_ui_map:
                if self._ui_scan_count > 1:
                    self._ui_changes.append(UIChange(
                        timestamp=now, change_type="appeared",
                        element_name=new_el.name, control_type=new_el.control_type,
                        new_value=new_el.value,
                    ))
            else:
                old_el = self._prev_ui_map[key]
                if new_el.value != old_el.value and (new_el.value or old_el.value):
                    self._ui_changes.append(UIChange(
                        timestamp=now, change_type="value_changed",
                        element_name=new_el.name, control_type=new_el.control_type,
                        old_value=old_el.value, new_value=new_el.value,
                    ))

        self._prev_ui_map = new_map
        # 古い変化を刈り込み
        if len(self._ui_changes) > 200:
            excess = len(self._ui_changes) - 200
            self._ui_changes = self._ui_changes[excess:]
            self._changes_cursor_ui = max(0, self._changes_cursor_ui - excess)

    def _ui_loop(self):
        comtypes.CoInitialize()
        self._uia = _create_uia()
        self._window_element = None

        try:
            while self._running:
                try:
                    if not self._window_element:
                        if not self._ui_find_window():
                            time.sleep(0.5)
                            continue

                    t0 = time.perf_counter()
                    elements = []
                    _scan_tree(self._uia, self._window_element, elements,
                               max_depth=self.max_depth)
                    elapsed = (time.perf_counter() - t0) * 1000

                    self._ui_scan_count += 1
                    self._ui_total_time += elapsed

                    with self._lock:
                        self._ui_detect_changes(elements)
                        self._ui_elements = elements

                except Exception as e:
                    logger.debug(f"UI scan error: {e}")
                    self._window_element = None  # 再検索

                time.sleep(self.ui_interval)
        finally:
            self._uia = None
            comtypes.CoUninitialize()

    # --- OpenCV スレッド ---

    def _cv_loop(self):
        while self._running:
            try:
                t0 = time.perf_counter()
                with self._lock:
                    regions = list(self._watch_regions.values())

                for region in regions:
                    frame = _capture_region(region.x, region.y, region.w, region.h)

                    if region.prev_frame is not None and frame.shape == region.prev_frame.shape:
                        # 3方式OR変化検出: diff=色変化、ssim=構造変化、optical-flow=動き
                        prev_gray = cv2.cvtColor(region.prev_frame, cv2.COLOR_BGR2GRAY)
                        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        score_diff = _diff_score(region.prev_frame, frame)
                        score_ssim = _ssim_score(prev_gray, curr_gray)
                        score_flow = _optical_flow_score(prev_gray, curr_gray)

                        # OR条件: どれか1つでも閾値超えで変化と判定
                        # 各方式の閾値は ChangeDetector.DEFAULT_THRESHOLDS に準拠
                        thresholds = ChangeDetector.DEFAULT_THRESHOLDS
                        changed = (
                            score_diff >= thresholds["diff"]
                            or score_ssim >= thresholds["ssim"]
                            or score_flow >= thresholds["optical-flow"]
                        )
                        score = max(score_diff, score_ssim, score_flow)

                        if changed:
                            detail = f"diff={score_diff:.4f} ssim={score_ssim:.4f} flow={score_flow:.4f}"
                            with self._lock:
                                self._visual_changes.append(VisualChange(
                                    timestamp=time.time(),
                                    region_name=region.name,
                                    score=score,
                                    description=detail,
                                ))
                                if len(self._visual_changes) > 200:
                                    excess = len(self._visual_changes) - 200
                                    self._visual_changes = self._visual_changes[excess:]
                                    self._changes_cursor_cv = max(0, self._changes_cursor_cv - excess)

                    region.prev_frame = frame

                elapsed = (time.perf_counter() - t0) * 1000
                self._cv_scan_count += 1
                self._cv_total_time += elapsed

            except Exception as e:
                logger.debug(f"CV scan error: {e}")

            time.sleep(self.cv_interval)

    # === 公開 API ===

    def start(self):
        """監視を開始（UIスレッド + CVスレッド）"""
        if self._running:
            return

        # ウィンドウ存在チェック（メインスレッド）
        uia = _create_uia()
        root = uia.GetRootElement()
        condition = uia.CreateTrueCondition()
        walker = uia.CreateTreeWalker(condition)
        found = False
        child = walker.GetFirstChildElement(root)
        while child:
            try:
                name = child.CurrentName
                if name and self.window_keyword in name:
                    logger.info(f"ScreenWatcher 開始: '{name}'")
                    found = True
                    break
            except Exception:
                pass
            try:
                child = walker.GetNextSiblingElement(child)
            except Exception:
                break

        if not found:
            raise RuntimeError(f"ウィンドウ '{self.window_keyword}' が見つかりません")

        self._running = True

        self._ui_thread = threading.Thread(target=self._ui_loop, daemon=True, name="ui-watcher")
        self._ui_thread.start()

        self._cv_thread = threading.Thread(target=self._cv_loop, daemon=True, name="cv-watcher")
        self._cv_thread.start()

    def stop(self):
        """監視を停止"""
        self._running = False
        if self._ui_thread:
            self._ui_thread.join(timeout=2.0)
        if self._cv_thread:
            self._cv_thread.join(timeout=2.0)

        ui_avg = self._ui_total_time / max(1, self._ui_scan_count)
        cv_avg = self._cv_total_time / max(1, self._cv_scan_count)
        logger.info(
            f"ScreenWatcher 停止 — "
            f"UI: {self._ui_scan_count}回 avg={ui_avg:.0f}ms | "
            f"CV: {self._cv_scan_count}回 avg={cv_avg:.0f}ms"
        )

    def is_running(self) -> bool:
        return self._running

    # --- UI Automation クエリ ---

    def find_element(self, name: str = "", control_type: str = "") -> Optional[UIElement]:
        """名前や型で UI 要素を検索（部分一致）"""
        with self._lock:
            for el in self._ui_elements:
                if not el.is_visible:
                    continue
                if name and name not in el.name:
                    continue
                if control_type and control_type != el.control_type:
                    continue
                return el
        return None

    def find_elements(self, name: str = "", control_type: str = "") -> list[UIElement]:
        """名前や型で UI 要素を複数検索"""
        with self._lock:
            return [el for el in self._ui_elements
                    if el.is_visible
                    and (not name or name in el.name)
                    and (not control_type or control_type == el.control_type)]

    def get_value(self, element_name: str) -> Optional[str]:
        """指定名の要素の値を取得"""
        el = self.find_element(name=element_name)
        return el.value if el else None

    def get_all_values(self) -> dict[str, str]:
        """値を持つ全要素を {名前: 値} で返す"""
        with self._lock:
            return {el.name: el.value for el in self._ui_elements
                    if el.value and el.is_visible}

    def get_status_texts(self) -> list[str]:
        """ステータスバー等のテキスト要素を返す"""
        with self._lock:
            return [el.name for el in self._ui_elements
                    if el.control_type == "Text" and el.is_visible and el.name.strip()]

    # --- OpenCV クエリ ---

    def watch_region(self, name: str, x: int, y: int, w: int, h: int,
                     threshold: float = 0.01):
        """監視領域を追加"""
        with self._lock:
            self._watch_regions[name] = WatchRegion(
                name=name, x=x, y=y, w=w, h=h, threshold=threshold,
            )

    def unwatch_region(self, name: str):
        """監視領域を削除"""
        with self._lock:
            self._watch_regions.pop(name, None)

    def color_at_cursor(self, radius: int = 1) -> dict:
        """カーソル位置の色情報（即時取得、スレッド不要）"""
        cx, cy = _get_cursor_pos()
        size = radius * 2 + 1
        img = _capture_region(cx - radius, cy - radius, size, size)
        b, g, r = img[radius, radius]

        pixel = img[radius:radius + 1, radius:radius + 1]
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0, 0]

        return {
            "pos": (cx, cy),
            "rgb": (int(r), int(g), int(b)),
            "hsv": (int(h), int(s), int(v)),
            "name": _color_name(int(h), int(s), int(v)),
        }

    def color_at(self, x: int, y: int) -> dict:
        """指定座標の色情報（即時取得）"""
        img = _capture_region(x, y, 1, 1)
        b, g, r = img[0, 0]
        hsv = cv2.cvtColor(img.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0, 0]
        return {
            "pos": (x, y),
            "rgb": (int(r), int(g), int(b)),
            "hsv": (int(h), int(s), int(v)),
            "name": _color_name(int(h), int(s), int(v)),
        }

    def screenshot(self, monitor: int = 0, resize_width: int = 1920) -> np.ndarray:
        """
        ScreenWatcher の CV パイプライン内でスクリーンショットを撮影する。

        通常の screenshot MCP ツールと異なり、潜水艦モデルの一部として動作:
        - AI が「今確認したい」と要請したタイミングでのみ実行
        - 撮影後、全 watch region の基準画像を同時更新（変化検出のリセット）
        - 定期スクリーンショットではない（それは Computer Use の劣化版）

        Parameters
        ----------
        monitor : int
            モニター番号 (0=全画面, 1=プライマリ, 2,3...=その他)
        resize_width : int
            リサイズ幅 (0=オリジナル解像度)

        Returns
        -------
        np.ndarray
            BGR 形式のスクリーンショット画像
        """
        from agent.screen_capture import ScreenCapture

        cap = ScreenCapture(monitor_index=monitor, resize_width=resize_width)
        frame = cap.capture()  # np.ndarray (BGR)

        # watch region の基準画像を更新（変化検出のベースラインをリセット）
        with self._lock:
            for region in self._watch_regions.values():
                region.prev_frame = _capture_region(region.x, region.y, region.w, region.h)

        return frame

    def region_snapshot(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """指定領域のスナップショットを取得"""
        return _capture_region(x, y, w, h)

    def compare_with_snapshot(self, x: int, y: int, w: int, h: int,
                              snapshot: np.ndarray) -> float:
        """現在の領域とスナップショットを比較して変化スコアを返す"""
        current = _capture_region(x, y, w, h)
        if current.shape != snapshot.shape:
            return 1.0
        gray1 = cv2.cvtColor(snapshot, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        return float(np.mean(cv2.absdiff(gray1, gray2))) / 255.0

    def find_color(self, x: int, y: int, w: int, h: int,
                   hsv_low: tuple, hsv_high: tuple, min_area: int = 10) -> list[dict]:
        """指定領域内で特定色の塊を検出"""
        frame = _capture_region(x, y, w, h)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            results.append({
                "center": (x + bx + bw // 2, y + by + bh // 2),
                "area": int(area),
                "rect": (x + bx, y + by, bw, bh),
            })
        results.sort(key=lambda r: r["area"], reverse=True)
        return results

    # --- 変化取得 ---

    def get_changes(self) -> list:
        """前回呼び出し以降の全変化（UI + CV）を時系列で返す"""
        with self._lock:
            ui = self._ui_changes[self._changes_cursor_ui:]
            cv = self._visual_changes[self._changes_cursor_cv:]
            self._changes_cursor_ui = len(self._ui_changes)
            self._changes_cursor_cv = len(self._visual_changes)

        combined = []
        combined.extend((c.timestamp, c) for c in ui)
        combined.extend((c.timestamp, c) for c in cv)
        combined.sort(key=lambda x: x[0])
        return [c for _, c in combined]

    def get_ui_changes(self) -> list[UIChange]:
        """UI変化のみ取得"""
        with self._lock:
            changes = self._ui_changes[self._changes_cursor_ui:]
            self._changes_cursor_ui = len(self._ui_changes)
            return list(changes)

    def get_visual_changes(self) -> list[VisualChange]:
        """視覚変化のみ取得"""
        with self._lock:
            changes = self._visual_changes[self._changes_cursor_cv:]
            self._changes_cursor_cv = len(self._visual_changes)
            return list(changes)

    # --- 待機 ---

    def wait_for_change(self, timeout: float = 5.0) -> list:
        """何らかの変化が起きるまで待機"""
        self.get_changes()  # 既存をクリア
        start = time.time()
        while time.time() - start < timeout:
            changes = self.get_changes()
            if changes:
                return changes
            time.sleep(0.02)
        return []

    def wait_for_value(self, element_name: str, expected: str,
                       timeout: float = 5.0, contains: bool = True) -> bool:
        """要素の値が期待値になるまで待機"""
        start = time.time()
        while time.time() - start < timeout:
            val = self.get_value(element_name)
            if val is not None:
                if (contains and expected in val) or (not contains and val == expected):
                    return True
            time.sleep(0.05)
        return False

    def wait_for_visual_change(self, region_name: str, timeout: float = 5.0) -> bool:
        """指定領域に視覚的変化が起きるまで待機"""
        self.get_visual_changes()  # クリア
        start = time.time()
        while time.time() - start < timeout:
            changes = self.get_visual_changes()
            for c in changes:
                if c.region_name == region_name:
                    return True
            time.sleep(0.02)
        return False

    # --- レポート ---

    def report(self) -> str:
        """ソナー報告: 現在の状態 + 直近の変化"""
        lines = []

        # UI 状態
        with self._lock:
            visible = [e for e in self._ui_elements if e.is_visible]

        if visible:
            buttons = [e for e in visible if e.control_type == "Button"]
            values = [e for e in visible if e.value]
            lines.append(f"UI要素: {len(visible)}個 (ボタン{len(buttons)}, 値あり{len(values)})")
            for e in values[:3]:
                lines.append(f"  [{e.control_type}] {e.name}: {(e.value or '')[:60]}")

        # ステータス
        statuses = self.get_status_texts()
        if statuses:
            useful = [s for s in statuses if any(c.isdigit() for c in s) or len(s) > 3]
            if useful:
                lines.append(f"ステータス: {' | '.join(useful[:5])}")

        # 監視領域
        with self._lock:
            regions = list(self._watch_regions.keys())
        if regions:
            lines.append(f"監視領域: {', '.join(regions)}")

        # 変化
        changes = self.get_changes()
        if changes:
            lines.append(f"--- 変化 ({len(changes)}件) ---")
            for c in changes[-10:]:
                lines.append(f"  {c}")

        # 統計
        ui_avg = self._ui_total_time / max(1, self._ui_scan_count)
        cv_avg = self._cv_total_time / max(1, self._cv_scan_count)
        lines.append(f"[UI {ui_avg:.0f}ms/回 | CV {cv_avg:.0f}ms/回]")

        return "\n".join(lines)

    def stats(self) -> dict:
        """統計情報"""
        return {
            "ui_scans": self._ui_scan_count,
            "ui_avg_ms": self._ui_total_time / max(1, self._ui_scan_count),
            "cv_scans": self._cv_scan_count,
            "cv_avg_ms": self._cv_total_time / max(1, self._cv_scan_count),
            "ui_changes": len(self._ui_changes),
            "cv_changes": len(self._visual_changes),
            "watch_regions": len(self._watch_regions),
        }


# === CLI テスト ===

def main():
    import sys
    keyword = sys.argv[1] if len(sys.argv) > 1 else "メモ帳"

    logging.basicConfig(level=logging.INFO)
    print(f"=== ScreenWatcher 統合テスト: '{keyword}' ===\n")

    watcher = ScreenWatcher(keyword, ui_interval=0.1, cv_interval=0.02)

    try:
        watcher.start()
    except RuntimeError as e:
        print(f"エラー: {e}")
        return

    time.sleep(0.5)

    # 1. UI 要素
    print("--- UI 要素 ---")
    buttons = watcher.find_elements(control_type="Button")
    for b in buttons[:5]:
        if b.name:
            print(f"  [{b.name}] ({b.center_x},{b.center_y}) {b.width}x{b.height}")
    print(f"  ... 他 {max(0, len(buttons)-5)} 個\n")

    # 2. テキスト値
    print("--- テキスト値 ---")
    for name, val in watcher.get_all_values().items():
        print(f"  {name or '(無名)'}: {val[:60]}")
    print()

    # 3. カーソル色
    print("--- カーソル色 ---")
    color = watcher.color_at_cursor()
    print(f"  位置: {color['pos']}, RGB: {color['rgb']}, 色: {color['name']}\n")

    # 4. ウィンドウ領域を監視
    el = watcher.find_element(control_type="Window")
    if el and el.is_visible:
        watcher.watch_region("window",
                             el.rect[0], el.rect[1], el.width, el.height)
        print(f"--- 監視領域追加: window ({el.width}x{el.height}) ---\n")

    # 5. 20秒間変化監視
    print("20秒間、変化を監視中...")
    print("  ウィンドウで何か操作してください\n")

    watcher.get_changes()
    start = time.time()
    while time.time() - start < 20:
        changes = watcher.get_changes()
        for c in changes:
            elapsed = time.time() - start
            print(f"  [{elapsed:.1f}s] {c}")
        time.sleep(0.1)

    # 統計
    s = watcher.stats()
    print(f"\n=== 統計 ===")
    print(f"UI: {s['ui_scans']}回, avg={s['ui_avg_ms']:.0f}ms, 変化{s['ui_changes']}件")
    print(f"CV: {s['cv_scans']}回, avg={s['cv_avg_ms']:.0f}ms, 変化{s['cv_changes']}件")

    watcher.stop()
    print("完了")


if __name__ == "__main__":
    main()
