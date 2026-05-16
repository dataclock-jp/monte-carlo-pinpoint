"""
UI Watcher — UI Automation API による連続画面監視ワーカー

潜水艦のソナー担当のように、画面の状態を常時監視し、
変化があれば報告する。操作ワーカーはスクリーンショットなしで
UI の状態を把握できる。

使い方:
    watcher = UIWatcher("メモ帳")
    watcher.start()

    # 現在の状態を取得
    state = watcher.get_state()

    # 変化を取得（前回取得以降の差分）
    changes = watcher.get_changes()

    # 特定の要素を検索
    element = watcher.find_element(name="ファイル")

    # テキスト値を取得
    value = watcher.get_value("テキスト エディター")

    watcher.stop()
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import comtypes
import comtypes.client

logger = logging.getLogger(__name__)

# UI Automation の COM 型情報（モジュールレベルで1回ロード）
_uia_module = comtypes.client.GetModule("UIAutomationCore.dll")


def _create_uia():
    """スレッドごとに IUIAutomation インスタンスを生成する。
    COM オブジェクトはスレッド間で共有できないため、
    各スレッドで独自のインスタンスが必要。"""
    return comtypes.CoCreateInstance(
        _uia_module.CUIAutomation._reg_clsid_,
        interface=_uia_module.IUIAutomation,
    )

# Control Type ID → 名前
_CONTROL_TYPES = {
    50000: "Button",
    50001: "Calendar",
    50002: "CheckBox",
    50003: "ComboBox",
    50004: "Edit",
    50005: "Hyperlink",
    50006: "Document",
    50007: "MenuItem",
    50008: "Menu",
    50009: "MenuBar",
    50010: "Group",
    50011: "MenuItem",
    50012: "Image",
    50013: "List",
    50014: "ListItem",
    50016: "ProgressBar",
    50017: "RadioButton",
    50018: "Tab",
    50019: "TabItem",
    50020: "Text",
    50021: "Thumb",
    50025: "ToolBar",
    50026: "StatusBar",
    50030: "ScrollBar",
    50031: "TitleBar",
    50032: "Window",
    50033: "Pane",
    50037: "Header",
}


@dataclass
class UIElement:
    """UI 要素の情報"""
    name: str
    control_type: str
    rect: tuple[int, int, int, int]  # (left, top, right, bottom)
    value: Optional[str] = None
    children_count: int = 0
    # 要素の中心座標（クリック用）
    center_x: int = 0
    center_y: int = 0

    def __post_init__(self):
        self.center_x = (self.rect[0] + self.rect[2]) // 2
        self.center_y = (self.rect[1] + self.rect[3]) // 2

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    @property
    def is_visible(self) -> bool:
        """画面上に実際に表示されているか（座標が有効か）"""
        return self.width > 0 and self.height > 0 and self.rect[0] >= -10000


@dataclass
class UIChange:
    """UI の変化を表す"""
    timestamp: float
    change_type: str  # "value_changed", "appeared", "disappeared", "moved"
    element_name: str
    control_type: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    details: str = ""

    def __str__(self):
        if self.change_type == "value_changed":
            return f'[{self.control_type}] "{self.element_name}" 値変化: "{self.old_value}" → "{self.new_value}"'
        elif self.change_type == "appeared":
            return f'[{self.control_type}] "{self.element_name}" 出現'
        elif self.change_type == "disappeared":
            return f'[{self.control_type}] "{self.element_name}" 消失'
        elif self.change_type == "moved":
            return f'[{self.control_type}] "{self.element_name}" 移動: {self.details}'
        return f'{self.change_type}: {self.element_name}'


@dataclass
class UIState:
    """UI の全体状態のスナップショット"""
    window_title: str
    timestamp: float
    elements: list[UIElement] = field(default_factory=list)
    scan_time_ms: float = 0.0

    def summary(self) -> str:
        """状態の要約（ソナー報告形式）"""
        visible = [e for e in self.elements if e.is_visible]
        buttons = [e for e in visible if e.control_type == "Button"]
        texts = [e for e in visible if e.control_type == "Text"]
        edits = [e for e in visible if e.control_type in ("Edit", "Document")]
        values = [e for e in visible if e.value is not None]

        lines = [
            f"ウィンドウ: {self.window_title}",
            f"要素数: {len(visible)} (全{len(self.elements)})",
            f"  ボタン: {len(buttons)}個",
            f"  テキスト: {len(texts)}個",
            f"  入力欄: {len(edits)}個",
        ]
        if values:
            lines.append("  値あり:")
            for e in values[:5]:
                v = e.value[:80] if e.value else ""
                lines.append(f'    [{e.control_type}] "{e.name}": "{v}"')
        lines.append(f"取得時間: {self.scan_time_ms:.1f}ms")
        return "\n".join(lines)


def _extract_element(uia, com_element, max_depth: int = 0, current_depth: int = 0) -> Optional[UIElement]:
    """COM 要素から UIElement を抽出。uia は呼び出し元スレッドの IUIAutomation インスタンス。"""
    try:
        name = com_element.CurrentName or ""
        control_type_id = com_element.CurrentControlType
        control_type = _CONTROL_TYPES.get(control_type_id, f"Unknown({control_type_id})")
        rect = com_element.CurrentBoundingRectangle
        rect_tuple = (rect.left, rect.top, rect.right, rect.bottom)

        value = None
        try:
            val_pattern = com_element.GetCurrentPattern(10002)  # UIA_ValuePatternId
            if val_pattern:
                val_iface = val_pattern.QueryInterface(
                    _uia_module.IUIAutomationValuePattern
                )
                value = val_iface.CurrentValue
        except Exception:
            pass

        # 子要素数をカウント（浅く）
        children_count = 0
        if current_depth < max_depth:
            try:
                condition = uia.CreateTrueCondition()
                walker = uia.CreateTreeWalker(condition)
                child = walker.GetFirstChildElement(com_element)
                while child:
                    children_count += 1
                    child = walker.GetNextSiblingElement(child)
            except Exception:
                pass

        return UIElement(
            name=name[:200],
            control_type=control_type,
            rect=rect_tuple,
            value=value[:500] if value else None,
            children_count=children_count,
        )
    except Exception:
        return None


def _scan_tree(uia, element, results: list, max_depth: int = 4, depth: int = 0):
    """UI 要素ツリーを再帰走査。uia は呼び出し元スレッドの IUIAutomation インスタンス。"""
    el = _extract_element(uia, element, max_depth=max_depth, current_depth=depth)
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


class UIWatcher:
    """UI Automation による連続画面監視ワーカー

    バックグラウンドスレッドで指定ウィンドウの UI 状態を継続的に監視し、
    変化があれば記録する。操作ワーカーは任意のタイミングで状態や変化を
    問い合わせることができる。
    """

    def __init__(self, window_keyword: str, scan_interval: float = 0.05,
                 max_depth: int = 4, max_changes: int = 100):
        """
        Args:
            window_keyword: 監視するウィンドウタイトルのキーワード
            scan_interval: スキャン間隔（秒）。デフォルト50ms
            max_depth: UI ツリーの最大探索深度
            max_changes: 保持する変化履歴の最大数
        """
        self.window_keyword = window_keyword
        self.scan_interval = scan_interval
        self.max_depth = max_depth
        self.max_changes = max_changes

        self._window_element = None
        self._current_state: Optional[UIState] = None
        self._prev_elements: dict[str, UIElement] = {}  # key → UIElement
        self._changes: list[UIChange] = []
        self._changes_cursor: int = 0  # get_changes 用の読み取り位置

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._scan_count = 0
        self._total_scan_time = 0.0
        self._uia = None  # スレッドごとの IUIAutomation インスタンス

    def _ensure_uia(self):
        """現在のスレッド用の UIA インスタンスを確保"""
        if self._uia is None:
            self._uia = _create_uia()
        return self._uia

    def _find_window(self):
        """ウィンドウ要素を検索"""
        uia = self._ensure_uia()
        root = uia.GetRootElement()
        condition = uia.CreateTrueCondition()
        walker = uia.CreateTreeWalker(condition)

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

    def _element_key(self, el: UIElement) -> str:
        """要素の同一性を判定するキー"""
        # 同じ名前+型+位置の要素を同一とみなす
        return f"{el.control_type}::{el.name}::{el.rect[0]},{el.rect[1]}"

    def _detect_changes(self, new_elements: list[UIElement]):
        """前回の状態との差分を検出"""
        new_map: dict[str, UIElement] = {}
        for el in new_elements:
            key = self._element_key(el)
            new_map[key] = el

        now = time.time()

        # 消失した要素
        for key, old_el in self._prev_elements.items():
            if key not in new_map and old_el.is_visible:
                self._add_change(UIChange(
                    timestamp=now,
                    change_type="disappeared",
                    element_name=old_el.name,
                    control_type=old_el.control_type,
                ))

        # 出現した要素 / 値の変化
        for key, new_el in new_map.items():
            if not new_el.is_visible:
                continue

            if key not in self._prev_elements:
                # 新しい要素が出現（初回スキャンは除外）
                if self._scan_count > 1:
                    self._add_change(UIChange(
                        timestamp=now,
                        change_type="appeared",
                        element_name=new_el.name,
                        control_type=new_el.control_type,
                        new_value=new_el.value,
                    ))
            else:
                old_el = self._prev_elements[key]
                # 値の変化
                if new_el.value != old_el.value and (new_el.value or old_el.value):
                    self._add_change(UIChange(
                        timestamp=now,
                        change_type="value_changed",
                        element_name=new_el.name,
                        control_type=new_el.control_type,
                        old_value=old_el.value,
                        new_value=new_el.value,
                    ))

        self._prev_elements = new_map

    def _add_change(self, change: UIChange):
        """変化を記録"""
        self._changes.append(change)
        if len(self._changes) > self.max_changes:
            # 古いものを削除、カーソルも調整
            excess = len(self._changes) - self.max_changes
            self._changes = self._changes[excess:]
            self._changes_cursor = max(0, self._changes_cursor - excess)

    def _scan_loop(self):
        """バックグラウンドスキャンループ"""
        # COM をこのスレッドで初期化し、スレッド専用の UIA インスタンスを作る
        comtypes.CoInitialize()
        self._uia = _create_uia()
        self._window_element = None  # メインスレッドの参照は使えない

        try:
            while self._running:
                try:
                    self._scan_once()
                except Exception as e:
                    logger.debug(f"UIWatcher scan error: {e}")

                time.sleep(self.scan_interval)
        finally:
            self._uia = None
            comtypes.CoUninitialize()

    def _scan_once(self):
        """1回のスキャン"""
        if not self._window_element:
            if not self._find_window():
                return

        t0 = time.perf_counter()
        elements_list = []
        _scan_tree(self._uia, self._window_element, elements_list, max_depth=self.max_depth)
        scan_time = (time.perf_counter() - t0) * 1000

        self._scan_count += 1
        self._total_scan_time += scan_time

        with self._lock:
            # 変化検出
            self._detect_changes(elements_list)

            # 状態更新
            window_title = ""
            try:
                window_title = self._window_element.CurrentName or ""
            except Exception:
                pass

            self._current_state = UIState(
                window_title=window_title,
                timestamp=time.time(),
                elements=elements_list,
                scan_time_ms=scan_time,
            )

    # === 公開 API ===

    def start(self):
        """監視を開始"""
        if self._running:
            return

        # メインスレッドでウィンドウの存在だけ確認
        # 実際の COM 操作はバックグラウンドスレッドで行う
        self._uia = _create_uia()
        if not self._find_window():
            self._uia = None
            raise RuntimeError(f"ウィンドウ '{self.window_keyword}' が見つかりません")

        window_name = ""
        try:
            window_name = self._window_element.CurrentName
        except Exception:
            pass
        logger.info(f"UIWatcher 開始: '{window_name}'")
        # メインスレッドの参照はクリア（バックグラウンドスレッドで再取得する）
        self._window_element = None
        self._uia = None

        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """監視を停止"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info(f"UIWatcher 停止 (スキャン{self._scan_count}回, "
                     f"平均{self._total_scan_time / max(1, self._scan_count):.1f}ms)")

    def is_running(self) -> bool:
        return self._running

    def get_state(self) -> Optional[UIState]:
        """現在の UI 状態を取得"""
        with self._lock:
            return self._current_state

    def get_changes(self) -> list[UIChange]:
        """前回呼び出し以降の変化を取得"""
        with self._lock:
            new_changes = self._changes[self._changes_cursor:]
            self._changes_cursor = len(self._changes)
            return list(new_changes)

    def get_all_changes(self) -> list[UIChange]:
        """全ての変化履歴を取得"""
        with self._lock:
            return list(self._changes)

    def find_element(self, name: str = "", control_type: str = "") -> Optional[UIElement]:
        """名前や型で要素を検索（部分一致）"""
        with self._lock:
            if not self._current_state:
                return None
            for el in self._current_state.elements:
                if not el.is_visible:
                    continue
                if name and name not in el.name:
                    continue
                if control_type and control_type != el.control_type:
                    continue
                return el
            return None

    def find_elements(self, name: str = "", control_type: str = "") -> list[UIElement]:
        """名前や型で要素を複数検索（部分一致）"""
        with self._lock:
            if not self._current_state:
                return []
            results = []
            for el in self._current_state.elements:
                if not el.is_visible:
                    continue
                if name and name not in el.name:
                    continue
                if control_type and control_type != el.control_type:
                    continue
                results.append(el)
            return results

    def get_value(self, element_name: str) -> Optional[str]:
        """指定名の要素の値を取得"""
        el = self.find_element(name=element_name)
        if el:
            return el.value
        return None

    def wait_for_change(self, timeout: float = 5.0) -> list[UIChange]:
        """変化が起きるまで待機（タイムアウト付き）"""
        start = time.time()
        # まず既存の未読変化をクリア
        self.get_changes()
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
                if contains and expected in val:
                    return True
                elif not contains and val == expected:
                    return True
            time.sleep(0.02)
        return False

    def report(self) -> str:
        """現在の状態と直近の変化をテキスト報告（ソナー報告）"""
        lines = []
        state = self.get_state()
        if state:
            lines.append(state.summary())

        changes = self.get_changes()
        if changes:
            lines.append(f"\n--- 変化 ({len(changes)}件) ---")
            for c in changes[-10:]:  # 最新10件
                lines.append(f"  {c}")

        if not lines:
            lines.append("(状態未取得)")

        return "\n".join(lines)


# === CLI テスト用 ===

def main():
    import sys
    keyword = sys.argv[1] if len(sys.argv) > 1 else "メモ帳"

    logging.basicConfig(level=logging.INFO)
    print(f"=== UIWatcher テスト: '{keyword}' ===\n")

    watcher = UIWatcher(keyword, scan_interval=0.05)

    try:
        watcher.start()
    except RuntimeError as e:
        print(f"エラー: {e}")
        return

    # 初回状態
    time.sleep(0.2)  # 最初のスキャン完了を待つ
    print(watcher.report())
    print()

    # 変化の監視（10秒間）
    print("=== 10秒間、変化を監視します（ウィンドウで何か操作してください）===\n")
    start = time.time()
    while time.time() - start < 10:
        changes = watcher.get_changes()
        for c in changes:
            elapsed = time.time() - start
            print(f"  [{elapsed:.1f}s] {c}")
        time.sleep(0.1)

    print()
    print(f"=== 終了 ===")
    watcher.stop()

    # 統計
    avg_ms = watcher._total_scan_time / max(1, watcher._scan_count)
    print(f"スキャン回数: {watcher._scan_count}")
    print(f"平均スキャン時間: {avg_ms:.1f}ms")
    print(f"変化検出数: {len(watcher._changes)}")


if __name__ == "__main__":
    main()
