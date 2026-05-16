"""
Video2AI MCP Server
===================
Claude Code から直接 PC 操作と動画解析を行うための MCP サーバー。

ツール一覧:
  screenshot        - スクリーンショット撮影
  screenshot_region - 指定領域のスクリーンショット
  get_mouse_position - マウス座標取得
  move_to           - マウスカーソル移動（クリックなし、位置確認用）
  peek              - カーソル周辺を拡大キャプチャ（クリック前の目視確認）
  click             - マウスクリック
  right_click       - 右クリック
  double_click      - ダブルクリック
  drag              - ドラッグ操作
  scroll            - スクロール
  type_text         - テキスト入力
  keypress          - キー操作
  focus_window      - ウィンドウ検索・フォーカス
  undo              - 操作の取消（Ctrl+Z送信）
  action_log        - 操作ログ表示
  list_windows      - ウィンドウ一覧
  memory_store      - 作業記憶に保存
  memory_recall     - 作業記憶を取得
  memory_clear      - 作業記憶をクリア
  memory_learn              - 長期記憶に短文の知見を保存（永続化、次セッションで利用可能）
  memory_knowledge          - 長期記憶から過去の経験・知見を FTS5 で検索
  memory_segment_learn      - 動画/画面/因果を含む Semantic Segment を記憶 (schema v3)
  memory_segment_knowledge  - FTS5 + vector + RRF のハイブリッド検索 (schema v3)
  canvas_create     - 脳内キャンバス作成
  canvas_draw       - キャンバスに描画
  canvas_paste      - キャンバス間の貼り付け
  canvas_view       - キャンバスを画像として取得
  canvas_snapshot   - 状態の保存/復元
  canvas_compare    - キャンバス比較
  canvas_list       - キャンバス一覧
  canvas_delete     - キャンバス削除
  dialectic_start   - 弁証法セッション開始
  dialectic_challenge - アンチテーゼ設定
  dialectic_synthesize - ジンテーゼ導出
  dialectic_iterate - 次ラウンドへ
  dialectic_conclude - 結論
  dialectic_view    - セッション全体表示
  dialectic_list    - セッション一覧
  save_file         - 保存ダイアログ経由のファイル保存（Alt+N → パス入力 → Enter）
  verify_text       - テキスト誤読検証（AI読取 vs OCR の機械的照合）
  workflow_record   - ワークフロー記録開始
  workflow_stop     - ワークフロー記録停止・保存
  workflow_run      - ワークフロー再生
  workflow_list     - ワークフロー一覧
  workflow_delete   - ワークフロー削除
  zoom_and_refine   - 再帰的サブグリッド分割で正確な座標を特定（自動深度調整対応）
  watcher_start     - ScreenWatcher 開始（ウィンドウの UI 要素・視覚変化を常時監視）
  watcher_stop      - ScreenWatcher 停止
  watcher_report    - ScreenWatcher レポート（現在の UI 状態 + 直近の変化）
  watcher_find      - UI 要素を名前・型で検索（座標付き）
  watcher_list_elements - 全 UI 要素一覧
  watcher_get_value - UI 要素の値を取得
  watcher_wait_value - 要素の値が期待値になるまで待機
  watcher_wait_change - 何らかの変化が起きるまで待機
  watcher_color     - 指定座標の色情報を取得
  watcher_watch_region - OpenCV 視覚変化の監視領域を追加
  watcher_unwatch_region - 監視領域を削除
  smart_find        - 階層的UI要素検索（テンプレート→UI Automation→フォールバック）
  template_save     - スクリーンショットからテンプレート画像を保存
  template_list     - テンプレート一覧
  template_delete   - テンプレート削除
  template_learn    - LLM視覚で画面を段階的にズームしてテンプレートを自動取得・保存
  ff_click          - フォーカスフリーでUI要素をクリック（カーソル・フォーカス不変）
  ff_type           - フォーカスフリーでUI要素にテキスト設定
  ff_list           - フォーカスフリーで操作可能なUI要素一覧
  ff_info           - UI要素の詳細情報（対応パターン、座標、HWND等）
  analyze_video     - Video2AI 動画解析

登録方法:
  claude mcp add video2ai-desktop -- python "<path>/mcp_server.py"
"""
import sys
import os
import logging
import subprocess
import functools
import asyncio
import threading
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

# MCP stdio transport は stdout を使うため、ログは stderr に出す
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

# Problem B fix (Option 3, 補助): HuggingFace Hub / Transformers の HTTP
# revision check と tqdm progress bar を抑止する。subprocess + asyncio +
# piped stdio の組合せで SentenceTransformer 初回ロードがハングする問題
# の amplifier 側 (Hub HTTP call / tqdm stdout write) を無効化する。
# MCP subprocess では stdout が JSON-RPC 専用、stdin/stderr も pipe のため、
# tqdm/requests が tty 相当の I/O を試みると async event loop と競合する。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ---------------------------------------------------------------------------
# Per-Monitor V2 DPI Awareness 設定（最優先で実行）
# ---------------------------------------------------------------------------
# マルチモニター環境で正確なマウス座標を扱うため、プロセスを DPI-aware に設定する。
# これにより mss / pyautogui / Win32 API が全て物理ピクセル座標で統一される。
# DPI-unaware のままだと、Windows が座標を仮想化し、スケーリング≠100% で座標がズレる。
if sys.platform == "win32":
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE_V2 = 2
        # 失敗しても致命的ではない（DPI-unaware として動作するだけ）
        hr = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        if hr == 0:  # S_OK
            _DPI_AWARE = True
            logging.info("DPI Awareness: Per-Monitor V2 設定成功")
        elif hr == -2147024891:  # E_ACCESSDENIED — 既に設定済み
            _DPI_AWARE = True
            logging.info("DPI Awareness: 既に設定済み")
        else:
            _DPI_AWARE = False
            logging.warning(f"DPI Awareness 設定失敗: HRESULT={hr:#010x}")
    except Exception as e:
        _DPI_AWARE = False
        logging.warning(f"DPI Awareness 設定失敗（DPI-unaware で動作）: {e}")
else:
    _DPI_AWARE = True  # Linux/macOS は DPI 仮想化なし

# プロジェクトルートを sys.path に追加（agent モジュールの import 用）
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mcp.server.fastmcp import FastMCP, Image
from agent.screen_capture import ScreenCapture
from agent.action_executor import ActionExecutor
from agent.working_memory import WorkingMemory, MemoryChunk
from agent.long_term_memory import LongTermMemory
from agent.mental_image import MentalImage
from agent.mental_canvas import CanvasManager
from agent.dialectic import DialecticalReasoner
from agent.ocr_windows import ocr_image, ocr_region
from agent.blacklist import OperationBlacklist
from agent.workflow import WorkflowManager
from agent.screen_watcher import ScreenWatcher
from agent.template_matcher import TemplateMatcher
from agent.focusfree import FocusFreeOperator
# Bundle 2.15: shared Phase 2 primitive co-located with this module so the
# MCP tool here and benchmark/run_benchmark.py cannot diverge again.
from pinpoint_core import hex_stencil_dots, clip_dots_to_bounds

mcp = FastMCP("video2ai-desktop")

# ---------------------------------------------------------------------------
# スクリーンショット座標 → 物理座標 変換
# ---------------------------------------------------------------------------
# Per-Monitor V2 DPI-aware プロセスでは mss / pyautogui ともに物理ピクセル座標で動作する。
# そのため DPI スケール除算は不要。変換はリサイズ補正のみ。
_last_screenshot_info: dict[int, tuple[int, int]] = {}  # monitor → (width, height)
# ウィンドウキャプチャ時の情報: monitor → (win_left, win_top, win_width, win_height)
_last_window_capture: dict[int, tuple[int, int, int, int]] = {}

# 座標検証ガードレール: 座標特定ツールが算出した座標のみを「検証済み」として記録
# 対象: pinpoint / smart_find / watcher_find / grid_find / som_find（アルゴリズムで座標を特定）
# 非対象: peek / move_to（AI の目測に依存するため座標源にならない）
import time as _time_module
_verified_coordinates: list[dict] = []  # [{"x": int, "y": int, "time": float, "tool": str}]
_COORD_VERIFY_TIMEOUT = 120  # 秒: この時間内の検証のみ有効
_COORD_VERIFY_RADIUS = 100  # px: この距離以内なら「検証済み」とみなす


def _record_verified_coord(phys_x: int, phys_y: int, tool: str):
    """座標特定ツールが算出した座標を記録する。
    peek / move_to は AI の目測に依存するため呼んではならない。
    """
    now = _time_module.time()
    # 古いエントリを削除
    _verified_coordinates[:] = [
        v for v in _verified_coordinates
        if now - v["time"] < _COORD_VERIFY_TIMEOUT
    ]
    _verified_coordinates.append({"x": phys_x, "y": phys_y, "time": now, "tool": tool})


def _is_coord_verified(phys_x: int, phys_y: int) -> str | None:
    """座標が最近検証されたか確認する。検証済みなら None、未検証なら警告メッセージ。"""
    now = _time_module.time()
    for v in reversed(_verified_coordinates):
        if now - v["time"] > _COORD_VERIFY_TIMEOUT:
            continue
        dist = ((v["x"] - phys_x) ** 2 + (v["y"] - phys_y) ** 2) ** 0.5
        if dist <= _COORD_VERIFY_RADIUS:
            return None  # 検証済み
    return (
        "⚠ 座標未検証: この座標は座標特定ツール（pinpoint / smart_find / watcher_find / "
        "grid_find / som_find）で特定されていません。AI の目測（スクリーンショットから座標を読む）"
        "はハルシネーションするため信頼できません。座標特定ツールを使うか、force=True で強制実行してください"
    )


@contextmanager
def _preserve_screenshot_state(*monitors: int):
    """内部ツールが撮るスクリーンショットからユーザーの座標変換状態を保護する。

    内部ツール（som_find, pinpoint, grid_find 等）は独自にスクリーンショットを
    撮り _last_screenshot_info を上書きする。これにより、ユーザーが直前に撮った
    screenshot() の座標空間が狂い、後続の click() で座標がずれるバグが発生する。

    このコンテキストマネージャはブロック突入時に状態を退避し、
    ブロック終了時に復元する。内部ツールはブロック内で自由に状態を
    上書きでき、自身の座標計算に使える。

    Usage:
        with _preserve_screenshot_state(monitor):
            # 内部スクリーンショット＆座標計算
            ...
        # ここで元の状態が復元される
    """
    saved_info = {}
    saved_window = {}
    for m in monitors:
        if m in _last_screenshot_info:
            saved_info[m] = _last_screenshot_info[m]
        if m in _last_window_capture:
            saved_window[m] = _last_window_capture[m]
    try:
        yield
    finally:
        for m in monitors:
            if m in saved_info:
                _last_screenshot_info[m] = saved_info[m]
            elif m in _last_screenshot_info:
                del _last_screenshot_info[m]
            if m in saved_window:
                _last_window_capture[m] = saved_window[m]
            elif m in _last_window_capture:
                del _last_window_capture[m]


def _preserves_screenshot_state(func):
    """内部ツール用デコレータ: ユーザーの座標変換状態を保護する。

    デコレートされた関数の ``monitor`` パラメータを読み取り、
    実行中の内部スクリーンショットがユーザーの _last_screenshot_info /
    _last_window_capture を汚染しないよう保存・復元する。

    monitor=0 (未指定・継続モード) の場合は保護をスキップする。
    """
    import inspect as _inspect

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = _inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        monitor = bound.arguments.get("monitor", 0)
        if monitor > 0:
            with _preserve_screenshot_state(monitor):
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper


def _get_monitor_dpi_scale(monitor_index: int) -> float:
    """モニターの DPI スケーリング倍率を取得する（情報表示・診断用）。

    DPI-aware プロセスでは座標変換に使わない。list_monitors の表示用。
    """
    if sys.platform != "win32" or monitor_index <= 0:
        return 1.0
    try:
        import ctypes
        from ctypes import wintypes

        # MonitorFromPoint でモニターハンドルを取得
        import mss
        with mss.mss() as sct:
            if monitor_index >= len(sct.monitors):
                return 1.0
            mon = sct.monitors[monitor_index]
            # モニター中央の座標を使って MonitorFromPoint
            cx = mon['left'] + mon['width'] // 2
            cy = mon['top'] + mon['height'] // 2

        MONITOR_DEFAULTTONEAREST = 2
        hmon = ctypes.windll.user32.MonitorFromPoint(
            ctypes.wintypes.POINT(cx, cy), MONITOR_DEFAULTTONEAREST
        )
        # GetDpiForMonitor (shcore.dll)
        dpi_x = ctypes.c_uint()
        dpi_y = ctypes.c_uint()
        MDT_EFFECTIVE_DPI = 0
        hr = ctypes.windll.shcore.GetDpiForMonitor(
            hmon, MDT_EFFECTIVE_DPI,
            ctypes.byref(dpi_x), ctypes.byref(dpi_y)
        )
        if hr == 0:  # S_OK
            return round(dpi_x.value / 96.0 * 4) / 4  # 96 DPI = 100%
    except Exception:
        pass
    return 1.0


def _screen_to_physical(monitor_index: int, sx: int, sy: int) -> tuple[int, int]:
    """スクリーンショット画像座標を物理スクリーン座標に変換する。

    DPI-aware プロセスでは mss が物理ピクセル座標を返すため、
    変換はリサイズ補正（screenshot の width パラメータ分）のみ。

    ウィンドウキャプチャモード（_last_window_capture にエントリがある場合）:
      ウィンドウの物理座標原点を基準に変換する。

    monitor_index: モニター番号（1, 2, 3...）
    sx, sy: スクリーンショット画像内の座標
        ※ 直前の screenshot(monitor=N, width=W) で撮った画像上の座標を想定。

    Returns: (physical_x, physical_y) — click/drag/move_to に渡せる絶対座標
    """
    import mss

    # ウィンドウキャプチャモード: ウィンドウの物理座標を基準に変換
    if monitor_index in _last_window_capture:
        win_left, win_top, win_w, win_h = _last_window_capture[monitor_index]
        if monitor_index in _last_screenshot_info:
            shot_w, shot_h = _last_screenshot_info[monitor_index]
        else:
            shot_w, shot_h = win_w, win_h
        px = win_left + int(sx * win_w / shot_w)
        py = win_top + int(sy * win_h / shot_h)
        logging.debug(
            f"座標変換(window): monitor={monitor_index} "
            f"screenshot({sx},{sy}) / shot({shot_w}x{shot_h}) "
            f"→ physical({px},{py}) [win_origin=({win_left},{win_top}), win={win_w}x{win_h}]"
        )
        return (px, py)

    with mss.mss() as sct:
        if monitor_index <= 0 or monitor_index >= len(sct.monitors):
            return (sx, sy)
        mon = sct.monitors[monitor_index]
    origin_x, origin_y = mon['left'], mon['top']
    mon_w, mon_h = mon['width'], mon['height']

    # 直前の screenshot のサイズを考慮（リサイズ補正）
    if monitor_index in _last_screenshot_info:
        shot_w, shot_h = _last_screenshot_info[monitor_index]
    else:
        shot_w, shot_h = mon_w, mon_h

    # 変換: screenshot座標 → 物理座標
    # sx/shot_w = 画面上の相対位置 → mon_w に展開してオフセット加算
    px = origin_x + int(sx * mon_w / shot_w)
    py = origin_y + int(sy * mon_h / shot_h)

    logging.debug(
        f"座標変換: monitor={monitor_index} "
        f"screenshot({sx},{sy}) / shot({shot_w}x{shot_h}) "
        f"→ physical({px},{py}) [origin=({origin_x},{origin_y}), mon={mon_w}x{mon_h}]"
    )
    return (px, py)


def _physical_to_screen(monitor_index: int, px: int, py: int) -> tuple[int, int]:
    """物理スクリーン座標をスクリーンショット画像座標に逆変換する。

    _screen_to_physical の逆関数。watcher_find 等で得た物理座標を
    zoom_and_refine のスクリーンショット座標に変換するために使う。
    """
    import mss
    with mss.mss() as sct:
        if monitor_index <= 0 or monitor_index >= len(sct.monitors):
            return (px, py)
        mon = sct.monitors[monitor_index]
    origin_x, origin_y = mon['left'], mon['top']
    mon_w, mon_h = mon['width'], mon['height']

    if monitor_index in _last_screenshot_info:
        shot_w, shot_h = _last_screenshot_info[monitor_index]
    else:
        shot_w, shot_h = mon_w, mon_h

    sx = int((px - origin_x) * shot_w / mon_w)
    sy = int((py - origin_y) * shot_h / mon_h)
    return (sx, sy)


def _build_screenshot_meta(
    monitor_index: int,
    shot_w: int,
    shot_h: int,
    window_capture: tuple[int, int, int, int] | None = None,
) -> dict:
    """ScreenshotMeta を構築する。Neo ノードの座標変換に必要な全情報を含む。

    Args:
        monitor_index: モニター番号
        shot_w, shot_h: スクリーンショット画像のサイズ
        window_capture: ウィンドウキャプチャの場合 (left, top, w, h)
    """
    import mss

    meta: dict = {
        "monitor": monitor_index,
        "screenshot_width": shot_w,
        "screenshot_height": shot_h,
        "origin_x": 0,
        "origin_y": 0,
        "physical_width": shot_w,
        "physical_height": shot_h,
        "dpi_scale": 1.0,
    }

    if window_capture:
        wl, wt, ww, wh = window_capture
        meta["origin_x"] = wl
        meta["origin_y"] = wt
        meta["physical_width"] = ww
        meta["physical_height"] = wh
        meta["dpi_scale"] = ww / shot_w if shot_w > 0 else 1.0
        meta["window_capture"] = True
    elif monitor_index > 0:
        try:
            with mss.mss() as sct:
                if monitor_index < len(sct.monitors):
                    mon = sct.monitors[monitor_index]
                    meta["origin_x"] = mon["left"]
                    meta["origin_y"] = mon["top"]
                    meta["physical_width"] = mon["width"]
                    meta["physical_height"] = mon["height"]
                    meta["dpi_scale"] = mon["width"] / shot_w if shot_w > 0 else 1.0
        except Exception:
            pass

    return meta


# ---------------------------------------------------------------------------
# カーソル自動復帰（ユーザー干渉最小化）
# ---------------------------------------------------------------------------
# AI がクリック/ドラッグ後にカーソルをユーザーの元の位置に戻す。
# これによりユーザーのカーソルが別モニターに飛んだまま行方不明にならない。

def _save_cursor() -> tuple[int, int]:
    """現在のカーソル位置を取得"""
    try:
        import ctypes
        import ctypes.wintypes
        point = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
        return (point.x, point.y)
    except Exception:
        return (0, 0)


def _restore_cursor(saved_x: int, saved_y: int) -> None:
    """カーソルを保存した位置に戻す"""
    try:
        import ctypes
        ctypes.windll.user32.SetCursorPos(saved_x, saved_y)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# カーソル位置検証（座標ズレ診断）
# ---------------------------------------------------------------------------
_CURSOR_DRIFT_THRESHOLD = 3  # px 以内なら正常


def _verify_cursor_position(expected_x: int, expected_y: int) -> str:
    """カーソルが期待位置にあるか検証する。ズレがあれば警告メッセージを返す。"""
    try:
        import pyautogui
        actual_x, actual_y = pyautogui.position()
        dx = abs(actual_x - expected_x)
        dy = abs(actual_y - expected_y)
        if dx <= _CURSOR_DRIFT_THRESHOLD and dy <= _CURSOR_DRIFT_THRESHOLD:
            return ""  # 正常
        msg = (f"⚠ カーソル位置ズレ検出: 期待({expected_x},{expected_y}) "
               f"→ 実際({actual_x},{actual_y}) 差分(dx={dx}, dy={dy})")
        logging.warning(msg)
        return msg
    except Exception:
        return ""


# 脳内画像（クリック位置検証用）
_mental_image = MentalImage()

# 直前にフォーカスしたウィンドウの HWND（type_text/keypress 前にフォーカス復帰用）
_last_target_hwnd: int = 0
_last_target_title: str = ""  # HWND リユース検出用


def _set_target_hwnd(hwnd: int) -> None:
    """フォーカス復帰用の HWND とタイトルを保存する。"""
    global _last_target_hwnd, _last_target_title
    _last_target_hwnd = hwnd
    if hwnd and sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                _last_target_title = buf.value
            else:
                _last_target_title = ""
        except Exception:
            _last_target_title = ""
    else:
        _last_target_title = ""


def _validate_target_hwnd() -> bool:
    """保存された HWND がまだ有効か（同じウィンドウか）を検証する。

    IsWindow だけでは HWND リユース（OS が閉じたウィンドウの ID を
    別のウィンドウに再割り当て）を検出できない。タイトルの部分一致で検証する。
    """
    global _last_target_hwnd, _last_target_title
    if not _last_target_hwnd or sys.platform != "win32":
        return False
    try:
        import ctypes
        user32 = ctypes.windll.user32
        if not user32.IsWindow(_last_target_hwnd):
            _last_target_hwnd = 0
            _last_target_title = ""
            return False
        # タイトル一致チェック（HWND リユース検出）
        if _last_target_title:
            length = user32.GetWindowTextLengthW(_last_target_hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(_last_target_hwnd, buf, length + 1)
                current_title = buf.value
                # タイトルが完全に無関係なら HWND リユースとみなす
                # （タイトル変更に対応するため、保存タイトルの主要部分が含まれるか確認）
                saved_words = set(_last_target_title.split()) - {"", "-", "—", "|"}
                current_words = set(current_title.split()) - {"", "-", "—", "|"}
                if saved_words and current_words and not saved_words & current_words:
                    logging.warning(
                        f"HWND リユース検出: HWND={_last_target_hwnd} "
                        f"保存タイトル='{_last_target_title}' → 現在='{current_title}'"
                    )
                    _last_target_hwnd = 0
                    _last_target_title = ""
                    return False
        return True
    except Exception:
        return False


def _restore_focus() -> bool:
    """直前にフォーカスしたウィンドウにフォーカスを復帰する。
    AttachThreadInput でフォアグラウンド制限を回避する（Alt+Escape 不使用）。
    Returns: フォーカス復帰に成功したか
    """
    global _last_target_hwnd, _last_target_title
    if not _validate_target_hwnd():
        return False
    try:
        import ctypes
        import time as _t
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        # 既にフォーカスがあるなら何もしない
        # ダイアログ等の子ウィンドウ（owned window）がフォアグラウンドの場合も同一とみなす
        fg = user32.GetForegroundWindow()
        if fg == _last_target_hwnd:
            return True
        if fg:
            owner = user32.GetWindow(fg, 4)  # GW_OWNER = 4
            if owner == _last_target_hwnd:
                return True

        for attempt in range(3):
            # AttachThreadInput でフォアグラウンド制限を回避（Alt キー不使用）
            fg = user32.GetForegroundWindow()
            fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
            my_tid = kernel32.GetCurrentThreadId()
            attached = False
            if fg_tid and fg_tid != my_tid:
                attached = bool(user32.AttachThreadInput(my_tid, fg_tid, True))
            user32.ShowWindow(_last_target_hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(_last_target_hwnd)
            if attached:
                user32.AttachThreadInput(my_tid, fg_tid, False)
            _t.sleep(0.05)
            # 検証: 実際にフォーカスが取れたか
            fg = user32.GetForegroundWindow()
            if fg == _last_target_hwnd:
                return True
            # owned window がフォアグラウンドでも OK
            if fg:
                owner = user32.GetWindow(fg, 4)
                if owner == _last_target_hwnd:
                    return True
            # 失敗時: 少し待ってリトライ
            _t.sleep(0.1)
        # 3回試して失敗
        logging.warning(f"フォーカス復帰失敗: HWND={_last_target_hwnd} (現在のFG={fg})")
        return False
    except Exception:
        return False


# ActionExecutor（脳内画像を注入）
# MCP モードは対話的なので高速設定（Web UI モードより速く）
_executor = ActionExecutor(
    dry_run=False,
    mental_image=_mental_image,
    move_duration=0.05,   # マウス移動: 0.3s → 0.05s
    click_delay=0.05,     # クリック後: 0.1s → 0.05s
)
# Ctrl+V 直前のフォーカス復帰を注入（MCP モードではペースト前にフォーカスが戻る問題対策）
_executor._pre_paste_focus_fn = _restore_focus

# 操作ブラックリスト（禁止リスト）
_blacklist = OperationBlacklist(project_root=_ROOT)

# 作業記憶（MCP サーバープロセス内でステートを保持）
_wm_capacity = int(os.environ.get("V2AI_WM_CAPACITY", "7"))
_working_memory = WorkingMemory(capacity=_wm_capacity)

# メンタルキャンバス（脳内画像）マネージャ
_canvas_mgr = CanvasManager()

# 長期記憶（操作経験の永続化）— schema v3 で vec_memories も管理
_long_term_memory = LongTermMemory()

# 弁証法的推論エンジン — 長期記憶と同じ DB ファイルに永続化する
# (dialectic_sessions テーブル、CREATE IF NOT EXISTS)
_dialectic = DialecticalReasoner(db_path=_long_term_memory._db_path)


# Embedding model を **eager sync load** しておく (Problem B fix, Option 2 改)。
#
# 設計経緯: 当初は background thread で warmup していたが、Windows Python 3.14
# + FastMCP の asyncio (Proactor) event loop + piped stdio 環境下で
# SentenceTransformer 初回ロードが 180s+ hang する現象を確認 (stderr 診断で
# 切り分け済、standalone 実行では 8s 完了)。原因は thread + HF filelock +
# ProactorEventLoop の競合が有力。
#
# 回避策: mcp.run() が asyncio loop に入る **前に** main thread で同期
# ロードし、以降は `_get_embedding_model()` の singleton キャッシュを
# 返すだけにする。起動時間が +8s 増えるが、確実に動作する。
try:
    from agent.long_term_memory import _get_embedding_model as _eager_load
    _eager_load()
    logging.warning("embedding model preloaded (sync)")
except Exception as exc:  # noqa: BLE001 — preload は best-effort
    logging.warning("embedding model preload failed: %s", exc)

# ---------------------------------------------------------------------------
# 操作ログ（MCP モード用 JSONL ログ）
# ---------------------------------------------------------------------------
import json
import time as _time
from collections import deque

_LOG_DIR = Path.home() / "video2ai_agent_sessions" / "mcp"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_path = _LOG_DIR / f"mcp_{_time.strftime('%Y%m%d_%H%M%S')}.jsonl"

# アクション履歴（Undo 用）
_action_history: deque[dict] = deque(maxlen=50)

# ワークフローマネージャー
_workflow_mgr = WorkflowManager()

# ScreenWatcher（UI Automation + OpenCV 統合画面監視）
_screen_watcher: ScreenWatcher | None = None

# テンプレートマッチャー（汎用UI要素検索）
_template_matcher = TemplateMatcher()

# フォーカスフリー操作（カーソル・フォーカスを動かさずに操作）
_focusfree = FocusFreeOperator()


def _log_action(action_type: str, target: str = "", value: str = "",
                success: bool = True, error: str = "",
                workflow_args: dict = None) -> None:
    """操作をJSONLログに記録し、アクション履歴に追加する。
    workflow_args: ワークフロー再生用の引数辞書（指定時のみ記録）。
    """
    entry = {
        "timestamp": _time.time(),
        "time": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action_type,
        "target": target,
        "value": value[:100] if value else "",
        "success": success,
        "error": error,
    }
    # JSONL に追記
    try:
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # Undo 用履歴に追加（成功した操作のみ）
    if success:
        _action_history.append(entry)
    # ワークフロー記録中なら操作をキャプチャ（成功した操作のみ）
    if success and workflow_args is not None and _workflow_mgr.is_recording:
        _workflow_mgr.record_step(action_type, workflow_args)


# ---------------------------------------------------------------------------
# PC 操作ツール
# ---------------------------------------------------------------------------

@mcp.tool()
def list_monitors() -> str:
    """接続されている全モニターの詳細情報を返す。
    DPIスケーリング、物理/論理解像度、座標原点を含む。
    マルチモニター環境での座標ズレ診断に有用。"""
    try:
        import mss
        with mss.mss() as sct:
            lines = []
            lines.append(f"DPI Awareness: {'Per-Monitor V2 (物理座標)' if _DPI_AWARE else 'DPI-unaware (論理座標)'}")
            lines.append(f"モニター: {len(sct.monitors)-1}台")
            lines.append("")
            for i, mon in enumerate(sct.monitors):
                if i == 0:
                    lines.append(f"  仮想デスクトップ全体: {mon['width']}x{mon['height']} 原点({mon['left']},{mon['top']})")
                else:
                    dpi_scale = _get_monitor_dpi_scale(i)
                    dpi_pct = int(dpi_scale * 100)
                    # DPI-aware の場合、mss の値が物理解像度
                    phys_w, phys_h = mon['width'], mon['height']
                    # 論理解像度 = 物理 / スケール
                    log_w = int(phys_w / dpi_scale) if dpi_scale > 0 else phys_w
                    log_h = int(phys_h / dpi_scale) if dpi_scale > 0 else phys_h
                    lines.append(f"  Monitor {i}:")
                    lines.append(f"    物理解像度: {phys_w}x{phys_h}")
                    if dpi_scale != 1.0:
                        lines.append(f"    論理解像度: {log_w}x{log_h}")
                    lines.append(f"    DPI スケーリング: {dpi_pct}% (x{dpi_scale})")
                    lines.append(f"    座標原点: ({mon['left']}, {mon['top']})")
                    lines.append(f"    座標範囲: X [{mon['left']}..{mon['left']+phys_w-1}], Y [{mon['top']}..{mon['top']+phys_h-1}]")
                    # スクリーンショット情報（直近のキャプチャサイズ）
                    if i in _last_screenshot_info:
                        sw, sh = _last_screenshot_info[i]
                        lines.append(f"    直近スクリーンショット: {sw}x{sh} (リサイズ率: {sw/phys_w:.2f})")
            return "\n".join(lines)
    except Exception as e:
        return f"エラー: {e}"


@mcp.tool()
def screenshot(monitor: int = 0, width: int = 1920, window: str = "",
               privacy_blur_faces: bool = False,
               privacy_redact_text: bool = False) -> list:
    """スクリーンショットを撮影して返す。

    画像と座標変換メタデータ（ScreenshotMeta）を返す。
    メタデータには screenshot_width/height, physical_width/height,
    origin_x/y, monitor, dpi_scale が含まれ、Neo ノードの
    ScreenToPhysical で座標変換に使用される。

    Args:
        monitor: モニター番号 (0=全画面, 1=プライマリ, 2,3...=その他)
        width: リサイズ幅 (0=オリジナル解像度)
        window: ウィンドウタイトルのキーワード。指定するとそのウィンドウだけをキャプチャする。
                ウィンドウが存在するモニターが自動検出され、座標変換もウィンドウ基準になる。
                click/move_to 等で monitor にそのモニター番号を指定すれば正しく変換される。
        privacy_blur_faces: 顔を検出してぼかす（Neoのプライバシーポリシーノードから指定）
        privacy_redact_text: 個人情報テキスト（クレカ番号、電話番号等）を墨消しする
    """
    if window:
        return _screenshot_window(window, width,
                                   privacy_blur_faces, privacy_redact_text)
    cap = ScreenCapture(monitor_index=monitor, resize_width=width)
    pil_img = cap.capture_as_pil()
    # スクリーンショットのサイズを記録（座標変換用）
    if monitor > 0:
        _last_screenshot_info[monitor] = (pil_img.width, pil_img.height)
        # モニター全体キャプチャの場合、ウィンドウキャプチャ情報をクリア
        _last_window_capture.pop(monitor, None)

    # プライバシーフィルタ適用（フィルタ後の画像のみ返却）
    if privacy_blur_faces or privacy_redact_text:
        from agent.privacy_filters import apply_privacy_filters
        pil_img = apply_privacy_filters(
            pil_img,
            blur_faces_enabled=privacy_blur_faces,
            redact_text_enabled=privacy_redact_text,
        )

    buf = BytesIO()
    pil_img.save(buf, format="PNG")

    # ScreenshotMeta を構築
    meta = _build_screenshot_meta(monitor, pil_img.width, pil_img.height)
    import json as _json
    return [Image(data=buf.getvalue(), format="png"),
            f"SCREENSHOT_META:{_json.dumps(meta)}"]


def _screenshot_window(keyword: str, width: int = 1920,
                       privacy_blur_faces: bool = False,
                       privacy_redact_text: bool = False) -> list:
    """指定ウィンドウだけをキャプチャする内部関数。"""
    import ctypes
    import ctypes.wintypes
    import mss

    user32 = ctypes.windll.user32

    # ウィンドウ検索
    target_hwnd = None
    target_title = ""

    def enum_callback(hwnd, _):
        nonlocal target_hwnd, target_title
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if keyword in buf.value:
                target_hwnd = hwnd
                target_title = buf.value
                return False  # 最初のマッチで停止
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )
    user32.EnumWindows(WNDENUMPROC(enum_callback), 0)

    if not target_hwnd:
        # ウィンドウ見つからない場合はモニター全体にフォールバック
        cap = ScreenCapture(monitor_index=0, resize_width=width)
        pil_img = cap.capture_as_pil()
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        meta = _build_screenshot_meta(0, pil_img.width, pil_img.height)
        import json as _json
        return [Image(data=buf.getvalue(), format="png"),
                f"SCREENSHOT_META:{_json.dumps(meta)}"]

    # ウィンドウの物理座標を取得
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(target_hwnd, ctypes.byref(rect))
    win_left = rect.left
    win_top = rect.top
    win_w = rect.right - rect.left
    win_h = rect.bottom - rect.top

    # 最大化ウィンドウの -8px オフセットを補正
    if win_left < 0:
        win_w += win_left  # 幅を縮める
        win_left = 0
    if win_top < 0:
        win_h += win_top
        win_top = 0

    # ウィンドウが属するモニターを検出
    with mss.mss() as sct:
        target_monitor = 0
        for i in range(1, len(sct.monitors)):
            mon = sct.monitors[i]
            # ウィンドウの中心がモニター内にあるか
            cx = win_left + win_w // 2
            cy = win_top + win_h // 2
            if (mon['left'] <= cx < mon['left'] + mon['width'] and
                    mon['top'] <= cy < mon['top'] + mon['height']):
                target_monitor = i
                break

    # ウィンドウ領域をキャプチャ
    cap = ScreenCapture(region=(win_left, win_top, win_w, win_h), resize_width=width)
    pil_img = cap.capture_as_pil()

    # 座標変換用の情報を記録
    if target_monitor > 0:
        _last_screenshot_info[target_monitor] = (pil_img.width, pil_img.height)
        _last_window_capture[target_monitor] = (win_left, win_top, win_w, win_h)

    logging.info(
        f"ウィンドウキャプチャ: '{target_title}' "
        f"rect=({win_left},{win_top},{win_w}x{win_h}) "
        f"monitor={target_monitor} shot={pil_img.width}x{pil_img.height}"
    )

    # プライバシーフィルタ適用
    if privacy_blur_faces or privacy_redact_text:
        from agent.privacy_filters import apply_privacy_filters
        pil_img = apply_privacy_filters(
            pil_img,
            blur_faces_enabled=privacy_blur_faces,
            redact_text_enabled=privacy_redact_text,
        )

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    meta = _build_screenshot_meta(
        target_monitor, pil_img.width, pil_img.height,
        window_capture=(win_left, win_top, win_w, win_h))
    import json as _json
    return [Image(data=buf.getvalue(), format="png"),
            f"SCREENSHOT_META:{_json.dumps(meta)}"]


@mcp.tool()
def screenshot_region(left: int, top: int, width: int, height: int,
                      resize_width: int = 0) -> Image:
    """画面の指定領域のみスクリーンショットを撮影する。ボタンやダイアログの詳細確認に便利。

    Args:
        left: 領域の左端 X 座標
        top: 領域の上端 Y 座標
        width: 領域の幅 (px)
        height: 領域の高さ (px)
        resize_width: リサイズ幅 (0=オリジナル解像度)
    """
    cap = ScreenCapture(region=(left, top, width, height), resize_width=resize_width)
    pil_img = cap.capture_as_pil()
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def load_image(path: str, output_width: int = 0,
               privacy_blur_faces: bool = False,
               privacy_redact_text: bool = False,
               privacy_strip_raw: bool = False) -> str:
    """ローカル画像ファイルを読み込んでbase64で返す。

    screenshot と同じ ActionResult 形式で返すため、Neo の ImageLoad ノードから
    Screenshot ノードと互換のパイプラインで利用できる。

    Args:
        path: 画像ファイルパス (jpg/png/bmp/webp/gif)
        output_width: リサイズ幅 (0=元サイズ)
        privacy_blur_faces: 顔を検出してぼかす
        privacy_redact_text: 個人情報テキストを墨消しする
        privacy_strip_raw: EXIFなどのメタデータを除去する
    """
    import json as _json
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        return _json.dumps({"success": False, "action": "load_image",
                            "error": f"File not found: {path}"})

    try:
        from PIL import Image as PILImage
        pil_img = PILImage.open(p).convert("RGB")
    except Exception as e:
        return _json.dumps({"success": False, "action": "load_image",
                            "error": f"Cannot open image: {e}"})

    # EXIF メタデータ除去
    if privacy_strip_raw:
        from PIL import Image as PILImage
        clean = PILImage.new(pil_img.mode, pil_img.size)
        clean.putdata(list(pil_img.getdata()))
        pil_img = clean

    # リサイズ
    if output_width > 0 and pil_img.width != output_width:
        scale = output_width / pil_img.width
        new_h = max(1, int(pil_img.height * scale))
        pil_img = pil_img.resize((output_width, new_h),
                                  PILImage.Resampling.BICUBIC)

    # プライバシーフィルタ
    if privacy_blur_faces or privacy_redact_text:
        from agent.privacy_filters import apply_privacy_filters
        pil_img = apply_privacy_filters(
            pil_img,
            blur_faces_enabled=privacy_blur_faces,
            redact_text_enabled=privacy_redact_text,
        )

    # base64 エンコード
    buf = BytesIO()
    fmt = p.suffix.lower().lstrip(".")
    if fmt in ("jpg", "jpeg"):
        save_fmt, mime_fmt = "JPEG", "jpeg"
    elif fmt == "webp":
        save_fmt, mime_fmt = "WEBP", "webp"
    else:
        save_fmt, mime_fmt = "PNG", "png"
    pil_img.save(buf, format=save_fmt)

    import base64 as _b64
    b64_data = _b64.b64encode(buf.getvalue()).decode("ascii")

    return _json.dumps({
        "success": True,
        "action": "load_image",
        "details": {
            "image": b64_data,
            "width": pil_img.width,
            "height": pil_img.height,
            "format": mime_fmt,
        }
    })


@mcp.tool()
def get_mouse_position() -> str:
    """現在のマウスカーソル位置を返す。どのモニター上にあるかも表示する。"""
    try:
        import pyautogui
        import mss
        x, y = pyautogui.position()
        # どのモニター上か判定
        mon_info = ""
        with mss.mss() as sct:
            for i in range(1, len(sct.monitors)):
                mon = sct.monitors[i]
                if (mon['left'] <= x < mon['left'] + mon['width'] and
                        mon['top'] <= y < mon['top'] + mon['height']):
                    local_x = x - mon['left']
                    local_y = y - mon['top']
                    mon_info = f" [Monitor {i} 内ローカル座標: ({local_x}, {local_y})]"
                    break
        return f"マウス位置: ({x}, {y}){mon_info}"
    except Exception as e:
        return f"取得失敗: {e}"


@mcp.tool()
def move_to(x: int, y: int, monitor: int = 0) -> str:
    """マウスカーソルを指定座標に移動する（クリックはしない）。
    クリック前の位置確認に使用。移動後に peek で周辺を拡大確認できる。

    Args:
        x: 移動先の X 座標（monitor 指定時はスクリーンショット内の座標）
        y: 移動先の Y 座標（monitor 指定時はスクリーンショット内の座標）
        monitor: 0=絶対座標, 1以上=そのモニターのスクリーンショット座標として自動変換
    """
    orig_x, orig_y = x, y
    if monitor > 0:
        x, y = _screen_to_physical(monitor, x, y)
    result = _executor.move_to(x, y)
    _log_action("move_to", f"({x},{y})", success=result.success, error=result.error or "",
                workflow_args={"x": orig_x, "y": orig_y, "monitor": monitor})
    if not result.success:
        return f"失敗: {result.error}"

    # カーソル位置の検証
    desc = result.description
    verify_msg = _verify_cursor_position(x, y)
    if verify_msg:
        desc += f"\n{verify_msg}"
    if monitor > 0:
        desc += f"\n座標変換: screenshot({orig_x},{orig_y}) → physical({x},{y}) [monitor={monitor}]"
    return desc


@mcp.tool()
def peek(x: int = 0, y: int = 0, size: int = 200) -> Image:
    """マウスカーソル周辺を拡大キャプチャして返す。クリック前の位置確認に最適。
    カーソルが正しい要素の上にあるかを目視確認してからクリックすることで、
    誤クリックを防止できる。

    座標を省略すると現在のカーソル位置を使用する。

    Args:
        x: 中心の X 座標（0 で現在のカーソル位置）
        y: 中心の Y 座標（0 で現在のカーソル位置）
        size: キャプチャ領域のサイズ（px、デフォルト200）
    """
    import pyautogui
    if x == 0 and y == 0:
        x, y = pyautogui.position()

    half = size // 2
    left = max(0, x - half)
    top = max(0, y - half)

    cap = ScreenCapture(region=(left, top, size, size))
    pil_img = cap.capture_as_pil()
    # 2x に拡大して見やすくする
    enlarged = pil_img.resize((size * 2, size * 2), resample=0)  # NEAREST for sharp pixels
    # 中央に十字線を描画（カーソル位置を示す）
    from PIL import ImageDraw
    draw = ImageDraw.Draw(enlarged)
    center = size  # 拡大後の中央
    draw.line([(center - 15, center), (center + 15, center)], fill="red", width=2)
    draw.line([(center, center - 15), (center, center + 15)], fill="red", width=2)



    buf = BytesIO()
    enlarged.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


# ---------------------------------------------------------------------------
# ActionResult JSON helper (Neo integration)
# ---------------------------------------------------------------------------

def _action_result(action: str, success: bool, details: dict | None = None,
                   error: str = "") -> str:
    """Return structured JSON for Neo-compatible ActionResult."""
    import json as _json
    result: dict = {"success": success, "action": action}
    if error:
        result["error"] = error
    if details:
        result["details"] = details
    return _json.dumps(result, ensure_ascii=False)


@mcp.tool()
def click(x: int, y: int, button: int | str = 1, verify: bool = False,
          monitor: int = 0, num_clicks: int = 1, force: bool = False) -> str:
    """指定したスクリーン座標をクリックする。

    Args:
        x: X 座標（monitor 指定時はスクリーンショット内の座標）
        y: Y 座標（monitor 指定時はスクリーンショット内の座標）
        button: マウスボタン (1=左, 2=中, 3=右) or ("left", "middle", "right")
        verify: True なら脳内画像で位置検証・画面変化確認し、ずれていれば自動補正する
        monitor: 0=絶対座標, 1以上=そのモニターのスクリーンショット座標として自動変換
        num_clicks: クリック回数 (デフォルト1)
        force: True なら座標検証ガードレールをスキップする
    """
    # 文字列 button を int に変換（Neo 互換）
    if isinstance(button, str):
        button = {"left": 1, "middle": 2, "right": 3}.get(button.lower(), 1)
    orig_x, orig_y = x, y
    saved_cursor = _save_cursor()
    if monitor > 0:
        x, y = _screen_to_physical(monitor, x, y)
    # 座標検証ガードレール: peek / pinpoint 等で事前確認されていない座標はブロック
    if not force:
        warn = _is_coord_verified(x, y)
        if warn:
            _restore_cursor(*saved_cursor)
            return _action_result("click", False,
                                  {"physical_x": x, "physical_y": y},
                                  error=warn)
    # verify 時: クリック前の画面をキャンバスに保存
    before_canvas = None
    if verify:
        try:
            before_canvas = _canvas_mgr.from_screenshot("_click_before")
            before_canvas.draw_marker(x, y, label=f"({x},{y})", color="red", size=20)
            before_canvas.snapshot("planned")
        except Exception:
            pass

    result = _executor.click(x, y, button, verify=verify)
    # num_clicks > 1 の場合は追加クリック
    for _ in range(num_clicks - 1):
        import time as _t
        _t.sleep(0.05)
        _executor.click(x, y, button, verify=False)
    # クリックしたウィンドウのHWNDを保存（後続のtype_text/keypressで復帰用）
    if result.success and sys.platform == "win32":
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            if hwnd:
                _set_target_hwnd(hwnd)
        except Exception:
            pass
    _log_action("click", f"({x},{y}) btn={button}", success=result.success, error=result.error or "",
                workflow_args={"x": orig_x, "y": orig_y, "button": button, "monitor": monitor})

    details: dict = {
        "physical_x": x, "physical_y": y,
        "button": button, "num_clicks": num_clicks,
        "message": result.description if result.success else result.error or "",
    }
    if monitor > 0:
        details["screenshot_x"] = orig_x
        details["screenshot_y"] = orig_y
        details["monitor"] = monitor

    if result.verification:
        v = result.verification
        details["verification"] = v.assessment

    # verify 時: クリック後の画面変化を比較
    if verify and before_canvas:
        try:
            after_canvas = _canvas_mgr.from_screenshot("_click_after")
            before_canvas.restore("planned")
            cmp = before_canvas.compare_with(after_canvas)
            details["screen_change"] = cmp.summary
            _canvas_mgr.delete("_click_before")
            _canvas_mgr.delete("_click_after")
        except Exception:
            pass

    # カーソル位置の検証
    verify_msg = _verify_cursor_position(x, y)
    if verify_msg:
        details["cursor_warning"] = verify_msg
    _restore_cursor(*saved_cursor)
    return _action_result("click", result.success, details,
                          error=result.error or "")


@mcp.tool()
def right_click(x: int, y: int, monitor: int = 0, force: bool = False) -> str:
    """指定したスクリーン座標を右クリックする。コンテキストメニューの表示に使用。

    Args:
        x: X 座標（monitor 指定時はスクリーンショット内の座標）
        y: Y 座標（monitor 指定時はスクリーンショット内の座標）
        monitor: 0=絶対座標, 1以上=そのモニターのスクリーンショット座標として自動変換
        force: True なら座標検証ガードレールをスキップする
    """
    orig_x, orig_y = x, y
    saved_cursor = _save_cursor()
    if monitor > 0:
        x, y = _screen_to_physical(monitor, x, y)
    if not force:
        warn = _is_coord_verified(x, y)
        if warn:
            _restore_cursor(*saved_cursor)
            return _action_result("right_click", False,
                                  {"physical_x": x, "physical_y": y},
                                  error=warn)
    result = _executor.click(x, y, button=3)
    _log_action("right_click", f"({x},{y})", success=result.success, error=result.error or "",
                workflow_args={"x": x, "y": y, "monitor": monitor})
    _restore_cursor(*saved_cursor)
    details: dict = {"physical_x": x, "physical_y": y,
                     "message": result.description if result.success else result.error or ""}
    if monitor > 0:
        details["screenshot_x"] = orig_x
        details["screenshot_y"] = orig_y
        details["monitor"] = monitor
    return _action_result("right_click", result.success, details,
                          error=result.error or "")


@mcp.tool()
def double_click(x: int, y: int, monitor: int = 0, force: bool = False) -> str:
    """指定したスクリーン座標をダブルクリックする。

    Args:
        x: X 座標（monitor 指定時はスクリーンショット内の座標）
        y: Y 座標（monitor 指定時はスクリーンショット内の座標）
        monitor: 0=絶対座標, 1以上=そのモニターのスクリーンショット座標として自動変換
        force: True なら座標検証ガードレールをスキップする
    """
    orig_x, orig_y = x, y
    saved_cursor = _save_cursor()
    if monitor > 0:
        x, y = _screen_to_physical(monitor, x, y)
    if not force:
        warn = _is_coord_verified(x, y)
        if warn:
            _restore_cursor(*saved_cursor)
            return _action_result("double_click", False,
                                  {"physical_x": x, "physical_y": y},
                                  error=warn)
    result = _executor.double_click(x, y)
    _log_action("double_click", f"({x},{y})", success=result.success, error=result.error or "",
                workflow_args={"x": x, "y": y, "monitor": monitor})
    _restore_cursor(*saved_cursor)
    details: dict = {"physical_x": x, "physical_y": y,
                     "message": result.description if result.success else result.error or ""}
    if monitor > 0:
        details["screenshot_x"] = orig_x
        details["screenshot_y"] = orig_y
        details["monitor"] = monitor
    return _action_result("double_click", result.success, details,
                          error=result.error or "")


@mcp.tool()
def drag(start_x: int, start_y: int, end_x: int, end_y: int,
         button: int = 1, duration: float = 0.5, monitor: int = 0) -> str:
    """ドラッグ操作。始点から終点までマウスボタンを押したまま移動する。
    ウィンドウのリサイズ、ファイル移動、範囲選択、描画などに使用。

    Args:
        start_x: ドラッグ開始 X 座標（monitor 指定時はスクリーンショット内の座標）
        start_y: ドラッグ開始 Y 座標（monitor 指定時はスクリーンショット内の座標）
        end_x: ドラッグ終了 X 座標（monitor 指定時はスクリーンショット内の座標）
        end_y: ドラッグ終了 Y 座標（monitor 指定時はスクリーンショット内の座標）
        button: マウスボタン (1=左, 2=中, 3=右)
        duration: ドラッグにかける時間（秒、デフォルト0.5）
        monitor: 0=絶対座標, 1以上=そのモニターのスクリーンショット座標として自動変換
    """
    orig_start = (start_x, start_y)
    orig_end = (end_x, end_y)
    saved_cursor = _save_cursor()
    if monitor > 0:
        start_x, start_y = _screen_to_physical(monitor, start_x, start_y)
        end_x, end_y = _screen_to_physical(monitor, end_x, end_y)
    result = _executor.drag(start_x, start_y, end_x, end_y,
                            button=button, duration=duration)
    _log_action("drag", f"({start_x},{start_y})→({end_x},{end_y})", success=result.success, error=result.error or "",
                workflow_args={"start_x": orig_start[0], "start_y": orig_start[1],
                               "end_x": orig_end[0], "end_y": orig_end[1],
                               "button": button, "duration": duration, "monitor": monitor})
    _restore_cursor(*saved_cursor)
    details: dict = {
        "physical_start": {"x": start_x, "y": start_y},
        "physical_end": {"x": end_x, "y": end_y},
        "button": button, "duration": duration,
        "message": result.description if result.success else result.error or "",
    }
    if monitor > 0:
        details["screenshot_start"] = {"x": orig_start[0], "y": orig_start[1]}
        details["screenshot_end"] = {"x": orig_end[0], "y": orig_end[1]}
        details["monitor"] = monitor
    return _action_result("drag", result.success, details,
                          error=result.error or "")


@mcp.tool()
def scroll(x: int, y: int, direction: str = "down", amount: int = 3,
           monitor: int = 0) -> str:
    """指定座標でスクロールする。

    Args:
        x: スクロール位置の X 座標（monitor 指定時はスクリーンショット内の座標）
        y: スクロール位置の Y 座標（monitor 指定時はスクリーンショット内の座標）
        direction: スクロール方向 ("up" または "down")
        amount: スクロール量（クリック数、デフォルト3）
        monitor: 0=絶対座標, 1以上=そのモニターのスクリーンショット座標として自動変換
    """
    orig_x, orig_y = x, y
    if monitor > 0:
        x, y = _screen_to_physical(monitor, x, y)
    result = _executor.scroll(x, y, direction=direction, amount=amount)
    _log_action("scroll", f"({x},{y}) {direction}×{amount}", success=result.success, error=result.error or "",
                workflow_args={"x": orig_x, "y": orig_y, "direction": direction,
                               "amount": amount, "monitor": monitor})
    details: dict = {"physical_x": x, "physical_y": y,
                     "direction": direction, "amount": amount,
                     "message": result.description if result.success else result.error or ""}
    if monitor > 0:
        details["monitor"] = monitor
    return _action_result("scroll", result.success, details,
                          error=result.error or "")


@mcp.tool()
def type_text(text: str, clear_first: bool = False) -> str:
    """現在のカーソル位置にテキストを入力する。日本語対応（クリップボード経由）。

    Args:
        text: 入力するテキスト
        clear_first: True なら入力前に Ctrl+A → Delete で既存テキストをクリア
    """
    denial = _blacklist.check_text(text)
    if denial:
        _log_action("type_text", value=text[:50], success=False, error="blacklisted")
        return _action_result("type_text", False, {"text": text[:50]},
                              error="blacklisted: " + denial)
    focus_ok = _restore_focus()
    if not focus_ok and _last_target_hwnd:
        _log_action("type_text", value=text, success=False, error="focus_lost")
        return _action_result("type_text", False, {"text": text[:50]},
                              error="focus_lost: 対象ウィンドウにフォーカスを取得できませんでした。focus_window で対象を再指定してください")
    result = _executor.type_text(text, clear_first=clear_first)
    _log_action("type_text", value=text, success=result.success, error=result.error or "",
                workflow_args={"text": text, "clear_first": clear_first})
    return _action_result("type_text", result.success,
                          {"text_length": len(text), "clear_first": clear_first,
                           "message": result.description if result.success else result.error or ""},
                          error=result.error or "")


@mcp.tool()
def keypress(key: str = "", keys: str = "") -> str:
    """キーまたはキーの組み合わせを押す。

    Args:
        key: キー名 (例: "Return", "Escape", "ctrl+s", "alt+tab", "win+s")
        keys: key のエイリアス（Neo 互換。key が空の場合に使用）
    """
    if not key and keys:
        key = keys
    if not key:
        return _action_result("keypress", False, error="key または keys を指定してください")
    denial = _blacklist.check_key(key)
    if denial:
        _log_action("keypress", value=key, success=False, error="blacklisted")
        return _action_result("keypress", False, {"key": key},
                              error="blacklisted: " + denial)
    focus_ok = _restore_focus()
    if not focus_ok and _last_target_hwnd:
        _log_action("keypress", value=key, success=False, error="focus_lost")
        return _action_result("keypress", False, {"key": key},
                              error="focus_lost: 対象ウィンドウにフォーカスを取得できませんでした。focus_window で対象を再指定してください")
    result = _executor.keypress(key)
    _log_action("keypress", value=key, success=result.success, error=result.error or "",
                workflow_args={"key": key})
    return _action_result("keypress", result.success,
                          {"key": key, "message": result.description if result.success else result.error or ""},
                          error=result.error or "")


@mcp.tool()
def focus_window(keyword: str) -> str:
    """ウィンドウタイトルにキーワードを含むウィンドウを検索してフォーカスする。

    Args:
        keyword: タイトルの検索キーワード (例: "メモ帳", "Chrome", "Excel")
    """
    denial = _blacklist.check_window(keyword)
    if denial:
        _log_action("focus_window", target=keyword, success=False, error="blacklisted")
        return denial
    result = _executor.focus_window(keyword)
    if result.success:
        # HWND を抽出して保存（description に "HWND=12345" が含まれる）
        import re
        m = re.search(r'HWND=(\d+)', result.description)
        if m:
            _set_target_hwnd(int(m.group(1)))
    _log_action("focus_window", target=keyword, success=result.success, error=result.error or "",
                workflow_args={"keyword": keyword})
    return result.description if result.success else f"失敗: {result.error}"


@mcp.tool()
def undo(count: int = 1) -> str:
    """直前の操作を取り消す（Ctrl+Z を送信）。テキスト入力やファイル操作の取消に使用。
    対象アプリが Ctrl+Z に対応している必要がある。

    Args:
        count: 取り消す回数（デフォルト1回、最大20回）
    """
    count = min(max(count, 1), 20)
    _restore_focus()
    results = []
    for i in range(count):
        result = _executor.keypress("ctrl+z")
        if not result.success:
            results.append(f"Undo {i+1}回目 失敗: {result.error}")
            break
        results.append(f"Undo {i+1}回目 成功")
        if count > 1:
            import time
            time.sleep(0.05)
    _log_action("undo", value=f"×{count}", success=True)
    # 直近の履歴を表示
    recent = list(_action_history)[-3:] if _action_history else []
    history_info = ""
    if recent:
        history_info = "\n直近の操作: " + " → ".join(
            f"{a['action']}({a.get('value') or a.get('target', '')})" for a in recent
        )
    return "\n".join(results) + history_info


@mcp.tool()
def action_log(last_n: int = 10) -> str:
    """MCP モードの操作ログを表示する。直近の操作履歴を確認してデバッグや振り返りに使用。

    Args:
        last_n: 表示する直近の件数（デフォルト10件、最大50件）
    """
    last_n = min(max(last_n, 1), 50)
    history = list(_action_history)[-last_n:]
    if not history:
        return "操作ログは空です"
    lines = [f"操作ログ（直近{len(history)}件、ログファイル: {_log_path}）:"]
    for i, a in enumerate(history, 1):
        status = "✓" if a["success"] else "✗"
        val = a.get("value") or a.get("target", "")
        lines.append(f"  {i}. [{a['time']}] {status} {a['action']} {val}")
    return "\n".join(lines)


@mcp.tool()
def list_windows() -> str:
    """デスクトップ上の全可視ウィンドウのタイトル一覧を返す。"""
    if sys.platform != "win32":
        return "Windows 以外では未対応です"
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

        exclude = {"", "program manager", "settings",
                   "microsoft text input application",
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
        return "\n".join(titles) if titles else "可視ウィンドウが見つかりません"
    except Exception as e:
        return f"エラー: {e}"


@mcp.tool()
def list_windows_detailed(include_hidden: bool = False,
                          include_minimized: bool = True) -> str:
    """デスクトップ上の全 top-level window の構造化メタデータを JSON で返す。

    W4 v0.1 (decision 058): `list_windows()` が改行区切り text を返すのに対し、
    本 tool は agent 自動化向けの構造化出力 (handle / title / app_name /
    visible / minimized / bounds / is_foreground) を返す。`capture_window`
    で参照する handle を取得する primary source。

    Args:
        include_hidden: IsWindowVisible=False の window も含む (default False)
        include_minimized: IsIconic=True の window も含む (default True、
            最小化中でも PrintWindow capture 可能なため)

    Returns:
        JSON string: [{handle, title, app_name, visible, minimized,
                       bounds: {x, y, w, h}, is_foreground}, ...]
        非 Win32 platform または enumerate 失敗時は ``{"error": "..."}`` を
        含む JSON を返却。
    """
    if sys.platform != "win32":
        import json as _json
        return _json.dumps({"error": "Windows 以外では未対応です (v0.1)"})
    try:
        from agent.window_capture import list_windows_structured
        wins = list_windows_structured(
            include_hidden=include_hidden,
            include_minimized=include_minimized,
        )
        import json as _json
        return _json.dumps(wins, ensure_ascii=False)
    except Exception as e:
        import json as _json
        return _json.dumps({"error": str(e)})


@mcp.tool()
def capture_window(title_regex: str = "", handle: int = 0,
                   occluded: bool = True,
                   resize_width: int = 0,
                   save_path: str = "",
                   privacy_blur_faces: bool = False,
                   privacy_redact_text: bool = False) -> list:
    """指定ウィンドウ単体をキャプチャし PNG bytes + ScreenshotMeta を返す。

    W4 v0.1 (decision 058): 既存 `screenshot(window=...)` は region-based で
    occluded 部分が隠れるが、本 tool は Win32 `PrintWindow` API で window に
    自身を描画させる → **occluded window でも自身のピクセルを取得可能**。

    既存 `screenshot()` と同じ return shape (Image + "SCREENSHOT_META:{...}")
    なので、VLM / OCR pipeline が変更なく消費できる。

    Args:
        title_regex: window title への正規表現 (re.search 経由、case-insensitive)
        handle: HWND。指定時は title_regex より優先
        occluded: True (default) = PrintWindow 経由、occluded 対応。
            PrintWindow 失敗時は region-capture fallback (logged warning)。
            False = 既存 `screenshot(window=)` と同じ region-only path
        resize_width: > 0 で幅を resize (0 = 元解像度)
        save_path: "" (default) = disk write なし (session-ephemeral)。
            path 指定時のみ PNG 保存、明示 opt-in
        privacy_blur_faces / privacy_redact_text: 既存 `apply_privacy_filters`
            を同 semantics で適用

    Returns:
        [Image(png_bytes), "SCREENSHOT_META:{...json...}"] or ["error message"]
    """
    if sys.platform != "win32":
        return ["Windows 以外では未対応です (v0.1)"]
    try:
        from agent.window_capture import (
            capture_window as _cap, find_window, WindowCaptureError)
    except ImportError as e:
        return [f"window_capture module import 失敗: {e}"]

    try:
        pil_img = _cap(title_regex=title_regex, handle=handle,
                       occluded=occluded, resize_width=resize_width,
                       save_path=save_path)
    except WindowCaptureError as e:
        logging.warning("capture_window 失敗: %s", e)
        return [f"capture_window 失敗: {e}"]

    # プライバシーフィルタ適用 (save_path 書込み後に reapply されるので、
    # save_path 用の bytes は「filter 前」のものになる仕様)
    if privacy_blur_faces or privacy_redact_text:
        from agent.privacy_filters import apply_privacy_filters
        pil_img = apply_privacy_filters(
            pil_img,
            blur_faces_enabled=privacy_blur_faces,
            redact_text_enabled=privacy_redact_text,
        )

    # 解決 HWND を log + meta に残す (debug + cross-call reference 用途)
    resolved_hwnd = 0
    try:
        resolved_hwnd = find_window(title_regex=title_regex, handle=handle)
    except WindowCaptureError:
        pass

    logging.info(
        "capture_window: hwnd=%s occluded=%s size=%sx%s save=%s",
        resolved_hwnd, occluded, pil_img.width, pil_img.height,
        bool(save_path),
    )

    buf = BytesIO()
    pil_img.save(buf, format="PNG")

    meta = {
        "screenshot_width": pil_img.width,
        "screenshot_height": pil_img.height,
        "window_handle": resolved_hwnd,
        "capture_method": "printwindow" if occluded else "region",
        "source": "capture_window",
    }
    import json as _json
    return [Image(data=buf.getvalue(), format="png"),
            f"SCREENSHOT_META:{_json.dumps(meta)}"]


@mcp.tool()
def save_file(filepath: str, app_keyword: str = "") -> str:
    """Windows の保存ダイアログ（名前を付けて保存）でファイルを保存する。
    手動で Alt+N → パス入力 → Enter を組み合わせるより確実。
    保存ダイアログが既に開いている状態で呼び出すこと。

    内部動作:
    1. 対象アプリにフォーカス復帰
    2. Alt+N でファイル名欄にフォーカス
    3. ファイル名欄をクリアしてフルパスを入力（クリップボード経由）
    4. Enter で保存実行
    5. 確認ダイアログ（上書き確認等）が出たら Enter で承認

    Args:
        filepath: 保存先のフルパス（例: "D:\\Documents\\report.txt"）
        app_keyword: 対象アプリのウィンドウキーワード（省略時は直前のフォーカスウィンドウを使用）
    """
    denial = _blacklist.check_filepath(filepath)
    if denial:
        _log_action("save_file", target=filepath, success=False, error="blacklisted")
        return denial

    import time as _t

    steps = []

    # 1. 対象アプリにフォーカス
    if app_keyword:
        result = _executor.focus_window(app_keyword)
        if not result.success:
            return f"失敗: '{app_keyword}' のウィンドウが見つかりません"
        import re
        m = re.search(r'HWND=(\d+)', result.description)
        if m:
            _set_target_hwnd(int(m.group(1)))
        steps.append(f"フォーカス: {result.description}")
        _t.sleep(0.1)
    else:
        _restore_focus()

    # 2. Alt+N でファイル名欄にフォーカス
    _executor.keypress("alt+n")
    _t.sleep(0.2)
    steps.append("Alt+N: ファイル名欄にフォーカス")

    # 3. 既存テキストをクリアしてパスを入力
    _executor.keypress("ctrl+a")
    _t.sleep(0.05)
    result = _executor.type_text(filepath)
    if not result.success:
        return f"失敗: パス入力エラー — {result.error}"
    steps.append(f"パス入力: {filepath}")
    _t.sleep(0.2)

    # 4. Enter で保存
    _executor.keypress("Return")
    steps.append("Enter: 保存実行")
    _t.sleep(0.5)

    # 5. 確認ダイアログ対応（上書き確認、形式確認等）
    # 短い待機後にもう一度 Enter（確認ダイアログがなければ無害）
    _executor.keypress("Return")
    steps.append("Enter: 確認ダイアログ承認（あれば）")

    _log_action("save_file", target=filepath, success=True,
                workflow_args={"filepath": filepath})
    return "保存完了: " + " → ".join(steps)


@mcp.tool()
def ocr_screen(monitor: int = 0, left: int = 0, top: int = 0,
               width: int = 0, height: int = 0) -> str:
    """画面上のテキストを Windows OCR で認識する。スクリーンショットでは読みにくい
    小さな文字やUI要素のテキストを正確に取得するのに有用。

    Args:
        monitor: モニター番号 (0=全画面, 1=プライマリ, ...)
        left: 認識領域の左端 X 座標 (0 で全画面)
        top: 認識領域の上端 Y 座標
        width: 認識領域の幅 (0 で全画面)
        height: 認識領域の高さ
    """
    text = ocr_region(monitor=monitor, left=left, top=top,
                      width=width, height=height)
    region = {"left": left, "top": top, "width": width, "height": height}
    if text:
        return json.dumps({"success": True, "text": text,
                           "monitor": monitor, "region": region},
                          ensure_ascii=False)
    return json.dumps({"success": False, "text": "",
                       "error": "no_text_detected",
                       "monitor": monitor, "region": region},
                      ensure_ascii=False)


@mcp.tool()
def verify_text(expected: str, monitor: int = 0, left: int = 0, top: int = 0,
                width: int = 0, height: int = 0) -> str:
    """AIが画面から読み取ったテキストをOCRと機械的に照合し、誤読を検出する。
    AIのハルシネーションに影響されない独立した検証手段。

    使い方: スクリーンショットで読み取ったテキストを expected に渡し、
    同じ領域を OCR で読み取った結果と Python の difflib で機械的に比較する。

    Args:
        expected: AIが読み取ったテキスト（検証対象）
        monitor: モニター番号 (0=全画面, 1=プライマリ, ...)
        left: 検証領域の左端 X 座標 (0 で全画面)
        top: 検証領域の上端 Y 座標
        width: 検証領域の幅 (0 で全画面)
        height: 検証領域の高さ
    """
    import difflib

    ocr_text = ocr_region(monitor=monitor, left=left, top=top,
                          width=width, height=height)
    if not ocr_text:
        return "検証不能: OCR がテキストを認識できませんでした"

    # 正規化（空白を全て除去して文字列レベルで比較 — OCRの1文字ずつスペース区切り問題を吸収）
    def normalize(s: str) -> str:
        return s.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

    norm_expected = normalize(expected)
    norm_ocr = normalize(ocr_text)

    # 文字列全体の類似度（空白除去した文字列同士で比較）
    ratio = difflib.SequenceMatcher(None, norm_expected, norm_ocr).ratio()

    # 文字レベルの差分（不一致箇所を特定）
    matcher = difflib.SequenceMatcher(None, norm_expected, norm_ocr)
    mismatches = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op != "equal":
            ai_part = norm_expected[i1:i2] if i1 < i2 else "(なし)"
            ocr_part = norm_ocr[j1:j2] if j1 < j2 else "(なし)"
            mismatches.append(f"{op}: AI「{ai_part}」 vs OCR「{ocr_part}」")

    # 結果レポート
    lines = [f"■ テキスト検証結果"]
    lines.append(f"  一致率: {ratio:.1%}")

    if ratio >= 0.90:
        lines.append(f"  判定: ✓ 高信頼（AI読取とOCRがほぼ一致）")
    elif ratio >= 0.7:
        lines.append(f"  判定: △ 要確認（部分的に不一致あり）")
    else:
        lines.append(f"  判定: ✗ 不一致（誤読の可能性が高い）")

    lines.append(f"  AI読取: {norm_expected[:120]}{'...' if len(norm_expected) > 120 else ''}")
    lines.append(f"  OCR読取: {norm_ocr[:120]}{'...' if len(norm_ocr) > 120 else ''}")

    if mismatches:
        lines.append(f"  不一致箇所 ({len(mismatches)}件):")
        for m in mismatches[:15]:
            lines.append(f"    {m}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 作業記憶ツール
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_store(action: str, target: str = "", result: str = "",
                 screen_state: str = "", note: str = "") -> str:
    """作業記憶にチャンクを保存する。複数ステップの作業中に中間状態を記録しておくと、
    後から参照できる。容量を超えると古い記憶から自動的に忘却される。

    Args:
        action: 実行した操作の要約（例: "Excelを開いてB3に売上データを入力"）
        target: 操作対象（例: "Excel B3セル", "メモ帳"）
        result: 操作結果（例: "成功、値123を入力済み"）
        screen_state: 操作後の画面状態の要約（例: "Excelが前面、B3に123が表示"）
        note: 補足メモ（例: "後でこの値をWord文書にコピーする必要あり"）
    """
    _working_memory.store(MemoryChunk(
        step=len(_working_memory) + 1,
        action_type=action,
        target=target,
        value=note,
        result=result,
        screen_state=screen_state,
    ))
    return json.dumps({"success": True, "action": "stored",
                       "count": len(_working_memory),
                       "capacity": _working_memory.capacity},
                      ensure_ascii=False)


@mcp.tool()
def memory_recall() -> str:
    """作業記憶の内容を全て取得する。直近の操作履歴と中間状態を確認できる。"""
    content = _working_memory.recall()
    return json.dumps({"success": True, "content": content or "",
                       "count": len(_working_memory),
                       "capacity": _working_memory.capacity},
                      ensure_ascii=False)


@mcp.tool()
def memory_clear() -> str:
    """作業記憶を全てクリアする。新しいタスクを始める時に使用。"""
    count = len(_working_memory)
    _working_memory.clear()
    return json.dumps({"success": True, "action": "cleared",
                       "deleted_count": count}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 長期記憶ツール（操作経験の永続化）
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_learn(title: str, content: str, app: str = "*",
                 category: str = "operational", context: str = "",
                 tags: str = "") -> str:
    """操作中に発見した知見を長期記憶に保存する。次のセッションでも利用できる。
    同じアプリ+タイトルの知識が既にあれば更新（確信度が上がる）。

    例: アプリの起動方法、ボタンの位置、操作のコツ、失敗から学んだ注意点など。

    Args:
        title: 知見のタイトル（例: "PAINT起動方法", "保存ダイアログの使い方"）
        content: 知見の詳細内容
        app: 関連アプリ名（"*" で汎用。例: "CLIP STUDIO PAINT", "Edge"）
        category: 種類（operational=操作方法, pitfall=注意点, pattern=成功パターン, app_specific=アプリ固有の発見）
        context: 発見時の状況（何をしようとしていたか）
        tags: カンマ区切りのタグ（例: "起動,ランチャー"）
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    entry = _long_term_memory.learn(
        category=category, app=app, title=title, content=content,
        context=context, tags=tag_list)
    action = "updated" if entry.access_count > 0 or entry.confidence > 0.7 else "created"
    return json.dumps({"success": True, "action": action, "id": entry.id,
                       "title": entry.title, "category": entry.category,
                       "app": entry.app, "confidence": round(entry.confidence, 2)},
                      ensure_ascii=False)


@mcp.tool()
def memory_knowledge(app: str = "", query: str = "", category: str = "",
                     limit: int = 10, entry_type: str = "") -> str:
    """長期記憶から過去の操作経験・知見を検索する。
    アプリ操作の前に参照すると、過去に学んだコツや注意点を活用できる。

    Args:
        app: アプリ名でフィルタ（空で全アプリ）
        query: キーワード検索（タイトル・内容・タグを検索）
        category: カテゴリでフィルタ（operational/pitfall/pattern/app_specific、空で全て）
        limit: 最大件数（デフォルト10）
        entry_type: 'fact' (短文事実) / 'segment' (動画/因果記憶) /
                    空文字 (両方) でフィルタ。decision 025 の filter 機能。
    """
    entries = _long_term_memory.recall(app=app, category=category,
                                       query=query, limit=limit,
                                       entry_type=entry_type)
    if not entries:
        return json.dumps({"success": True, "entries": [], "count": 0},
                          ensure_ascii=False)

    entry_list = []
    for e in entries:
        item = {
            "id": e.id, "title": e.title, "content": e.content,
            "category": e.category, "app": e.app,
            "confidence": round(e.confidence, 2),
            "access_count": e.access_count,
            "tags": e.tags if e.tags else [],
            "entry_type": e.entry_type,
        }
        # Segment 固有フィールドは値があるときだけ含める (fact の出力を
        # 汚さないため)
        if e.entry_type == "segment":
            for field in ("intent", "action_log", "ocr_dump",
                          "source_path", "timestamp_start",
                          "timestamp_end", "monitor_id"):
                val = getattr(e, field)
                if val is not None:
                    item[field] = val
        entry_list.append(item)
    return json.dumps({"success": True, "entries": entry_list,
                       "count": len(entry_list)}, ensure_ascii=False)


@mcp.tool()
async def memory_segment_learn(intent: str, action_log: str = "",
                         ocr_dump: str = "", ui_graph_json: str = "",
                         source_path: str = "",
                         timestamp_start: float = 0.0,
                         timestamp_end: float = 0.0,
                         monitor_id: int = 0,
                         app: str = "*", category: str = "pattern",
                         title: str = "", tags: str = "") -> str:
    """動画/画面/因果関係を含む Semantic Segment を長期記憶に保存する。

    memory_learn が短文の操作知識 (title+content) を記録するのに対し、
    こちらは「操作の意図 → 実行した action → 結果の画面状態」の
    因果関係を 1 行に記録する Autobiographical Causality 形式
    (Curry 2025 Memory-Node Encapsulation)。

    video/screenshot/cognitive workflow の記憶に使用する。
    AnalyzeVideo の出力や ClickWithCausality cognitive template の
    最終ノードから呼ばれることを想定。

    Args:
        intent: 操作の意図を自然言語で (必須、LLM 要約推奨)
                例: "Save the document in Notepad"
        action_log: 実行した action の記述
                    例: "Click(coord=(300,25))"
        ocr_dump: action 後の画面 OCR テキスト (結果の状態)
                  例: "File Edit Format View Help\\nmy_document.txt"
        ui_graph_json: 認識された重要 UI 要素の JSON 文字列 (FTS5 対象外)
                       例: '[{"type":"Button","name":"Save"}]'
        source_path: 元動画/画像ファイルのパス
        timestamp_start: 時間範囲の開始 (UNIX 秒 or 動画内秒、0 で未指定)
        timestamp_end: 時間範囲の終了 (同上、0 で未指定)
        monitor_id: マルチモニタ環境での screen 番号 (0 で未指定)
        app: アプリ名 (既存 memory_learn と同じ語彙)
        category: 記憶の種類 (operational/pitfall/pattern/app_specific)
        title: 明示的なタイトル (空なら intent を流用、length 100 まで)
        tags: カンマ区切りのタグ
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    # CPU-bound embedding + SQLite insert は worker thread に逃がして
    # MCP event loop の stdin/stdout reader を停止させない (Problem B fix)
    entry = await asyncio.to_thread(
        _long_term_memory.learn_segment,
        intent=intent,
        action_log=action_log,
        ocr_dump=ocr_dump,
        ui_graph_json=ui_graph_json,
        source_path=source_path,
        timestamp_start=timestamp_start if timestamp_start else None,
        timestamp_end=timestamp_end if timestamp_end else None,
        monitor_id=monitor_id if monitor_id else None,
        app=app,
        category=category,
        title=title,
        tags=tag_list,
    )
    result = {
        "success": True,
        "action": "created",
        "id": entry.id,
        "title": entry.title,
        "entry_type": entry.entry_type,
        "category": entry.category,
        "app": entry.app,
        "confidence": round(entry.confidence, 2),
        "vector_indexed": _long_term_memory.vec_enabled,
    }
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def memory_segment_knowledge(query: str, entry_type: str = "all",
                             fts_weight: float = 0.5,
                             vector_weight: float = 0.5,
                             app: str = "", category: str = "",
                             limit: int = 5,
                             time_range_start: Optional[float] = None,
                             time_range_end: Optional[float] = None) -> str:
    """長期記憶からハイブリッド検索 (FTS5 + vector + RRF) で抽出する。

    memory_knowledge が単純 FTS5 検索なのに対し、こちらは
    multilingual-e5-small でクエリを埋め込み、sqlite-vec による
    意味的近傍を FTS5 ランクと Reciprocal Rank Fusion で統合する。

    日本語/英語混在クエリ、"以前似たようなエラーが出た時" のような
    曖昧な自然言語クエリに強い。

    Args:
        query: 検索クエリ (人間言語、JP/EN/混在可、必須)
        entry_type: 'fact' / 'segment' / 'all' (default)
        fts_weight: FTS5 側の RRF 重み (0.0-1.0, default 0.5)
        vector_weight: vector 側の RRF 重み (0.0-1.0, default 0.5)
        app: アプリ名フィルタ (空で全アプリ、'*' は wildcard)
        category: category フィルタ (空で全て)
        limit: 最大返却件数 (default 5)
        time_range_start: 時間範囲フィルタ下限 (UNIX 秒 or 動画内秒、
            None で無効、v0.5.1)。指定時は Option A セマンティクス
            (NULL timestamp row 除外) で動作する。
        time_range_end: 時間範囲フィルタ上限 (UNIX 秒 or 動画内秒、
            None で無効、v0.5.1)。下限と独立に指定可能。
    """
    # Embedding encode + FTS5/vec query は worker thread に逃がす (Problem B fix)
    tuples = await asyncio.to_thread(
        _long_term_memory.recall_hybrid,
        query=query,
        entry_type=entry_type,
        fts_weight=fts_weight,
        vector_weight=vector_weight,
        app=app,
        category=category,
        limit=limit,
        time_range_start=time_range_start,
        time_range_end=time_range_end,
    )
    if not tuples:
        return json.dumps({
            "success": True, "results": [], "count": 0,
            "fts_count": 0, "vector_count": 0,
            "rrf_score_max": 0.0,
            "vector_enabled": _long_term_memory.vec_enabled,
        }, ensure_ascii=False)

    fts_count = sum(1 for _, f, _, _ in tuples if f is not None)
    vec_count = sum(1 for _, _, v, _ in tuples if v is not None)
    max_rrf = max(r for _, _, _, r in tuples)

    results = []
    for e, fts_score, vec_score, rrf in tuples:
        item = {
            "id": e.id,
            "entry_type": e.entry_type,
            "title": e.title,
            "content": e.content,
            "category": e.category,
            "app": e.app,
            "confidence": round(e.confidence, 2),
            "rrf_score": round(float(rrf), 6),
            "fts_score": float(fts_score) if fts_score is not None else None,
            "vector_score": float(vec_score) if vec_score is not None else None,
        }
        if e.entry_type == "segment":
            for field in ("intent", "action_log", "ocr_dump",
                          "source_path", "timestamp_start",
                          "timestamp_end", "monitor_id"):
                val = getattr(e, field)
                if val is not None:
                    item[field] = val
        results.append(item)

    return json.dumps({
        "success": True,
        "results": results,
        "count": len(results),
        "fts_count": fts_count,
        "vector_count": vec_count,
        "rrf_score_max": round(max_rrf, 6),
        "vector_enabled": _long_term_memory.vec_enabled,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# メンタルキャンバス（脳内画像）ツール
# ---------------------------------------------------------------------------

@mcp.tool()
def canvas_create(name: str, width: int = 800, height: int = 600,
                  color: str = "white", from_screenshot: bool = False,
                  monitor: int = 0) -> str:
    """脳内キャンバスを作成する。思考の可視化、操作シミュレーション、計画の図示に使用。

    Args:
        name: キャンバス名（後から参照するための識別子）
        width: 幅 (px)。from_screenshot=True の場合は無視
        height: 高さ (px)。from_screenshot=True の場合は無視
        color: 背景色（"white", "black", "red" 等、または未指定で白）
        from_screenshot: True なら現在の画面をキャプチャしてキャンバスにする
        monitor: from_screenshot 時のモニター番号 (0=全画面, 1=プライマリ, ...)
    """
    if from_screenshot:
        canvas = _canvas_mgr.from_screenshot(name, monitor=monitor)
    else:
        canvas = _canvas_mgr.create(name, width, height, color)
    return f"キャンバス '{name}' を作成しました ({canvas.width}x{canvas.height})"


@mcp.tool()
def canvas_draw(name: str, shape: str, x: int = 0, y: int = 0,
                x2: int = 0, y2: int = 0,
                w: int = 0, h: int = 0, r: int = 0,
                text: str = "", color: str = "red",
                fill: bool = False, size: int = 16,
                line_width: int = 2, label: str = "") -> str:
    """脳内キャンバスに図形やテキストを描画する。

    Args:
        name: 描画先のキャンバス名
        shape: 描画する図形の種類:
            "rect" — 矩形 (x, y, w, h で指定)
            "circle" — 円 (x, y=中心, r=半径)
            "line" — 直線 (x, y → x2, y2)
            "arrow" — 矢印 (x, y → x2, y2)
            "text" — テキスト (x, y に text を描画)
            "marker" — 十字マーカー (x, y に label 付き)
        x: X座標（矩形/テキストの左上、円の中心X、線の始点X）
        y: Y座標
        x2: 線/矢印の終点X
        y2: 線/矢印の終点Y
        w: 矩形の幅
        h: 矩形の高さ
        r: 円の半径
        text: テキスト内容（shape="text" の場合）
        color: 色名（"red", "blue", "green", "black" 等）
        fill: True なら塗りつぶし（rect, circle）
        size: テキストのフォントサイズ
        line_width: 線の太さ
        label: マーカーのラベル
    """
    canvas = _canvas_mgr.get(name)
    if not canvas:
        return f"エラー: キャンバス '{name}' が見つかりません"

    shape = shape.lower()
    if shape == "rect":
        canvas.draw_rect(x, y, w, h, color=color, fill=fill, width=line_width)
    elif shape == "circle":
        canvas.draw_circle(x, y, r, color=color, fill=fill, width=line_width)
    elif shape == "line":
        canvas.draw_line(x, y, x2, y2, color=color, width=line_width)
    elif shape == "arrow":
        canvas.draw_arrow(x, y, x2, y2, color=color, width=line_width)
    elif shape == "text":
        canvas.draw_text(x, y, text, color=color, size=size)
    elif shape == "marker":
        canvas.draw_marker(x, y, label=label, color=color, size=size)
    else:
        return f"エラー: 未知の図形 '{shape}'。rect/circle/line/arrow/text/marker が使えます"
    return f"描画しました: {shape} on '{name}'"


@mcp.tool()
def canvas_paste(target_name: str, source_name: str,
                 x: int = 0, y: int = 0) -> str:
    """あるキャンバスの内容を別のキャンバスに貼り付ける。

    Args:
        target_name: 貼り付け先のキャンバス名
        source_name: 貼り付け元のキャンバス名
        x: 貼り付け位置のX座標
        y: 貼り付け位置のY座標
    """
    target = _canvas_mgr.get(target_name)
    source = _canvas_mgr.get(source_name)
    if not target:
        return f"エラー: キャンバス '{target_name}' が見つかりません"
    if not source:
        return f"エラー: キャンバス '{source_name}' が見つかりません"
    target.paste_canvas(source, x, y)
    return f"'{source_name}' を '{target_name}' の ({x},{y}) に貼り付けました"


@mcp.tool()
def canvas_view(name: str) -> Image:
    """脳内キャンバスの現在の内容を画像として返す。AIがキャンバスの状態を確認するために使用。

    Args:
        name: 表示するキャンバス名
    """
    canvas = _canvas_mgr.get(name)
    if not canvas:
        # エラー時は赤い1x1画像を返す
        return Image(data=b'\x89PNG\r\n\x1a\n', format="png")
    return Image(data=canvas.to_bytes("PNG"), format="png")


@mcp.tool()
def canvas_snapshot(name: str, label: str = "default",
                    restore: bool = False) -> str:
    """キャンバスの状態を保存または復元する。シミュレーション前に保存し、結果が気に入らなければ復元。

    Args:
        name: キャンバス名
        label: スナップショットのラベル（複数保存可能）
        restore: True なら保存ではなく復元
    """
    canvas = _canvas_mgr.get(name)
    if not canvas:
        return f"エラー: キャンバス '{name}' が見つかりません"

    if restore:
        if canvas.restore(label):
            return f"スナップショット '{label}' から復元しました"
        return f"エラー: スナップショット '{label}' が見つかりません。保存済み: {canvas.list_snapshots()}"
    else:
        canvas.snapshot(label)
        return f"スナップショット '{label}' を保存しました (全{len(canvas._snapshots)}件)"


@mcp.tool()
def canvas_compare(name1: str, name2: str = "", snapshot_label: str = "") -> str:
    """2つのキャンバスまたはスナップショットを比較して差分を返す。

    Args:
        name1: 比較元のキャンバス名
        name2: 比較先のキャンバス名（空の場合はスナップショットと比較）
        snapshot_label: name2 が空の場合、このラベルのスナップショットと比較（デフォルト: "default"）
    """
    canvas1 = _canvas_mgr.get(name1)
    if not canvas1:
        return f"エラー: キャンバス '{name1}' が見つかりません"

    if name2:
        canvas2 = _canvas_mgr.get(name2)
        if not canvas2:
            return f"エラー: キャンバス '{name2}' が見つかりません"
        result = canvas1.compare_with(canvas2)
    else:
        label = snapshot_label or "default"
        result = canvas1.compare_with_snapshot(label)
        if result is None:
            return f"エラー: スナップショット '{label}' が見つかりません"

    return result.summary


@mcp.tool()
def canvas_list() -> str:
    """全てのキャンバスの一覧を返す。"""
    details = _canvas_mgr.list_detail()
    if not details:
        return "キャンバスはありません"
    lines = [f"  {d['name']}: {d['size']} (snapshots={d['snapshots']})" for d in details]
    return f"キャンバス ({len(details)}件):\n" + "\n".join(lines)


@mcp.tool()
def canvas_delete(name: str = "", all: bool = False) -> str:
    """キャンバスを削除する。

    Args:
        name: 削除するキャンバス名
        all: True なら全キャンバスを削除
    """
    if all:
        count = _canvas_mgr.clear_all()
        return f"全キャンバスを削除しました ({count}件)"
    if _canvas_mgr.delete(name):
        return f"キャンバス '{name}' を削除しました"
    return f"エラー: キャンバス '{name}' が見つかりません"


# ---------------------------------------------------------------------------
# ガイド描画 (Guided Drawing)
# ---------------------------------------------------------------------------

@mcp.tool()
@_preserves_screenshot_state
def canvas_guide_draw(
    shape: str = "circle",
    num_points: int = 36,
    monitor: int = 3,
    cx: int = 0, cy: int = 0,
    r: int = 0,
    radius_ratio: float = 0.35,
    execute: bool = False,
    step_duration: float = 0.02,
    guide_color: str = "green",
) -> str:
    """メンタルキャンバスで描画ガイドを生成し、座標列を返す。
    スクリーンショットからペイントソフトのキャンバス領域を自動検出し、
    指定した図形の描画パス座標を計算する。

    フロー: screenshot → 白領域検出 → 図形座標生成 → ガイド描画 → 座標返却

    Args:
        shape: 図形タイプ ("circle" or "ellipse")
        num_points: 輪郭の点数 (多いほど滑らか、デフォルト36)
        monitor: キャプチャ対象モニター番号
        cx: 図形中心X (0=自動検出)
        cy: 図形中心Y (0=自動検出)
        r: 半径 (0=自動計算、キャンバスの短辺×radius_ratio)
        radius_ratio: 自動半径のキャンバス短辺に対する比率 (デフォルト0.35)
        execute: Trueなら座標返却に加えてdrag_pathも実行する
        step_duration: drag_path実行時の各点間の移動時間(秒)
        guide_color: ガイド描画の色 (デフォルト: green)
    """
    import json
    from PIL import Image as PILImage
    from agent.mental_canvas import MentalCanvas

    # 1. スクリーンショット撮影 (フルサイズ = 物理解像度)
    cap = ScreenCapture(monitor_index=monitor, resize_width=0)
    pil_img = cap.capture_as_pil()
    if pil_img is None:
        return "エラー: スクリーンショット撮影に失敗しました"

    # 2. メンタルキャンバスに取り込み
    canvas_name = "_guide_draw"
    canvas = MentalCanvas(width=pil_img.width, height=pil_img.height)
    canvas.paste(pil_img)
    _canvas_mgr._canvases[canvas_name] = canvas

    # スクリーンショットサイズを記録（座標変換用）
    _last_screenshot_info[monitor] = (pil_img.width, pil_img.height)

    # 3. 白い矩形領域を検出
    region = canvas.detect_white_region()
    if region is None:
        return "エラー: 白いキャンバス領域が検出できませんでした"
    rx, ry, rw, rh = region

    # 4. 中心と半径を決定
    if cx == 0:
        cx = rx + rw // 2
    if cy == 0:
        cy = ry + rh // 2
    if r == 0:
        r = int(min(rw, rh) * radius_ratio)

    # 5. 座標点列を生成
    points = MentalCanvas.generate_shape_points(
        shape=shape, cx=cx, cy=cy, r=r, num_points=num_points)

    # 6. ガイドをキャンバスに描画
    canvas.draw_circle(cx, cy, r, color=guide_color, width=3)
    canvas.draw_marker(cx, cy, label=f"center({cx},{cy})", color=guide_color)

    # 7. スクリーンショット座標→物理座標に変換
    physical_points = []
    for px, py in points:
        phys_x, phys_y = _screen_to_physical(monitor, px, py)
        physical_points.append((phys_x, phys_y))

    result = {
        "detected_region": {"x": rx, "y": ry, "w": rw, "h": rh},
        "shape_center": {"cx": cx, "cy": cy},
        "radius": r,
        "points_count": len(points),
        "points_screenshot": points,
        "points_physical": physical_points,
    }

    # 8. 実行
    if execute:
        exec_result = _executor.drag_path(
            physical_points, button=1, step_duration=step_duration)
        result["execution"] = exec_result.description
        _log_action("canvas_guide_draw", f"{shape} r={r} {len(points)}pts execute=True",
                    success=exec_result.success, error=exec_result.error or "")

    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def drag_path(
    points: str,
    button: int = 1,
    step_duration: float = 0.02,
    monitor: int = 0,
) -> str:
    """複数の座標を通るドラッグ操作。マウスボタンを離さずに全座標を順に辿る。
    ペイントソフトでの滑らかな曲線描画に使用。

    Args:
        points: JSON配列文字列 [[x1,y1], [x2,y2], ...]
        button: マウスボタン (1=左, 2=中, 3=右)
        step_duration: 各ポイント間の移動時間（秒、デフォルト0.02）
        monitor: 0=絶対座標, 1以上=そのモニターのスクリーンショット座標として自動変換
    """
    import json
    try:
        pts = json.loads(points)
    except json.JSONDecodeError as e:
        return _action_result("drag_path", False, error=f"JSON解析失敗: {e}")

    if len(pts) < 2:
        return _action_result("drag_path", False, error="2点以上の座標が必要です")

    saved_cursor = _save_cursor()
    num_points = len(pts)

    # 座標変換
    if monitor > 0:
        pts = [list(_screen_to_physical(monitor, p[0], p[1])) for p in pts]

    result = _executor.drag_path(
        [(p[0], p[1]) for p in pts],
        button=button, step_duration=step_duration)
    _log_action("drag_path", f"{num_points}点パスドラッグ",
                success=result.success, error=result.error or "")
    _restore_cursor(*saved_cursor)
    details: dict = {
        "num_points": num_points, "button": button,
        "step_duration": step_duration,
        "message": result.description if result.success else result.error or "",
    }
    if monitor > 0:
        details["monitor"] = monitor
    return _action_result("drag_path", result.success, details,
                          error=result.error or "")


# ---------------------------------------------------------------------------
# 筆圧ペンストローク
# ---------------------------------------------------------------------------

@mcp.tool()
def pen_stroke(
    points: str,
    pressures: str = "",
    pressure_curve: str = "bell",
    peak_pressure: int = 800,
    interval_ms: float = 8.0,
    step_px: float = 3.0,
    tilt_x: int = 0,
    tilt_y: int = 0,
    monitor: int = 0,
) -> str:
    """筆圧付きペンストロークを描画する。CSP等のペイントソフトに仮想ペンタブレットとして入力。
    対象アプリは「タブレットPC」モード（WM_POINTER受信）に設定が必要。

    Args:
        points: JSON配列 [[x1,y1], [x2,y2], ...] ウェイポイント座標（2点以上）
        pressures: JSON配列 [0..1024, ...] 各点の筆圧（省略時は pressure_curve で自動生成）
        pressure_curve: 筆圧カーブ種類 "bell"(入り抜き) / "linear" / "attack"(入り重視) / "constant"
        peak_pressure: カーブ生成時のピーク筆圧 (0-1024)
        interval_ms: ポイント間の待機時間(ミリ秒)。短すぎるとエラー
        step_px: ウェイポイント間の補間間隔(ピクセル)。小さいほど滑らか
        tilt_x: ペンの傾き X軸 (-90..+90度)
        tilt_y: ペンの傾き Y軸 (-90..+90度)
        monitor: 0=物理座標, 1以上=そのモニターのスクリーンショット座標として自動変換
    """
    import json
    from agent.pen_input import (
        inject_pen_stroke, interpolate_points,
        pressure_curve_bell, pressure_curve_linear,
        pressure_curve_attack,
    )

    try:
        waypoints = json.loads(points)
    except json.JSONDecodeError as e:
        return f"エラー: points JSON解析失敗 — {e}"

    if len(waypoints) < 2:
        return "エラー: 2点以上の座標が必要です"

    # 座標変換
    if monitor > 0:
        waypoints = [list(_screen_to_physical(monitor, p[0], p[1])) for p in waypoints]

    # ウェイポイント間を補間して滑らかな座標列を生成
    smooth_pts = interpolate_points(
        [(p[0], p[1]) for p in waypoints], step_px=step_px)

    n = len(smooth_pts)

    # 筆圧リスト
    if pressures:
        try:
            press_list = json.loads(pressures)
        except json.JSONDecodeError as e:
            return f"エラー: pressures JSON解析失敗 — {e}"
        # 補間後の点数に合わせてリサンプリング
        if len(press_list) != n:
            orig_n = len(press_list)
            press_list = [
                press_list[int(i * (orig_n - 1) / (n - 1))] for i in range(n)
            ]
    else:
        # 自動生成
        peak = max(0, min(1024, peak_pressure))
        if pressure_curve == "bell":
            press_list = pressure_curve_bell(n, peak=peak)
        elif pressure_curve == "attack":
            press_list = pressure_curve_attack(n, peak=peak)
        elif pressure_curve == "constant":
            press_list = [peak] * n
        else:  # linear
            press_list = pressure_curve_linear(n, start=peak, end=peak)

    saved_cursor = _save_cursor()

    result = inject_pen_stroke(
        smooth_pts, press_list,
        interval_ms=interval_ms, tilt_x=tilt_x, tilt_y=tilt_y,
    )

    _restore_cursor(*saved_cursor)

    _log_action("pen_stroke", f"{n}点ペンストローク (curve={pressure_curve})",
                success=result["success"], error=result.get("error", ""))

    if result["success"]:
        return (f"✓ ペンストローク完了: {result['points_injected']}点注入, "
                f"curve={pressure_curve}, peak={peak_pressure}")
    else:
        return f"失敗: {result['error']} ({result['points_injected']}/{n}点注入)"


@mcp.tool()
def pen_bezier(
    p0: str, p1: str, p2: str, p3: str,
    num_points: int = 50,
    pressure_curve: str = "bell",
    peak_pressure: int = 800,
    interval_ms: float = 8.0,
    tilt_x: int = 0,
    tilt_y: int = 0,
    monitor: int = 0,
) -> str:
    """3次ベジェ曲線で筆圧付きペンストロークを描画する。滑らかな曲線描画に最適。

    Args:
        p0: 始点 "[x,y]"
        p1: 制御点1 "[x,y]" （曲線の曲がり具合を決める）
        p2: 制御点2 "[x,y]"
        p3: 終点 "[x,y]"
        num_points: 曲線の分割数（多いほど滑らか）
        pressure_curve: 筆圧カーブ "bell" / "linear" / "attack" / "constant"
        peak_pressure: ピーク筆圧 (0-1024)
        interval_ms: ポイント間の待機時間(ミリ秒)
        tilt_x: ペンの傾き X軸 (-90..+90度)
        tilt_y: ペンの傾き Y軸 (-90..+90度)
        monitor: 0=物理座標, 1以上=スクリーンショット座標
    """
    import json
    from agent.pen_input import (
        inject_pen_stroke, bezier_points,
        pressure_curve_bell, pressure_curve_linear,
        pressure_curve_attack,
    )

    try:
        pt0 = tuple(json.loads(p0))
        pt1 = tuple(json.loads(p1))
        pt2 = tuple(json.loads(p2))
        pt3 = tuple(json.loads(p3))
    except (json.JSONDecodeError, TypeError) as e:
        return f"エラー: 座標JSON解析失敗 — {e}"

    # ベジェ曲線の座標列生成
    pts = bezier_points(pt0, pt1, pt2, pt3, num_points=num_points)

    # 座標変換
    if monitor > 0:
        pts = [_screen_to_physical(monitor, p[0], p[1]) for p in pts]

    n = len(pts)
    peak = max(0, min(1024, peak_pressure))
    if pressure_curve == "bell":
        press_list = pressure_curve_bell(n, peak=peak)
    elif pressure_curve == "attack":
        press_list = pressure_curve_attack(n, peak=peak)
    elif pressure_curve == "constant":
        press_list = [peak] * n
    else:
        press_list = pressure_curve_linear(n, start=peak, end=peak)

    saved_cursor = _save_cursor()

    result = inject_pen_stroke(
        pts, press_list,
        interval_ms=interval_ms, tilt_x=tilt_x, tilt_y=tilt_y,
    )

    _restore_cursor(*saved_cursor)

    _log_action("pen_bezier", f"ベジェ曲線 {n}点ペンストローク",
                success=result["success"], error=result.get("error", ""))

    if result["success"]:
        return (f"✓ ベジェ曲線ストローク完了: {result['points_injected']}点注入, "
                f"curve={pressure_curve}")
    else:
        return f"失敗: {result['error']} ({result['points_injected']}/{n}点注入)"


# ---------------------------------------------------------------------------
# ストローク抽出・筆圧生成パイプライン
# ---------------------------------------------------------------------------

@mcp.tool()
def contour_to_strokes(
    image_path: str,
    min_length_px: float = 10.0,
    simplify_epsilon: float = 2.0,
    canny_low: int = 50,
    canny_high: int = 150,
) -> str:
    """画像からエッジを検出し、描画可能なストロークリストに変換する。

    内部で Canny エッジ検出 → findContours → approxPolyDP を実行。
    pen_stroke で描画するための前処理として使用。

    Args:
        image_path: 入力画像のパス（PNG/JPG）
        min_length_px: この長さ未満のストロークを除去（ノイズ除去）
        simplify_epsilon: 輪郭近似精度（小=忠実、大=滑らか。デフォルト: 2.0）
        canny_low: Canny エッジ検出の低閾値
        canny_high: Canny エッジ検出の高閾値
    """
    import json as _json
    import cv2
    import numpy as np
    from agent.stroke_pipeline import contour_to_strokes as _contour_to_strokes

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return _action_result("contour_to_strokes", False,
                                  error=f"画像を読み込めません: {image_path}")

        edges = cv2.Canny(img, canny_low, canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        result = _contour_to_strokes(contours, min_length_px, simplify_epsilon)
        return _action_result("contour_to_strokes", True, {
            "stroke_count": result["stroke_count"],
            "strokes": result["strokes"],
            "image_size": {"w": int(img.shape[1]), "h": int(img.shape[0])},
        })
    except Exception as e:
        return _action_result("contour_to_strokes", False, error=str(e))


@mcp.tool()
def stroke_ordering(
    strokes: str,
    start_position: str = "[0, 0]",
    strategy: str = "nearest",
) -> str:
    """ストロークを描画順に並べ替える。ペンの空中移動距離を最小化。

    Args:
        strokes: JSON配列 — contour_to_strokes の出力の strokes フィールド
        start_position: JSON [x, y] — 描画開始位置
        strategy: "nearest"(最近傍) / "top-to-bottom" / "left-to-right"
    """
    import json as _json
    from agent.stroke_pipeline import stroke_ordering as _stroke_ordering

    try:
        stroke_list = _json.loads(strokes) if isinstance(strokes, str) else strokes
        start = _json.loads(start_position) if isinstance(start_position, str) else start_position
    except _json.JSONDecodeError as e:
        return _action_result("stroke_ordering", False, error=f"JSON解析失敗: {e}")

    try:
        result = _stroke_ordering(stroke_list, start, strategy)
        return _action_result("stroke_ordering", True, {
            "ordered_strokes": result["ordered_strokes"],
            "total_draw_length": result["total_draw_length"],
            "total_lift_length": result["total_lift_length"],
            "stroke_count": len(result["ordered_strokes"]),
        })
    except Exception as e:
        return _action_result("stroke_ordering", False, error=str(e))


@mcp.tool()
def stroke_smoothing(
    strokes: str,
    method: str = "bezier",
    smoothness: float = 0.5,
    jitter: float = 0.0,
) -> str:
    """ストロークの点列を平滑化する。ベジェフィッティングまたは移動平均。

    Args:
        strokes: JSON配列 — Stroke のリスト
        method: "bezier" / "moving_average" / "catmull_rom"
        smoothness: 平滑度 0.0-1.0（0=元のまま、1=最大平滑化）
        jitter: 手描き揺れ量(px)。0=揺れなし、1-2で自然な手描き感
    """
    import json as _json
    from agent.stroke_pipeline import stroke_smoothing as _stroke_smoothing

    try:
        stroke_list = _json.loads(strokes) if isinstance(strokes, str) else strokes
    except _json.JSONDecodeError as e:
        return _action_result("stroke_smoothing", False, error=f"JSON解析失敗: {e}")

    try:
        result = _stroke_smoothing(stroke_list, method, smoothness, jitter)
        return _action_result("stroke_smoothing", True, {
            "smoothed_strokes": result["smoothed_strokes"],
            "stroke_count": len(result["smoothed_strokes"]),
        })
    except Exception as e:
        return _action_result("stroke_smoothing", False, error=str(e))


@mcp.tool()
def stroke_to_coordinates(
    strokes: str,
    canvas_rect: str,
    image_size: str,
) -> str:
    """ストロークを物理座標に変換し、pen_stroke に渡せる形にする。

    画像上の相対座標 → 画面上の物理座標（CSP キャンバス位置に合わせる）。

    Args:
        strokes: JSON配列 — Stroke のリスト
        canvas_rect: JSON {"x","y","w","h"} — CSPキャンバスの画面上の位置（物理座標）
        image_size: JSON {"w","h"} — 元画像のサイズ
    """
    import json as _json
    from agent.stroke_pipeline import stroke_to_coordinates as _stroke_to_coordinates

    try:
        stroke_list = _json.loads(strokes) if isinstance(strokes, str) else strokes
        rect = _json.loads(canvas_rect) if isinstance(canvas_rect, str) else canvas_rect
        img_sz = _json.loads(image_size) if isinstance(image_size, str) else image_size
    except _json.JSONDecodeError as e:
        return _action_result("stroke_to_coordinates", False, error=f"JSON解析失敗: {e}")

    try:
        result = _stroke_to_coordinates(stroke_list, rect, img_sz)
        return _action_result("stroke_to_coordinates", True, {
            "stroke_coords": result["stroke_coords"],
            "stroke_count": result["stroke_count"],
        })
    except Exception as e:
        return _action_result("stroke_to_coordinates", False, error=str(e))


@mcp.tool()
def pressure_curve_generate(
    num_points: int,
    curve_type: str = "bell",
    base_pressure: float = 0.6,
    variation: float = 0.3,
) -> str:
    """ストロークの点数に対応する筆圧カーブを生成する。

    Args:
        num_points: ストロークの点数
        curve_type: "bell"(入り抜き) / "linear" / "attack"(入り強め) / "constant"
        base_pressure: 基本筆圧 0.0-1.0
        variation: 筆圧変動幅 0.0-0.5
    """
    from agent.stroke_pipeline import pressure_curve_generate as _pcg

    try:
        result = _pcg(num_points, curve_type, base_pressure, variation)
        return _action_result("pressure_curve_generate", True, {
            "pressure_curve": result,
            "num_points": len(result["values"]),
        })
    except Exception as e:
        return _action_result("pressure_curve_generate", False, error=str(e))


@mcp.tool()
def pressure_from_curvature(
    stroke: str,
    min_pressure: float = 0.1,
    max_pressure: float = 0.9,
    curvature_sensitivity: float = 0.5,
) -> str:
    """ストロークの曲率に基づいて筆圧を自動生成する。

    直線部分は太く（高筆圧）、急カーブは細く（低筆圧）。漫画Gペンの描き味を再現。
    終端にテーパー（抜き）を自動付与。

    Args:
        stroke: JSON — 単一 Stroke {"points": [[x,y],...], "is_closed": bool, ...}
        min_pressure: 最小筆圧 0.0-1.0
        max_pressure: 最大筆圧 0.0-1.0
        curvature_sensitivity: 曲率感度 0.0-1.0（高いほどメリハリ）
    """
    import json as _json
    from agent.stroke_pipeline import pressure_from_curvature as _pfc

    try:
        stroke_data = _json.loads(stroke) if isinstance(stroke, str) else stroke
    except _json.JSONDecodeError as e:
        return _action_result("pressure_from_curvature", False, error=f"JSON解析失敗: {e}")

    try:
        result = _pfc(stroke_data, min_pressure, max_pressure, curvature_sensitivity)
        return _action_result("pressure_from_curvature", True, {
            "pressure_curve": result,
            "num_points": len(result["values"]),
        })
    except Exception as e:
        return _action_result("pressure_from_curvature", False, error=str(e))


@mcp.tool()
def pressure_merge(
    curves: str,
    method: str = "multiply",
) -> str:
    """複数の筆圧カーブを合成する。bell × curvature のように重ね合わせ可能。

    Args:
        curves: JSON配列 — PressureCurve のリスト [{"values": [...], "curve_type": str}, ...]
        method: "multiply"(掛け算) / "average"(平均) / "max"(最大値)
    """
    import json as _json
    from agent.stroke_pipeline import pressure_merge as _pm

    try:
        curve_list = _json.loads(curves) if isinstance(curves, str) else curves
    except _json.JSONDecodeError as e:
        return _action_result("pressure_merge", False, error=f"JSON解析失敗: {e}")

    try:
        result = _pm(curve_list, method)
        return _action_result("pressure_merge", True, {
            "pressure_curve": result,
            "num_points": len(result["values"]),
        })
    except Exception as e:
        return _action_result("pressure_merge", False, error=str(e))


# ---------------------------------------------------------------------------
# 弁証法的推論ツール
# ---------------------------------------------------------------------------

@mcp.tool()
def dialectic_start(name: str, goal: str, thesis: str) -> str:
    """弁証法的推論を開始する。いきなりゴールを目指すのではなく、テーゼに対して
    意図的にアンチテーゼ（反論・矛盾・別視点）を設定し、止揚してより良い結論に到達する。

    Args:
        name: セッション名（後から参照するための識別子）
        goal: 最終的に達成したいゴール
        thesis: 初期のテーゼ（主張・計画・仮説）
    """
    session = _dialectic.start(name, goal, thesis)
    return json.dumps({"success": True, "session": name, "round": 1,
                       "phase": "thesis", "goal": goal, "thesis": thesis,
                       "next": "dialectic_challenge"}, ensure_ascii=False)


@mcp.tool()
def dialectic_challenge(name: str, antithesis: str) -> str:
    """テーゼに対するアンチテーゼ（反論・矛盾・別の視点）を設定する。
    テーゼの弱点、見落とし、前提の誤りなどを意図的に提示する。

    Args:
        name: セッション名
        antithesis: テーゼへの反論・矛盾・別の視点
    """
    triad = _dialectic.challenge(name, antithesis)
    if not triad:
        return json.dumps({"success": False, "session": name,
                           "error": "session_not_found_or_invalid_phase"}, ensure_ascii=False)
    return json.dumps({"success": True, "session": name, "round": triad.round,
                       "phase": "antithesis", "thesis": triad.thesis,
                       "antithesis": triad.antithesis,
                       "next": "dialectic_synthesize"}, ensure_ascii=False)


@mcp.tool()
def dialectic_synthesize(name: str, synthesis: str) -> str:
    """テーゼとアンチテーゼを止揚（アウフヘーベン）してジンテーゼを導く。
    両者の対立を解消し、より高次の結論を提示する。

    Args:
        name: セッション名
        synthesis: 止揚された高次の結論
    """
    triad = _dialectic.synthesize(name, synthesis)
    if not triad:
        return json.dumps({"success": False, "session": name,
                           "error": "session_not_found_or_antithesis_missing"}, ensure_ascii=False)
    return json.dumps({"success": True, "session": name, "round": triad.round,
                       "phase": "synthesis", "thesis": triad.thesis,
                       "antithesis": triad.antithesis, "synthesis": triad.synthesis,
                       "next": "dialectic_conclude or dialectic_iterate"},
                      ensure_ascii=False)


@mcp.tool()
def dialectic_iterate(name: str, new_antithesis: str = "") -> str:
    """現在のジンテーゼを新たなテーゼとして次のラウンドを開始する。
    螺旋的に思考を深めていく。

    Args:
        name: セッション名
        new_antithesis: 新ラウンドのアンチテーゼ（省略時は後から dialectic_challenge で設定）
    """
    triad = _dialectic.iterate(name, new_antithesis)
    if not triad:
        return json.dumps({"success": False, "session": name,
                           "error": "session_not_found_or_incomplete"}, ensure_ascii=False)
    result = {"success": True, "session": name, "round": triad.round,
              "phase": "antithesis" if triad.antithesis else "thesis",
              "thesis": triad.thesis}
    if triad.antithesis:
        result["antithesis"] = triad.antithesis
        result["next"] = "dialectic_synthesize"
    else:
        result["next"] = "dialectic_challenge"
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def dialectic_conclude(name: str, conclusion: str = "") -> str:
    """弁証法セッションを結論づける。

    Args:
        name: セッション名
        conclusion: 最終結論（省略時は最新のジンテーゼを使用）
    """
    result = _dialectic.conclude(name, conclusion)
    if result is None:
        return json.dumps({"success": False, "session": name,
                           "error": "session_not_found"}, ensure_ascii=False)
    return json.dumps({"success": True, "session": name,
                       "conclusion": result}, ensure_ascii=False)


@mcp.tool()
def dialectic_view(name: str) -> str:
    """弁証法セッションの全体像（全ラウンドのテーゼ・アンチテーゼ・ジンテーゼ）を表示する。

    Args:
        name: セッション名
    """
    view_text = _dialectic.view(name)
    if view_text is None:
        return json.dumps({"success": False, "session": name,
                           "error": "session_not_found"}, ensure_ascii=False)
    return json.dumps({"success": True, "session": name,
                       "view": view_text}, ensure_ascii=False)


@mcp.tool()
def dialectic_list() -> str:
    """全ての弁証法セッションの一覧を返す。"""
    sessions = _dialectic.list_sessions()
    return json.dumps({"success": True, "sessions": sessions},
                      ensure_ascii=False)


# ---------------------------------------------------------------------------
# ブラックリスト管理ツール
# ---------------------------------------------------------------------------

@mcp.tool()
def blacklist_show() -> str:
    """現在のブラックリスト（操作禁止リスト）設定を表示する。
    組み込みデフォルトとユーザー設定の両方を含む。"""
    return _blacklist.summary()


@mcp.tool()
def blacklist_reload() -> str:
    """ブラックリスト設定ファイルを再読み込みする。
    blacklist.json を編集した後に呼び出すと、サーバー再起動なしで新設定が反映される。"""
    return _blacklist.reload()


@mcp.tool()
def blacklist_add(category: str, pattern: str) -> str:
    """ブラックリストにルールを追加する。ユーザーの自然言語指示を解釈して呼び出す。
    追加後は即座に有効になり、blacklist.json に永続化される。

    Args:
        category: ルール種別。以下のいずれか:
            "key"    — 禁止キー (例: "win+l", "ctrl+shift+delete")
            "path"   — 禁止パス (例: "D:/secret/**", "**/.env")
            "window" — 禁止ウィンドウ (例: "*メール*", "*Outlook*")
            "text"   — 禁止テキスト (例: "rm -rf", "format c:")
        pattern: 追加するパターン。glob 形式 (* ? **) をサポート。
    """
    return _blacklist.add(category, pattern)


@mcp.tool()
def blacklist_remove(category: str, pattern: str) -> str:
    """ブラックリストからルールを削除する。組み込みデフォルトは削除できない。

    Args:
        category: ルール種別 ("key", "path", "window", "text")
        pattern: 削除するパターン（blacklist_show で表示される値と一致させること）
    """
    return _blacklist.remove(category, pattern)


# ---------------------------------------------------------------------------
# ゴール明確化 — 視覚的計画プレビュー
# ---------------------------------------------------------------------------

@mcp.tool()
def plan_preview(steps: str, monitor: int = 0) -> Image:
    """操作計画を現在の画面に重ねて可視化する。
    ユーザーの指示を実行する前に、計画を視覚的に提示して確認を得るために使用。
    注釈付きスクリーンショットを返す。

    使い方:
    1. screenshot で画面を確認し、操作対象の座標を特定する
    2. 計画ステップを steps にJSON配列で渡す
    3. 返った注釈画像をユーザーに見せて確認を得る
    4. 承認されたら実行、修正があれば再計画

    Args:
        steps: JSON配列文字列。各要素:
            {"description": "ファイルメニューをクリック", "x": 100, "y": 30}
            x, y はスクリーンショット画像上の座標（省略可）。
            省略時はパネルにテキストのみ表示。
        monitor: スクリーンショット対象モニター (0=全画面, 1以上=特定モニター)
    """
    import json as _json

    # ステップをパース
    try:
        step_list = _json.loads(steps)
    except Exception as e:
        return _error_image(f"steps の JSON パースに失敗: {e}")

    if not isinstance(step_list, list) or len(step_list) == 0:
        return _error_image("steps は1つ以上のステップを含むJSON配列である必要があります")

    # スクリーンショット取得 → キャンバスに取り込み
    cap = ScreenCapture(monitor_index=monitor, resize_width=1920)
    pil_img = cap.capture_as_pil()
    canvas = _canvas_mgr.from_image("_plan_preview", pil_img)
    img_w, img_h = pil_img.width, pil_img.height

    # 座標付きステップを収集（マーカー + 矢印用）
    coord_steps = []
    for i, step in enumerate(step_list):
        if "x" in step and "y" in step:
            coord_steps.append((i, int(step["x"]), int(step["y"])))

    # 座標付きステップにマーカーを描画
    for i, x, y in coord_steps:
        canvas.draw_numbered_marker(x, y, i + 1, color="blue", size=20)

    # ステップ間に矢印を描画
    for idx in range(len(coord_steps) - 1):
        _, x1, y1 = coord_steps[idx]
        _, x2, y2 = coord_steps[idx + 1]
        canvas.draw_arrow(x1, y1, x2, y2, color=(0, 200, 80), width=3)

    # 右側に説明パネルを描画
    panel_w = 420
    panel_x = img_w - panel_w - 10
    panel_y = 10
    line_h = 28
    panel_h = len(step_list) * line_h + 20

    # 半透明パネル背景（PIL は直接半透明描画できないので、暗い矩形で代用）
    overlay = pil_img.copy()
    from PIL import ImageDraw as _IDraw
    ov_draw = _IDraw.Draw(overlay)
    ov_draw.rectangle(
        [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
        fill=(0, 0, 0),
    )
    # ブレンド（70% 元画像 + 30% 黒パネル → 半透明効果）
    from PIL import Image as _PILImage
    blended = _PILImage.blend(pil_img, overlay, alpha=0.6)
    canvas.image = blended

    # 座標付きマーカーを再描画（ブレンド後に描き直す）
    for i, x, y in coord_steps:
        canvas.draw_numbered_marker(x, y, i + 1, color="blue", size=20)
    for idx in range(len(coord_steps) - 1):
        _, x1, y1 = coord_steps[idx]
        _, x2, y2 = coord_steps[idx + 1]
        canvas.draw_arrow(x1, y1, x2, y2, color=(0, 200, 80), width=3)

    # テキストを描画
    for i, step in enumerate(step_list):
        desc = step.get("description", "")
        label = f" {i+1}. {desc}"
        tx = panel_x + 8
        ty = panel_y + 10 + i * line_h
        canvas.draw_text(tx, ty, label, color="white", size=18)

    # 画像を返す
    result_bytes = canvas.to_bytes("PNG")
    _canvas_mgr.delete("_plan_preview")
    return Image(data=result_bytes, format="png")


def _error_image(message: str) -> Image:
    """エラーメッセージを画像として返す。"""
    from PIL import Image as _PILImage, ImageDraw as _IDraw
    img = _PILImage.new("RGB", (600, 100), (40, 40, 40))
    draw = _IDraw.Draw(img)
    draw.text((10, 10), f"Error: {message}", fill="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


# ---------------------------------------------------------------------------
# ワークフロー（操作の記録・再生）
# ---------------------------------------------------------------------------

# ワークフロー再生時にステップを実行するディスパッチャ
def _execute_workflow_step(tool: str, args: dict) -> str:
    """ワークフローのステップを実行する。各ツール関数を直接呼び出す。
    type_text/keypress の前にフォーカスを明示的に復帰させる。"""
    import time as _wf_time
    # ステップ間のディレイ: フォーカス安定化のため十分な時間を取る
    _wf_time.sleep(0.3)
    dispatch = {
        "click": lambda a: click(a.get("x", 0), a.get("y", 0),
                                 a.get("button", 1), monitor=a.get("monitor", 0)),
        "right_click": lambda a: right_click(a.get("x", 0), a.get("y", 0),
                                             monitor=a.get("monitor", 0)),
        "double_click": lambda a: double_click(a.get("x", 0), a.get("y", 0),
                                               monitor=a.get("monitor", 0)),
        "drag": lambda a: drag(a.get("start_x", 0), a.get("start_y", 0),
                               a.get("end_x", 0), a.get("end_y", 0),
                               a.get("button", 1), a.get("duration", 0.5),
                               monitor=a.get("monitor", 0)),
        "scroll": lambda a: scroll(a.get("x", 0), a.get("y", 0),
                                   a.get("direction", "down"), a.get("amount", 3),
                                   monitor=a.get("monitor", 0)),
        "type_text": lambda a: type_text(a.get("text", ""),
                                         a.get("clear_first", False)),
        "keypress": lambda a: keypress(a.get("key", "")),
        "focus_window": lambda a: focus_window(a.get("keyword", "")),
        "save_file": lambda a: save_file(a.get("filepath", "")),
        "move_to": lambda a: move_to(a.get("x", 0), a.get("y", 0),
                                     monitor=a.get("monitor", 0)),
    }
    fn = dispatch.get(tool)
    if not fn:
        return f"未対応ツール: {tool}"
    return str(fn(args))


@mcp.tool()
def workflow_record(name: str, description: str = "",
                    tags: str = "") -> str:
    """操作の記録を開始する。記録中の操作は全てワークフローに保存される。
    記録を終了するには workflow_stop を呼ぶ。

    Args:
        name: ワークフロー名（一意な識別子）
        description: ワークフローの説明
        tags: カンマ区切りのタグ（例: "clip,保存"）
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    return _workflow_mgr.start_recording(name, description, tag_list)


@mcp.tool()
def workflow_stop() -> str:
    """操作の記録を停止してワークフローを保存する。"""
    return _workflow_mgr.stop_recording()


@mcp.tool()
def workflow_run(name: str) -> str:
    """保存済みワークフローを再生する。記録された操作を順次実行する。

    Args:
        name: 再生するワークフロー名
    """
    # 再生中は記録しない（再帰防止）
    was_recording = _workflow_mgr.is_recording
    if was_recording:
        saved_name = _workflow_mgr.recording_name
    return _workflow_mgr.run(name, _execute_workflow_step)


@mcp.tool()
def workflow_list(tag: str = "") -> str:
    """保存済みワークフローの一覧を表示する。

    Args:
        tag: このタグを持つワークフローのみ表示（省略で全件）
    """
    workflows = _workflow_mgr.list_workflows(tag)
    if not workflows:
        return "保存済みワークフローはありません。"
    lines = [f"ワークフロー一覧 ({len(workflows)}件):"]
    for wf in workflows:
        tags_str = f" [{', '.join(wf['tags'])}]" if wf['tags'] else ""
        lines.append(
            f"  {wf['name']}: {wf['description'] or '(説明なし)'} "
            f"({wf['steps']}ステップ, 実行{wf['run_count']}回){tags_str}")
    return "\n".join(lines)


@mcp.tool()
def workflow_delete(name: str) -> str:
    """保存済みワークフローを削除する。

    Args:
        name: 削除するワークフロー名
    """
    return _workflow_mgr.delete(name)


# ---------------------------------------------------------------------------
# Zoom-and-Refine（再帰的サブグリッド分割による高精度座標特定）
# ---------------------------------------------------------------------------
# スクリーンショット座標 → zoom_and_refine の状態管理
_zoom_state: dict = {}  # 現在のズーム状態を保持


@mcp.tool()
@_preserves_screenshot_state
def zoom_and_refine(monitor: int = 0, cell: str = "",
                    width: int = 1920, grid_size: int = 3,
                    region: str = "",
                    target_px: int = 0,
                    target_x: int = -1, target_y: int = -1) -> Image:
    """画面をグリッドで分割し、段階的にズームして正確な座標を特定する。

    UI要素の正確な位置を知りたいとき、以下の手順で使う:
      1. zoom_and_refine(monitor=3) → 全画面をグリッドに分割した画像が返る
      2. 目的の要素があるセルを特定（例: "B2"）
      3. zoom_and_refine(cell="B2") → そのセルを拡大して更にグリッドに分割
      4. 必要なら更に zoom_and_refine(cell="A1") → 更に拡大
      5. zoom_and_refine(cell="click") → カーソルを移動し peek 画像で視認確認
      6. 確認OK → click(x=..., y=..., monitor=...) で実行

    カラーホイール等の位置が既知の場合、region で直接ズーム:
      zoom_and_refine(monitor=3, region="100,200,300,300") → 指定領域からスタート

    **自動深度調整** — target_px 指定で、セルが十分小さくなるまで自動ズーム:
      zoom_and_refine(monitor=3, target_px=50)
        → 1セルの物理サイズが 50px 以下になるまで自動でズーム（最大 depth=6）
        → 最終グリッド画像を返す。セルを選んで cell="click" するだけ

    **座標指定の自動ズーム** — watcher_find で得た物理座標に自動でズーム:
      zoom_and_refine(monitor=3, target_x=1837, target_y=500, target_px=30)
        → 物理座標を自動でスクリーンショット座標に変換 → 自動ズーム → peek 確認画像

    Args:
        monitor: 最初のキャプチャ対象モニター番号 (1, 2, 3...)。
                 0 を指定するとリセットせずに前回の状態を継続。
        cell: グリッドセル名。"A1"~"C3" (3×3時), "A1"~"D4" (4×4時), "A1"~"E5" (5×5時)。
              "click" を指定するとカーソルを移動し peek 画像で視認確認。
              空文字で初期キャプチャ。
        width: 初期スクリーンショットのリサイズ幅
        grid_size: グリッド分割数 (3~5)。大きいほど1回のステップで精度が上がる。
                   デフォルト3。カラーホイール等の細かい選択には4や5を推奨。
        region: 初期ズーム領域 "x,y,w,h"（スクリーンショット座標）。
                指定すると全画面表示をスキップしてこの領域から開始。
        target_px: 自動深度調整の目標セルサイズ（物理ピクセル）。
                   0=手動（従来動作）。30~100 推奨。指定すると 1セルの物理幅が
                   この値以下になるまで自動でズームする（最大 depth=6）。
        target_x: 自動ズームのターゲット X 座標（物理ピクセル）。
                  watcher_find の center_x をそのまま渡せる。
                  内部でスクリーンショット座標に自動変換される。-1 で無効。
        target_y: 自動ズームのターゲット Y 座標（物理ピクセル）。
                  watcher_find の center_y をそのまま渡せる。
    """
    from PIL import ImageDraw, ImageFont

    # grid_size のバリデーション
    grid_size = max(3, min(5, grid_size))

    # --- 初期化（monitor 指定時 or 初回） ---
    if monitor > 0 or not _zoom_state:
        if monitor <= 0:
            return Image(data=b"", format="png")  # state なしエラー回避
        # スクリーンショット撮影
        cap = ScreenCapture(monitor_index=monitor, resize_width=width)
        pil_img = cap.capture_as_pil()
        _last_screenshot_info[monitor] = (pil_img.width, pil_img.height)

        _zoom_state.clear()
        _zoom_state["monitor"] = monitor
        _zoom_state["image"] = pil_img
        _zoom_state["depth"] = 0
        _zoom_state["history"] = []
        _zoom_state["grid_size"] = grid_size

        # region 指定時は直接その領域からスタート
        if region:
            try:
                parts = [int(v.strip()) for v in region.split(",")]
                if len(parts) == 4:
                    rx, ry, rw, rh = parts
                    # クランプ
                    rx = max(0, min(rx, pil_img.width - 1))
                    ry = max(0, min(ry, pil_img.height - 1))
                    rw = max(1, min(rw, pil_img.width - rx))
                    rh = max(1, min(rh, pil_img.height - ry))
                    _zoom_state["region"] = (rx, ry, rw, rh)
            except ValueError:
                _zoom_state["region"] = (0, 0, pil_img.width, pil_img.height)
        else:
            _zoom_state["region"] = (0, 0, pil_img.width, pil_img.height)

        # --- 自動深度調整（target_px 指定時） ---
        if target_px > 0:
            # 物理座標 → スクリーンショット座標に変換
            if target_x >= 0 and target_y >= 0:
                target_x, target_y = _physical_to_screen(
                    monitor, target_x, target_y)
            return _zoom_auto_depth(target_px, target_x, target_y)

        cropped = pil_img.crop(_zoom_state["region"])
        return _zoom_draw_grid(cropped, 0)

    # --- click: カーソルを移動 → peek で視認確認画像を返す ---
    if cell.lower() == "click":
        rx, ry, rw, rh = _zoom_state["region"]
        center_x = rx + rw // 2
        center_y = ry + rh // 2
        mon = _zoom_state["monitor"]
        # スクリーンショット座標 → 物理座標
        phys_x, phys_y = _screen_to_physical(mon, center_x, center_y)
        depth = _zoom_state["depth"]
        history = " → ".join(_zoom_state["history"])
        _zoom_state.clear()

        # カーソルを移動して peek で視認確認
        import pyautogui
        _executor.move_to(phys_x, phys_y)
        verify_msg = _verify_cursor_position(phys_x, phys_y)

        # peek: カーソル周辺を拡大キャプチャ
        half = 100
        left = max(0, phys_x - half)
        top_y = max(0, phys_y - half)
        cap = ScreenCapture(region=(left, top_y, 200, 200))
        peek_img = cap.capture_as_pil()
        enlarged = peek_img.resize((400, 400), resample=0)  # NEAREST for sharp
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(enlarged)
        # 十字線（カーソル位置）
        center = 200
        draw.line([(center - 20, center), (center + 20, center)], fill="red", width=2)
        draw.line([(center, center - 20), (center, center + 20)], fill="red", width=2)
        # 情報テキスト
        info_lines = [
            f"depth={depth}  path: {history}",
            f"screenshot: ({center_x}, {center_y})  physical: ({phys_x}, {phys_y})",
            f"→ click(x={center_x}, y={center_y}, monitor={mon})",
        ]
        if verify_msg:
            info_lines.append(verify_msg)
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
        # 下部に情報バー
        bar_h = 16 * len(info_lines) + 4
        draw.rectangle([(0, 400 - bar_h), (400, 400)], fill=(0, 0, 0, 220))
        for i, line in enumerate(info_lines):
            draw.text((4, 400 - bar_h + 2 + i * 16), line, fill="white", font=font)

        buf = BytesIO()
        enlarged.save(buf, format="PNG")
        return Image(data=buf.getvalue(), format="png")

    # --- セル選択: 領域を絞り込み ---
    cell = cell.upper().strip()
    gs = _zoom_state.get("grid_size", 3)
    # grid_size 変更が指定された場合は更新
    if grid_size != 3 or "grid_size" not in _zoom_state:
        _zoom_state["grid_size"] = grid_size
        gs = grid_size

    valid_cols = "ABCDE"[:gs]
    valid_rows = "12345"[:gs]
    if len(cell) < 2 or cell[0] not in valid_cols or cell[1] not in valid_rows:
        return _zoom_draw_grid(
            _zoom_state["image"].crop(_zoom_state["region"]),
            _zoom_state["depth"],
            error=f"無効なセル名: '{cell}'。{valid_cols[0]}1~{valid_cols[-1]}{valid_rows[-1]} で指定してください。"
        )

    col = ord(cell[0]) - ord("A")
    row = int(cell[1]) - 1

    rx, ry, rw, rh = _zoom_state["region"]
    cell_w = rw // gs
    cell_h = rh // gs
    new_x = rx + col * cell_w
    new_y = ry + row * cell_h

    _zoom_state["region"] = (new_x, new_y, cell_w, cell_h)
    _zoom_state["depth"] += 1
    _zoom_state["history"].append(cell)

    # 領域を切り出し
    cropped = _zoom_state["image"].crop((new_x, new_y, new_x + cell_w, new_y + cell_h))

    return _zoom_draw_grid(cropped, _zoom_state["depth"])


def _zoom_calc_phys_cell(gs: int) -> tuple[int, int]:
    """現在の zoom_state から 1セルの物理ピクセルサイズを計算する。"""
    mon_idx = _zoom_state.get("monitor", 0)
    if mon_idx <= 0:
        return (9999, 9999)
    try:
        import mss
        with mss.mss() as sct:
            if mon_idx >= len(sct.monitors):
                return (9999, 9999)
            mon = sct.monitors[mon_idx]
            mon_w, mon_h = mon['width'], mon['height']
        shot_w, shot_h = _last_screenshot_info.get(mon_idx, (mon_w, mon_h))
        region = _zoom_state.get("region", (0, 0, shot_w, shot_h))
        cell_w_shot = region[2] // gs
        cell_h_shot = region[3] // gs
        phys_w = int(cell_w_shot * mon_w / shot_w)
        phys_h = int(cell_h_shot * mon_h / shot_h)
        return (phys_w, phys_h)
    except Exception:
        return (9999, 9999)


def _zoom_auto_depth(target_px: int, target_x: int = -1, target_y: int = -1) -> Image:
    """target_px 以下になるまで自動でズームし続ける。

    target_x, target_y が指定されている場合:
      その座標を含むセルを自動選択 → 最終的に peek 確認画像を返す。
    指定されていない場合:
      セルサイズが target_px 以下になるまでズーム → 最終グリッド画像を返す。
    """
    MAX_AUTO_DEPTH = 6
    gs = _zoom_state.get("grid_size", 3)
    has_target = target_x >= 0 and target_y >= 0

    # 自動ズームループ
    while _zoom_state["depth"] < MAX_AUTO_DEPTH:
        phys_w, phys_h = _zoom_calc_phys_cell(gs)
        max_phys = max(phys_w, phys_h)

        if max_phys <= target_px:
            break  # 十分小さくなった

        if not has_target:
            # ターゲット座標なし → ズームの指針がないのでここで返す
            # （「あと何段必要か」の情報をグリッド画像に含めて返す）
            break

        # ターゲット座標を含むセルを特定
        rx, ry, rw, rh = _zoom_state["region"]
        cell_w = rw // gs
        cell_h = rh // gs

        if cell_w <= 0 or cell_h <= 0:
            break

        col = min((target_x - rx) // cell_w, gs - 1)
        row = min((target_y - ry) // cell_h, gs - 1)
        col = max(0, col)
        row = max(0, row)

        # セル選択
        cell_name = f"{'ABCDE'[col]}{row + 1}"
        new_x = rx + col * cell_w
        new_y = ry + row * cell_h

        _zoom_state["region"] = (new_x, new_y, cell_w, cell_h)
        _zoom_state["depth"] += 1
        _zoom_state["history"].append(cell_name)

    # 結果を返す
    if has_target:
        # ターゲット座標あり → peek 確認画像（cell="click" と同等）
        rx, ry, rw, rh = _zoom_state["region"]
        center_x = rx + rw // 2
        center_y = ry + rh // 2
        mon = _zoom_state["monitor"]
        phys_x, phys_y = _screen_to_physical(mon, center_x, center_y)
        depth = _zoom_state["depth"]
        history = " → ".join(_zoom_state["history"])
        phys_w, phys_h = _zoom_calc_phys_cell(gs)
        _zoom_state.clear()

        # カーソルを移動して peek で視認確認
        _executor.move_to(phys_x, phys_y)
        verify_msg = _verify_cursor_position(phys_x, phys_y)

        # peek: カーソル周辺を拡大キャプチャ
        half = 100
        left = max(0, phys_x - half)
        top_y = max(0, phys_y - half)
        cap = ScreenCapture(region=(left, top_y, 200, 200))
        peek_img = cap.capture_as_pil()
        enlarged = peek_img.resize((400, 400), resample=0)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(enlarged)
        center = 200
        draw.line([(center - 20, center), (center + 20, center)], fill="red", width=2)
        draw.line([(center, center - 20), (center, center + 20)], fill="red", width=2)
        info_lines = [
            f"AUTO depth={depth}  cell={phys_w}×{phys_h}px  path: {history}",
            f"screenshot: ({center_x}, {center_y})  physical: ({phys_x}, {phys_y})",
            f"→ click(x={center_x}, y={center_y}, monitor={mon})",
        ]
        if verify_msg:
            info_lines.append(verify_msg)
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
        bar_h = 16 * len(info_lines) + 4
        draw.rectangle([(0, 400 - bar_h), (400, 400)], fill=(0, 0, 0, 220))
        for i, line in enumerate(info_lines):
            draw.text((4, 400 - bar_h + 2 + i * 16), line, fill="white", font=font)

        buf = BytesIO()
        enlarged.save(buf, format="PNG")
        return Image(data=buf.getvalue(), format="png")
    else:
        # ターゲットなし → 推奨 depth 情報付きグリッド画像
        rx, ry, rw, rh = _zoom_state["region"]
        pil_img = _zoom_state["image"]
        cropped = pil_img.crop((rx, ry, rx + rw, ry + rh))
        return _zoom_draw_grid(cropped, _zoom_state["depth"])


def _zoom_draw_grid(pil_img, depth: int, error: str = "") -> Image:
    """画像にグリッドを描画して返す。grid_size に応じた N×N グリッド。"""
    from PIL import ImageDraw, ImageFont

    gs = _zoom_state.get("grid_size", 3)

    # 見やすいサイズに拡大（最低 600px 幅）
    min_display = 600
    if pil_img.width < min_display:
        scale = min_display / pil_img.width
        new_w = int(pil_img.width * scale)
        new_h = int(pil_img.height * scale)
        img = pil_img.resize((new_w, new_h), resample=1)  # BILINEAR
    else:
        img = pil_img.copy()

    draw = ImageDraw.Draw(img)
    w, h = img.size
    cell_w = w // gs
    cell_h = h // gs

    # グリッド線（緑）
    for i in range(1, gs):
        draw.line([(cell_w * i, 0), (cell_w * i, h)], fill="#00FF00", width=2)
        draw.line([(0, cell_h * i), (w, cell_h * i)], fill="#00FF00", width=2)

    # 外枠
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline="#00FF00", width=2)

    # セルラベル (A1~E5)
    try:
        font = ImageFont.truetype("arial.ttf", max(12, min(cell_w, cell_h) // 6))
    except Exception:
        font = ImageFont.load_default()

    cols = "ABCDE"[:gs]
    for r in range(gs):
        for c in range(gs):
            label = f"{cols[c]}{r+1}"
            lx = c * cell_w + 4
            ly = r * cell_h + 2
            # 背景付きで見やすく
            draw.rectangle(
                [(lx, ly), (lx + len(label) * max(10, cell_w // 10), ly + max(16, cell_h // 8))],
                fill=(0, 0, 0, 180)
            )
            draw.text((lx + 2, ly + 1), label, fill="#00FF00", font=font)

    # --- 物理セルサイズを計算 ---
    phys_cell_info = ""
    mon_idx = _zoom_state.get("monitor", 0)
    if mon_idx > 0:
        try:
            import mss
            with mss.mss() as sct:
                if mon_idx < len(sct.monitors):
                    mon = sct.monitors[mon_idx]
                    mon_w, mon_h = mon['width'], mon['height']
                    if mon_idx in _last_screenshot_info:
                        shot_w, shot_h = _last_screenshot_info[mon_idx]
                    else:
                        shot_w, shot_h = mon_w, mon_h
                    # 現在の region のセルサイズ（スクリーンショット座標）
                    region = _zoom_state.get("region", (0, 0, shot_w, shot_h))
                    region_cell_w = region[2] // gs
                    region_cell_h = region[3] // gs
                    # 物理ピクセルに変換
                    phys_w = int(region_cell_w * mon_w / shot_w)
                    phys_h = int(region_cell_h * mon_h / shot_h)
                    phys_cell_info = f"  cell={phys_w}×{phys_h}px"
        except Exception:
            pass

    # 推奨残りステップ数を計算（target_px=50 基準）
    depth_hint = ""
    if phys_cell_info:
        try:
            phys_max = max(phys_w, phys_h)
            if phys_max > 50:
                import math
                remaining = math.ceil(math.log(phys_max / 50) / math.log(gs))
                depth_hint = f"  →あと{remaining}段で~50px"
        except Exception:
            pass

    # 深度表示 + 物理セルサイズ
    info = f"depth={depth}  {gs}×{gs} grid{phys_cell_info}{depth_hint}"
    if _zoom_state.get("history"):
        info += f"  path: {' → '.join(_zoom_state['history'])}"
    if error:
        info += f"  ⚠ {error}"
    draw.rectangle([(0, h - 22), (w, h)], fill=(0, 0, 0, 200))
    try:
        info_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        info_font = ImageFont.load_default()
    draw.text((4, h - 20), info, fill="white", font=info_font)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


# ---------------------------------------------------------------------------
# ScreenWatcher ツール（UI Automation + OpenCV 統合画面監視）
# ---------------------------------------------------------------------------
# 「潜水艦のソナー担当」— バックグラウンドで画面を常時監視し、
# スクリーンショット+LLM解析なしでUI状態をリアルタイムに把握する。
#
# 典型的な使い方:
#   1. watcher_start("メモ帳")  — 監視開始
#   2. watcher_find("保存")     — ボタン座標を取得（0.01ms）
#   3. click(x, y)              — クリック
#   4. watcher_wait_change()    — 変化を待つ
#   5. watcher_report()         — 現在の状態を確認
#   6. watcher_stop()           — 監視終了


@mcp.tool()
def watcher_start(
    window: str,
    ui_interval: float = 0.1,
    cv_interval: float = 0.02,
    max_depth: int = 4,
) -> str:
    """ScreenWatcher を開始する。指定ウィンドウの UI 要素と視覚変化をバックグラウンドで常時監視する。

    スクリーンショット+LLM 解析（2000-5000ms + API課金）に対し、
    UI 要素の座標取得は 0.01ms、テキスト値の取得は 0.001ms で返る。

    Args:
        window: 監視するウィンドウタイトルのキーワード（部分一致）
        ui_interval: UI Automation のスキャン間隔（秒）。デフォルト 0.1s
        cv_interval: OpenCV のスキャン間隔（秒）。デフォルト 0.02s
        max_depth: UI ツリーの最大探索深度（深いほど詳細だが遅い）
    """
    global _screen_watcher
    if _screen_watcher and _screen_watcher.is_running():
        _screen_watcher.stop()

    try:
        _screen_watcher = ScreenWatcher(
            window_keyword=window,
            ui_interval=ui_interval,
            cv_interval=cv_interval,
            max_depth=max_depth,
        )
        _screen_watcher.start()
        # 最初のスキャン完了を待つ
        import time as _t
        _t.sleep(0.3)
        _log_action("watcher_start", target=window,
                     workflow_args={"window": window})
        # 初回レポートを JSON で返す
        import json as _json
        with _screen_watcher._lock:
            visible = [e for e in _screen_watcher._ui_elements if e.is_visible]
        buttons = [e for e in visible if e.control_type == "Button"]
        values = [e for e in visible if e.value]
        result = {
            "success": True,
            "window": window,
            "element_count": len(visible),
            "buttons": len(buttons),
            "with_values": len(values),
            "ui_avg_ms": round(_screen_watcher.stats().get("ui_avg_ms", 0), 1),
            "cv_avg_ms": round(_screen_watcher.stats().get("cv_avg_ms", 0), 1),
        }
        return _json.dumps(result, ensure_ascii=False)
    except RuntimeError as e:
        _screen_watcher = None
        import json as _json
        return _json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    except Exception as e:
        _screen_watcher = None
        import json as _json
        return _json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@mcp.tool()
def watcher_stop() -> str:
    """ScreenWatcher を停止する。"""
    global _screen_watcher
    import json as _json
    if not _screen_watcher or not _screen_watcher.is_running():
        return _json.dumps({"success": False, "error": "ScreenWatcher is not running"})
    stats = _screen_watcher.stats()
    _screen_watcher.stop()
    _screen_watcher = None
    _log_action("watcher_stop")
    result = {
        "success": True,
        "stats": {
            "ui_scans": stats["ui_scans"],
            "ui_avg_ms": round(stats["ui_avg_ms"], 1),
            "cv_scans": stats["cv_scans"],
            "cv_avg_ms": round(stats["cv_avg_ms"], 1),
            "ui_changes": stats["ui_changes"],
            "cv_changes": stats["cv_changes"],
        },
    }
    return _json.dumps(result, ensure_ascii=False)


@mcp.tool()
def watcher_report() -> str:
    """ScreenWatcher のソナー報告。現在の UI 状態 + 直近の変化を返す。

    監視中でなければエラーメッセージを返す。
    """
    if not _screen_watcher or not _screen_watcher.is_running():
        return "ScreenWatcher は動作していません。watcher_start() で開始してください"
    return _screen_watcher.report()


@mcp.tool()
def watcher_find(
    name: str = "",
    control_type: str = "",
) -> str:
    """UI 要素を名前・型で検索し、座標を返す（部分一致）。

    ボタンのクリック座標を事前に取得するのに最適。
    スクリーンショット+LLM解析の代わりに 0.01ms で座標が手に入る。

    Args:
        name: 要素名（部分一致）。例: "保存", "ファイル", "OK"
        control_type: コントロール型。例: "Button", "Edit", "MenuItem", "Text"
    """
    import json as _json
    if not _screen_watcher or not _screen_watcher.is_running():
        return _json.dumps({"elements": [], "count": 0, "error": "ScreenWatcher is not running"})

    elements = _screen_watcher.find_elements(name=name, control_type=control_type)

    # ウィンドウが属するモニター番号を取得
    monitor_num = 0
    for mon_idx, (wl, wt, ww, wh) in _last_window_capture.items():
        monitor_num = mon_idx
        break

    el_list = []
    for el in elements[:20]:
        entry = {
            "control_type": el.control_type,
            "name": el.name,
            "center": {"x": el.center_x, "y": el.center_y, "monitor": monitor_num},
            "size": {"width": el.width, "height": el.height},
        }
        if el.value:
            entry["value"] = el.value[:100]
        el_list.append(entry)

    # 最初の要素の座標を検証済みとして記録
    if elements:
        _record_verified_coord(elements[0].center_x, elements[0].center_y, "watcher_find")

    result = {"elements": el_list, "count": len(elements)}
    if len(elements) > 20:
        result["truncated"] = len(elements) - 20
    return _json.dumps(result, ensure_ascii=False)


@mcp.tool()
def watcher_list_elements(
    control_type: str = "",
    with_values_only: bool = False,
) -> str:
    """監視中ウィンドウの全 UI 要素を一覧表示する。

    Args:
        control_type: フィルタするコントロール型（空欄で全て）
        with_values_only: True にすると値を持つ要素のみ表示
    """
    import json as _json
    if not _screen_watcher or not _screen_watcher.is_running():
        return _json.dumps({"elements": [], "count": 0, "error": "ScreenWatcher is not running"})

    elements = _screen_watcher.find_elements(control_type=control_type)
    if with_values_only:
        elements = [e for e in elements if e.value]

    # ウィンドウが属するモニター番号を取得
    monitor_num = 0
    for mon_idx, (wl, wt, ww, wh) in _last_window_capture.items():
        monitor_num = mon_idx
        break

    el_list = []
    for el in elements[:50]:
        entry = {
            "control_type": el.control_type,
            "name": el.name,
            "center": {"x": el.center_x, "y": el.center_y, "monitor": monitor_num},
            "size": {"width": el.width, "height": el.height},
        }
        if el.value:
            entry["value"] = el.value[:100]
        el_list.append(entry)

    result = {"elements": el_list, "count": len(elements)}
    if len(elements) > 50:
        result["truncated"] = len(elements) - 50
    return _json.dumps(result, ensure_ascii=False)


@mcp.tool()
def watcher_get_value(element_name: str) -> str:
    """指定名の UI 要素の値を取得する（0.001ms）。

    テキストボックスの入力内容、ステータスバーの表示値などを
    スクリーンショットなしで即座に取得できる。

    Args:
        element_name: 要素名（部分一致）
    """
    import json as _json
    if not _screen_watcher or not _screen_watcher.is_running():
        return _json.dumps({"success": False, "action": "watcher_get_value",
                            "error": "ScreenWatcher は動作していません。watcher_start() で開始してください"},
                           ensure_ascii=False)
    val = _screen_watcher.get_value(element_name)
    if val is None:
        return _json.dumps({"success": False, "action": "watcher_get_value",
                            "error": f"要素 '{element_name}' が見つからないか、値がありません"},
                           ensure_ascii=False)
    return _json.dumps({"success": True, "action": "watcher_get_value",
                        "details": {"element_name": element_name, "value": val}},
                       ensure_ascii=False)


@mcp.tool()
def watcher_wait_value(
    element_name: str,
    expected: str,
    timeout: float = 5.0,
    contains: bool = True,
) -> str:
    """UI 要素の値が期待値になるまで待機する。

    操作後の結果確認に使う。例: type_text 後にテキストが反映されたか確認。

    Args:
        element_name: 要素名（部分一致）
        expected: 期待する値（contains=True なら部分一致）
        timeout: タイムアウト（秒）
        contains: True=部分一致, False=完全一致
    """
    import json as _json
    if not _screen_watcher or not _screen_watcher.is_running():
        return _json.dumps({"success": False, "action": "watcher_wait_value",
                            "error": "ScreenWatcher は動作していません。watcher_start() で開始してください"},
                           ensure_ascii=False)
    matched = _screen_watcher.wait_for_value(
        element_name, expected, timeout=timeout, contains=contains)
    val = _screen_watcher.get_value(element_name)
    if matched:
        return _json.dumps({"success": True, "action": "watcher_wait_value",
                            "details": {"element_name": element_name, "expected": expected,
                                        "matched": True, "current_value": val}},
                           ensure_ascii=False)
    return _json.dumps({"success": False, "action": "watcher_wait_value",
                        "error": f"タイムアウト ({timeout}s)",
                        "details": {"element_name": element_name, "expected": expected,
                                    "matched": False, "current_value": val,
                                    "timeout": timeout}},
                       ensure_ascii=False)


@mcp.tool()
def watcher_wait_change(timeout: float = 5.0) -> str:
    """何らかの UI 変化または視覚変化が起きるまで待機する。

    操作後に「画面が変わった」ことを確認するのに使う。
    スクリーンショット再撮影+LLM比較の代わりに使える。

    Args:
        timeout: タイムアウト（秒）
    """
    import json as _json
    if not _screen_watcher or not _screen_watcher.is_running():
        return _json.dumps({"success": False, "action": "watcher_wait_change",
                            "error": "ScreenWatcher は動作していません。watcher_start() で開始してください"},
                           ensure_ascii=False)
    changes = _screen_watcher.wait_for_change(timeout=timeout)
    if not changes:
        return _json.dumps({"success": False, "action": "watcher_wait_change",
                            "error": f"タイムアウト ({timeout}s): 変化なし",
                            "details": {"timeout": timeout, "change_count": 0, "changes": []}},
                           ensure_ascii=False)
    change_list = []
    for c in changes[-15:]:
        entry = {"type": str(type(c).__name__), "description": str(c)}
        if hasattr(c, "change_type"):
            entry["change_type"] = c.change_type
            entry["element_name"] = c.element_name
            entry["control_type"] = c.control_type
            if c.old_value is not None:
                entry["old_value"] = c.old_value
            if c.new_value is not None:
                entry["new_value"] = c.new_value
        if hasattr(c, "region_name"):
            entry["region_name"] = c.region_name
            entry["score"] = round(c.score, 4)
        change_list.append(entry)
    return _json.dumps({"success": True, "action": "watcher_wait_change",
                        "details": {"change_count": len(changes), "changes": change_list}},
                       ensure_ascii=False)


@mcp.tool()
def watcher_color(x: int = -1, y: int = -1) -> str:
    """指定座標（またはカーソル位置）の色情報を取得する（16ms）。

    カラーピッカーの代わりに使える。RGB, HSV, 色名を返す。

    Args:
        x: X座標（物理ピクセル）。-1 でカーソル位置
        y: Y座標（物理ピクセル）。-1 でカーソル位置
    """
    # 色取得はスレッド不要なので、ScreenWatcher がなくても動作する
    from agent.screen_watcher import _capture_region, _get_cursor_pos, _color_name
    import cv2 as _cv2

    if x < 0 or y < 0:
        cx, cy = _get_cursor_pos()
    else:
        cx, cy = x, y

    img = _capture_region(cx, cy, 1, 1)
    b, g, r = img[0, 0]
    hsv = _cv2.cvtColor(img.reshape(1, 1, 3), _cv2.COLOR_BGR2HSV)
    h, s, v = hsv[0, 0]
    color = {
        "pos": (cx, cy),
        "rgb": (int(r), int(g), int(b)),
        "hsv": (int(h), int(s), int(v)),
        "name": _color_name(int(h), int(s), int(v)),
    }

    return (f"位置: ({color['pos'][0]}, {color['pos'][1]})\n"
            f"RGB: {color['rgb']}\n"
            f"HSV: {color['hsv']}\n"
            f"色名: {color['name']}")


@mcp.tool()
def watcher_watch_region(
    name: str,
    x: int,
    y: int,
    w: int,
    h: int,
    threshold: float = 0.01,
) -> str:
    """OpenCV による視覚変化の監視領域を追加する。

    追加した領域は CV スレッドが常時監視し、閾値を超える変化があれば
    watcher_report() や watcher_wait_change() で報告される。

    Args:
        name: 領域名（任意の文字列、後で参照用）
        x: 左上X座標（物理ピクセル）
        y: 左上Y座標（物理ピクセル）
        w: 幅
        h: 高さ
        threshold: 変化判定の閾値（0-1、小さいほど敏感）
    """
    if not _screen_watcher or not _screen_watcher.is_running():
        return "ScreenWatcher は動作していません。watcher_start() で開始してください"
    _screen_watcher.watch_region(name, x, y, w, h, threshold=threshold)
    return f"監視領域 '{name}' を追加: ({x},{y}) {w}x{h} 閾値={threshold}"


@mcp.tool()
def watcher_unwatch_region(name: str) -> str:
    """監視領域を削除する。

    Args:
        name: 削除する領域名
    """
    if not _screen_watcher or not _screen_watcher.is_running():
        return "ScreenWatcher は動作していません"
    _screen_watcher.unwatch_region(name)
    return f"監視領域 '{name}' を削除"


@mcp.tool()
def watcher_screenshot(monitor: int = 0, width: int = 0,
                       region_x: int = 0, region_y: int = 0,
                       region_w: int = 0, region_h: int = 0,
                       window: str = "") -> Image:
    """ScreenWatcher の CV パイプライン内でスクリーンショットを撮影する（潜水艦モデル準拠）。

    通常の screenshot と異なり:
    - 潜水艦モデルの一部として動作（Hook でブロックされない）
    - 撮影後、全 watch region の基準画像を自動更新（変化検出リセット）
    - AI が「今確認したい」と要請したタイミングでのみ使う（定期撮影ではない）

    UIA 非対応アプリ（CSP、Blender、DAZ Studio 等）での視覚確認に使用。

    window を指定するとそのウィンドウだけをキャプチャする（推奨）。
    ウィンドウがモニター全体を占めていない場合でも正確な座標変換が可能になる。

    width=0（デフォルト）の場合、物理解像度に応じて自動計算される:
    - 2560px以下: そのまま（リサイズなし）
    - 2561px以上: 物理幅の半分（最低1920px）
    これにより 4K/6K/8K モニターでもスケール比が最大2倍に抑えられる。

    region_x/y/w/h を指定すると、直前のスクリーンショット座標系で領域を切り出して
    拡大表示する（4Kモニターのダイアログボタン等の詳細確認に便利）。
    領域指定時は座標変換用の _last_screenshot_info を更新しない。

    Args:
        monitor: モニター番号 (0=全画面, 1=プライマリ, 2,3...=その他)
        width: リサイズ幅 (0=自動計算, スケール比が2倍を超えないよう調整)
        region_x: 切り出し領域の左端X（スクリーンショット座標、0=領域指定なし）
        region_y: 切り出し領域の上端Y（スクリーンショット座標）
        region_w: 切り出し領域の幅（0=領域指定なし）
        region_h: 切り出し領域の高さ（0=領域指定なし）
        window: ウィンドウタイトルのキーワード。指定するとそのウィンドウだけをキャプチャする。
    """
    if not _screen_watcher or not _screen_watcher.is_running():
        return "ScreenWatcher は動作していません。watcher_start() で開始してください"

    import cv2 as _cv2
    from PIL import Image as PILImage

    # ウィンドウキャプチャモード
    if window:
        return _watcher_screenshot_window(window, width, region_x, region_y, region_w, region_h)

    # リサイズ幅の自動計算
    if width == 0:
        width = _auto_resize_width(monitor)

    frame_bgr = _screen_watcher.screenshot(monitor=monitor, resize_width=width)

    # BGR → RGB → PIL → PNG
    frame_rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(frame_rgb)

    # 領域指定がある場合: クロップして返す（座標変換情報は更新しない）
    if region_w > 0 and region_h > 0:
        rx = max(0, min(region_x, pil_img.width - 1))
        ry = max(0, min(region_y, pil_img.height - 1))
        rr = min(rx + region_w, pil_img.width)
        rb = min(ry + region_h, pil_img.height)
        pil_img = pil_img.crop((rx, ry, rr, rb))
    else:
        # 全体キャプチャ: スクリーンショットのサイズを記録（座標変換用）
        if monitor > 0:
            _last_screenshot_info[monitor] = (pil_img.width, pil_img.height)
            _last_window_capture.pop(monitor, None)

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


def _auto_resize_width(monitor: int) -> int:
    """モニター物理解像度に応じたリサイズ幅を自動計算する。
    スケール比が2倍を超えないよう調整。2560px以下はリサイズなし。
    """
    import mss
    try:
        with mss.mss() as sct:
            if 0 < monitor < len(sct.monitors):
                phys_w = sct.monitors[monitor]['width']
            else:
                phys_w = sct.monitors[0]['width']
    except Exception:
        return 1920
    if phys_w <= 2560:
        return phys_w  # リサイズ不要
    return max(1920, phys_w // 2)


def _watcher_screenshot_window(keyword: str, width: int,
                                region_x: int, region_y: int,
                                region_w: int, region_h: int) -> Image:
    """ScreenWatcher 経由のウィンドウキャプチャ。"""
    import ctypes
    import ctypes.wintypes
    import mss
    import cv2 as _cv2
    from PIL import Image as PILImage

    user32 = ctypes.windll.user32

    # ウィンドウ検索
    target_hwnd = None

    def enum_cb(hwnd, _):
        nonlocal target_hwnd
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if keyword.lower() in buf.value.lower():
            target_hwnd = hwnd
            return False
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

    if not target_hwnd:
        return f"エラー: '{keyword}' を含むウィンドウが見つかりません"

    # ウィンドウの物理座標
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(target_hwnd, ctypes.byref(rect))
    win_left = rect.left
    win_top = rect.top
    win_w = rect.right - rect.left
    win_h = rect.bottom - rect.top

    # 最大化ウィンドウの負オフセット補正
    if win_left < 0:
        win_w += win_left
        win_left = 0
    if win_top < 0:
        win_h += win_top
        win_top = 0

    # 所属モニター検出
    with mss.mss() as sct:
        target_monitor = 0
        for i in range(1, len(sct.monitors)):
            mon = sct.monitors[i]
            cx = win_left + win_w // 2
            cy = win_top + win_h // 2
            if (mon['left'] <= cx < mon['left'] + mon['width'] and
                    mon['top'] <= cy < mon['top'] + mon['height']):
                target_monitor = i
                break

    # リサイズ幅の自動計算
    if width == 0:
        if win_w <= 2560:
            width = win_w
        else:
            width = max(1920, win_w // 2)

    # ウィンドウ領域をキャプチャ
    cap = ScreenCapture(region=(win_left, win_top, win_w, win_h), resize_width=width)
    pil_img = cap.capture_as_pil()

    # watch region の基準画像を更新
    if _screen_watcher and _screen_watcher.is_running():
        from agent.screen_watcher import _capture_region
        with _screen_watcher._lock:
            for region in _screen_watcher._watch_regions.values():
                region.prev_frame = _capture_region(region.x, region.y, region.w, region.h)

    # 領域指定がある場合: クロップ
    if region_w > 0 and region_h > 0:
        rx = max(0, min(region_x, pil_img.width - 1))
        ry = max(0, min(region_y, pil_img.height - 1))
        rr = min(rx + region_w, pil_img.width)
        rb = min(ry + region_h, pil_img.height)
        pil_img = pil_img.crop((rx, ry, rr, rb))
    else:
        # 座標変換用の情報を記録
        if target_monitor > 0:
            _last_screenshot_info[target_monitor] = (pil_img.width, pil_img.height)
            _last_window_capture[target_monitor] = (win_left, win_top, win_w, win_h)

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


# ---------------------------------------------------------------------------
# フォーカスフリー操作（カーソル・フォーカスを動かさずにアプリを操作）
# ---------------------------------------------------------------------------
# ユーザーがPCを使っている最中でも、AIが裏でアプリを操作できる。
# 3層フォールバック: UI Automation → PostMessage → SendInput


@mcp.tool()
def ff_click(
    window: str,
    element: str,
    control_type: str = "",
) -> str:
    """フォーカスもカーソルも動かさずに UI 要素をクリックする。

    ユーザーが操作中でもメニューが閉じたりカーソルが飛んだりしない。
    3層フォールバック:
      1. UI Automation Invoke（最も確実、フォーカス不要）
      2. PostMessage（多くのWin32アプリで動作、フォーカス不要）
      3. SendInput（最終手段、カーソルが動くが元の位置に戻す）

    Args:
        window: ウィンドウタイトルのキーワード（部分一致）
        element: クリックする要素の名前（部分一致）
        control_type: フィルタ用のコントロール型（空で全て）
    """
    result = _focusfree.click_element(window, element, control_type)
    _log_action("ff_click", target=f"{window}/{element}",
                 value=result.method, success=result.success)
    return str(result)


@mcp.tool()
def ff_type(
    window: str,
    element: str,
    value: str,
    control_type: str = "",
) -> str:
    """フォーカスもカーソルも動かさずに UI 要素にテキストを設定する。

    Args:
        window: ウィンドウタイトルのキーワード
        element: テキストを設定する要素の名前（部分一致）
        value: 設定するテキスト
        control_type: フィルタ用のコントロール型
    """
    result = _focusfree.set_value(window, element, value, control_type)
    _log_action("ff_type", target=f"{window}/{element}",
                 value=f"{result.method}: {value[:50]}", success=result.success)
    return str(result)


@mcp.tool()
def ff_list(window: str) -> str:
    """指定ウィンドウ内のフォーカスフリーで操作可能な UI 要素を一覧する。

    各要素が対応するパターン（Invoke, Value, Toggle 等）も表示。
    これを見て ff_click / ff_type で操作する対象を決める。

    Args:
        window: ウィンドウタイトルのキーワード
    """
    elements = _focusfree.list_actionable(window)
    if not elements:
        return f"ウィンドウ '{window}' に操作可能な要素が見つかりません"

    lines = [f"操作可能な要素: {len(elements)}件"]
    for el in elements[:30]:
        patterns = []
        if el.has_invoke: patterns.append("Invoke")
        if el.has_value: patterns.append("Value")
        if el.has_toggle: patterns.append("Toggle")
        if el.has_expand: patterns.append("Expand")
        if el.has_selection: patterns.append("Select")
        p_str = ",".join(patterns)
        val_str = f' = "{el.value[:40]}"' if el.value else ""
        lines.append(
            f"  [{el.control_type}] \"{el.name}\" "
            f"({el.center_x},{el.center_y}) "
            f"{{{p_str}}}{val_str}"
        )
    if len(elements) > 30:
        lines.append(f"  ... 他 {len(elements) - 30} 件")
    return "\n".join(lines)


@mcp.tool()
def ff_info(
    window: str,
    element: str,
    control_type: str = "",
) -> str:
    """UI 要素の詳細情報を取得する（対応パターン、座標、HWND 等）。

    Args:
        window: ウィンドウタイトルのキーワード
        element: 要素名（部分一致）
        control_type: フィルタ用
    """
    info = _focusfree.get_element_info(window, element, control_type)
    if not info:
        return f"要素 '{element}' が見つかりません（ウィンドウ '{window}'）"

    patterns = []
    if info.has_invoke: patterns.append("Invoke（クリック可能）")
    if info.has_value: patterns.append("Value（テキスト設定可能）")
    if info.has_toggle: patterns.append("Toggle（チェックボックス）")
    if info.has_expand: patterns.append("ExpandCollapse（メニュー開閉）")
    if info.has_selection: patterns.append("SelectionItem（選択）")

    lines = [
        f"要素: \"{info.name}\" [{info.control_type}]",
        f"位置: ({info.center_x},{info.center_y}) {info.width}x{info.height}",
        f"HWND: {info.hwnd or '(なし)'}",
    ]
    if info.value:
        lines.append(f"値: \"{info.value[:100]}\"")
    if patterns:
        lines.append(f"対応パターン:")
        for p in patterns:
            lines.append(f"  ✓ {p}")
    else:
        lines.append("対応パターン: なし（PostMessage or SendInput が必要）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# グリッド線バイナリサーチ (grid_find)
# ---------------------------------------------------------------------------
# X軸・Y軸を独立に1D探索して UI 要素の座標を特定する。
# 番号付きの線を画面に引き、LLM に「どの線がターゲットを横切るか」を聞く。
# zoom_and_refine より判断が単純（線番号を答えるだけ）でステートレス。


def _grid_find_parse_response(text: str, num_lines: int) -> dict:
    """LLM の応答をパースする。

    Returns:
        {"type": "exact", "line": N}
        {"type": "between", "line_a": A, "line_b": B}
        {"type": "not_found"}
    """
    import re as _re
    text = text.strip().upper()

    # "L5" or "L 5" — 線がターゲットを横切る (search で長文中も対応)
    m = _re.search(r"\bL\s*(\d+)\b", text)
    if m:
        n = int(m.group(1))
        if 1 <= n <= num_lines:
            return {"type": "exact", "line": n}

    # "B3-4" or "B 3-4" or "B3,4" — 2本の線の間
    m = _re.search(r"\bB\s*(\d+)\s*[-,]\s*(\d+)\b", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a and b <= num_lines + 1 and a < b:
            return {"type": "between", "line_a": a, "line_b": b}

    if "NONE" in text:
        return {"type": "not_found"}

    return {"type": "not_found"}


def _grid_find_line_pos(line_num: int, lo: int, hi: int, num_lines: int) -> int:
    """線番号 (1-indexed) からピクセル位置を算出する。

    線は lo..hi の範囲の両端を含めて等間隔に配置される。
    line 1 = lo 付近、line N = hi 付近（端のUI要素もカバー）。
    line_num=0 → lo、line_num=num_lines+1 → hi として端を表現。
    """
    if line_num <= 0:
        return lo
    if line_num > num_lines:
        return hi
    # 両端にもマージンを持たせるが小さめ (2%) にする
    margin = int((hi - lo) * 0.02)
    inner_lo = lo + margin
    inner_hi = hi - margin
    if num_lines == 1:
        return (inner_lo + inner_hi) // 2
    return inner_lo + int((line_num - 1) * (inner_hi - inner_lo) / (num_lines - 1))


def _grid_find_draw_lines(
    pil_img: "Image",
    axis: str,
    lo: int,
    hi: int,
    num_lines: int,
    crosshair_pos: int = -1,
) -> "Image":
    """画像に番号付きグリッド線を描画する。

    Args:
        pil_img: 元のスクリーンショット (PIL Image)
        axis: "x" (縦線) or "y" (横線)
        lo: 探索範囲の開始ピクセル
        hi: 探索範囲の終了ピクセル
        num_lines: 描画する線の本数
        crosshair_pos: 確定済みの他軸座標 (>= 0 ならクロスヘア表示)

    Returns:
        線を描画した PIL Image のコピー
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # 色: 縦線=シアン、横線=マゼンタ
    line_color = (0, 220, 220) if axis == "x" else (220, 0, 220)

    # フォント
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # 探索範囲をハイライト（半透明オーバーレイ）
    overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    if axis == "x":
        # 探索外を暗くする
        if lo > 0:
            ov_draw.rectangle([0, 0, lo, h], fill=(0, 0, 0, 60))
        if hi < w:
            ov_draw.rectangle([hi, 0, w, h], fill=(0, 0, 0, 60))
    else:
        if lo > 0:
            ov_draw.rectangle([0, 0, w, lo], fill=(0, 0, 0, 60))
        if hi < h:
            ov_draw.rectangle([0, hi, w, h], fill=(0, 0, 0, 60))
    img = PILImage.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 番号付き線を描画
    for i in range(1, num_lines + 1):
        pos = _grid_find_line_pos(i, lo, hi, num_lines)
        label = str(i)

        if axis == "x":
            # 縦線
            draw.line([(pos, 0), (pos, h)], fill=line_color, width=2)
            # ラベル（上部）
            bbox = font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx, ly = pos - tw // 2, 4
            draw.rectangle([lx - 2, ly - 1, lx + tw + 2, ly + th + 1],
                           fill=(0, 0, 0))
            draw.text((lx, ly), label, fill=(255, 255, 255), font=font)
        else:
            # 横線
            draw.line([(0, pos), (w, pos)], fill=line_color, width=2)
            # ラベル（左端）
            bbox = font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx, ly = 4, pos - th // 2
            draw.rectangle([lx - 2, ly - 1, lx + tw + 2, ly + th + 1],
                           fill=(0, 0, 0))
            draw.text((lx, ly), label, fill=(255, 255, 255), font=font)

    # 確定済み他軸のクロスヘア
    if crosshair_pos >= 0:
        ch_color = (255, 200, 0)  # 黄色
        if axis == "x":
            # Y 確定済み → 横線
            draw.line([(0, crosshair_pos), (w, crosshair_pos)],
                      fill=ch_color, width=1)
        else:
            # X 確定済み → 縦線
            draw.line([(crosshair_pos, 0), (crosshair_pos, h)],
                      fill=ch_color, width=1)

    return img


def _grid_find_ask_llm(
    b64_image: str,
    target: str,
    axis: str,
    num_lines: int,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """LLMにグリッド線画像を見せてターゲット位置を聞く。"""
    import requests as _req

    direction = "vertical" if axis == "x" else "horizontal"
    axis_label = "X (horizontal position)" if axis == "x" else "Y (vertical position)"
    edge_lo = "left" if axis == "x" else "above"
    edge_hi = "right" if axis == "x" else "below"

    system = (
        "You locate UI elements on screen. "
        f"The image has numbered {direction} lines (1-{num_lines}). "
        f"Find which line is closest to the target's {axis_label}. "
        "Reply with ONLY one of these codes, nothing else:\n"
        "L5 = line 5 crosses target\n"
        "B3-4 = target is between lines 3 and 4\n"
        f"B0-1 = target is {edge_lo} of line 1\n"
        f"B{num_lines}-{num_lines + 1} = target is {edge_hi} of line {num_lines}\n"
        "NONE = target not visible\n"
        "Output ONLY the code (e.g. L5 or B3-4). No words, no explanation."
    )
    user = f"Target: {target}\nAnswer with code only:"

    payload = {
        "model": model,
        "max_tokens": 10,
        "temperature": 0,
        "system": system,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": user},
                ],
            },
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers, timeout=30,
        )
        if resp.status_code != 200:
            logging.warning("grid_find LLM API error: %s %s",
                            resp.status_code, resp.text[:200])
            return {"type": "not_found"}
        text = resp.json()["content"][0]["text"].strip()
        logging.warning("grid_find LLM response (axis=%s): %r", axis, text)
        parsed = _grid_find_parse_response(text, num_lines)
        parsed["_raw"] = text
        return parsed
    except Exception as e:
        logging.warning("grid_find LLM API failed: %s", e)
        return {"type": "not_found"}


def _grid_find_to_base64(pil_img: "Image", quality: int = 80) -> str:
    """PIL Image → base64 JPEG 文字列"""
    import base64
    import io
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _grid_find_get_api_key() -> str | None:
    """ANTHROPIC_API_KEY を取得する。"""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            claude_json = Path.home() / ".claude.json"
            if claude_json.exists():
                import json as _json
                cfg = _json.loads(claude_json.read_text(encoding="utf-8"))
                api_key = (cfg.get("mcpServers", {})
                           .get("video2ai-desktop", {})
                           .get("env", {})
                           .get("ANTHROPIC_API_KEY", ""))
        except Exception:
            pass
    return api_key or None


def _grid_find_search_axis(
    pil_img: "Image",
    target: str,
    axis: str,
    num_lines: int,
    precision_px: int,
    api_key: str,
    model: str,
    shot_size: int,
    mon_phys_size: int,
    crosshair_pos: int = -1,
) -> tuple[int, int, list[str]]:
    """1軸のバイナリサーチを実行する（再帰ズーム付き）。

    区間が狭くなったら、その区間だけクロップして拡大し線を引く。
    これにより線の間隔が常に広く保たれ、LLM が正確に判断できる。
    precision_px=1 にすればピクセル精度に到達可能。

    Args:
        pil_img: スクリーンショット (PIL)
        target: 探索対象の説明
        axis: "x" or "y"
        num_lines: 1ステップあたりの線数
        precision_px: 目標精度（物理ピクセル）
        api_key: Anthropic API キー
        model: Claude モデル名
        shot_size: スクリーンショットのこの軸のサイズ (px)
        mon_phys_size: モニターの物理サイズ (px)
        crosshair_pos: 確定済みの他軸座標 (スクリーンショット座標)

    Returns:
        (screenshot_coord, physical_coord, step_log)
    """
    from PIL import Image as _PILImg

    lo, hi = 0, shot_size
    steps = []
    max_iterations = 8  # 再帰ズームで多めに
    ZOOM_UPSCALE = 800  # クロップ後の拡大サイズ
    # 区間がこのピクセル数以下になったらクロップ→拡大に切替
    ZOOM_THRESHOLD = shot_size // 4

    for iteration in range(max_iterations):
        # 物理ピクセルでの現在の区間幅
        phys_interval = int((hi - lo) * mon_phys_size / shot_size)
        if phys_interval <= precision_px:
            break

        # 区間が狭い場合はクロップ→拡大（再帰ズーム）
        interval = hi - lo
        if interval < ZOOM_THRESHOLD and interval > 0:
            # クロップ範囲を広めに取る（ターゲットの視覚的コンテキスト維持）
            ctx = max(interval * 3, 100)  # 区間の3倍 or 最低100px
            crop_lo = max(0, lo - ctx)
            crop_hi = min(shot_size, hi + ctx)

            if axis == "x":
                cropped = pil_img.crop((crop_lo, 0, crop_hi, pil_img.height))
            else:
                cropped = pil_img.crop((0, crop_lo, pil_img.width, crop_hi))

            # 拡大（アスペクト比維持）
            cw, ch = cropped.size
            if axis == "x":
                scale_factor = ZOOM_UPSCALE / cw
                new_size = (ZOOM_UPSCALE, max(1, int(ch * scale_factor)))
            else:
                scale_factor = ZOOM_UPSCALE / ch
                new_size = (max(1, int(cw * scale_factor)), ZOOM_UPSCALE)
            # Clamp to API max image dimension (8000px)
            max_dim = max(new_size)
            if max_dim > 7900:
                clamp_sf = 7900 / max_dim
                new_size = (max(1, int(new_size[0] * clamp_sf)),
                            max(1, int(new_size[1] * clamp_sf)))
                scale_factor = scale_factor * clamp_sf
            zoomed = cropped.resize(new_size, _PILImg.Resampling.BICUBIC)

            # ズーム画像内での lo/hi 位置を計算
            zoom_lo = int((lo - crop_lo) * scale_factor)
            zoom_hi = int((hi - crop_lo) * scale_factor)
            zoom_size = new_size[0] if axis == "x" else new_size[1]

            # クロスヘアもスケーリング
            zoom_crosshair = -1
            if crosshair_pos >= 0:
                if axis == "x":
                    zoom_crosshair = crosshair_pos  # Y軸は変わらない
                else:
                    zoom_crosshair = crosshair_pos  # X軸は変わらない

            # ズーム画像に線を描画して LLM に質問
            annotated = _grid_find_draw_lines(
                zoomed, axis, zoom_lo, zoom_hi, num_lines, zoom_crosshair)
            b64 = _grid_find_to_base64(annotated)
            result = _grid_find_ask_llm(b64, target, axis, num_lines,
                                         api_key, model)
            raw = result.get("_raw", "?")

            if result["type"] == "exact":
                # ズーム座標→元画像座標に変換
                zoom_pos = _grid_find_line_pos(
                    result["line"], zoom_lo, zoom_hi, num_lines)
                pos = crop_lo + int(zoom_pos / scale_factor)
                phys = int(pos * mon_phys_size / shot_size)
                steps.append(
                    f"  step{iteration + 1} [zoom]: L{result['line']} → "
                    f"shot={pos}, phys={phys} (raw={raw!r})")
                return pos, phys, steps

            elif result["type"] == "between":
                zoom_new_lo = _grid_find_line_pos(
                    result["line_a"], zoom_lo, zoom_hi, num_lines)
                zoom_new_hi = _grid_find_line_pos(
                    result["line_b"], zoom_lo, zoom_hi, num_lines)
                new_lo = crop_lo + int(zoom_new_lo / scale_factor)
                new_hi = crop_lo + int(zoom_new_hi / scale_factor)
                steps.append(
                    f"  step{iteration + 1} [zoom]: B{result['line_a']}-"
                    f"{result['line_b']} → [{new_lo}..{new_hi}] "
                    f"(raw={raw!r})")
                lo, hi = new_lo, new_hi

            else:
                steps.append(
                    f"  step{iteration + 1} [zoom]: NOT FOUND (raw={raw!r})")
                return -1, -1, steps

        else:
            # 通常モード: 元画像全体に線を描画
            annotated = _grid_find_draw_lines(
                pil_img, axis, lo, hi, num_lines, crosshair_pos)
            b64 = _grid_find_to_base64(annotated)
            result = _grid_find_ask_llm(b64, target, axis, num_lines,
                                         api_key, model)
            raw = result.get("_raw", "?")

            if result["type"] == "exact":
                pos = _grid_find_line_pos(result["line"], lo, hi, num_lines)
                phys = int(pos * mon_phys_size / shot_size)
                steps.append(
                    f"  step{iteration + 1}: L{result['line']} → "
                    f"shot={pos}, phys={phys} (raw={raw!r})")
                return pos, phys, steps

            elif result["type"] == "between":
                new_lo = _grid_find_line_pos(
                    result["line_a"], lo, hi, num_lines)
                new_hi = _grid_find_line_pos(
                    result["line_b"], lo, hi, num_lines)
                steps.append(
                    f"  step{iteration + 1}: B{result['line_a']}-"
                    f"{result['line_b']} → [{new_lo}..{new_hi}] "
                    f"(raw={raw!r})")
                lo, hi = new_lo, new_hi

            else:
                steps.append(
                    f"  step{iteration + 1}: NOT FOUND (raw={raw!r})")
                return -1, -1, steps

    # 収束: 区間の中央を返す
    pos = (lo + hi) // 2
    phys = int(pos * mon_phys_size / shot_size)
    steps.append(f"  converged: shot={pos}, phys={phys} "
                 f"(interval={hi - lo}px)")
    return pos, phys, steps


@mcp.tool()
@_preserves_screenshot_state
def grid_find(
    target: str,
    monitor: int = 3,
    width: int = 1920,
    num_lines: int = 8,
    precision_px: int = 30,
    window: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> Image:
    """グリッド線バイナリサーチでUI要素の座標を特定する。

    X軸・Y軸を独立に探索。番号付き線を引いてLLMに「どの線がターゲットを
    横切るか」を聞き、区間を絞り込む。各軸1-2ステップ、合計3-4回のLLM呼出し
    で±30px精度の座標を返す。ステートレス（1回の呼出しで完了）。

    UIA非対応アプリ（CSP、ゲーム、レガシーアプリ等）のボタンやメニューの
    座標特定に適する。ピンポイント精度が必要な場合は SoM 方式を使用。

    window を指定するとそのウィンドウだけをキャプチャし、探索範囲を限定して
    精度を大幅に向上させる（推奨）。

    Args:
        target: 探したいUI要素の説明（例: "保存ボタン", "赤いスライダー"）
        monitor: モニター番号（window 指定時は自動検出）
        width: スクリーンショットのリサイズ幅
        num_lines: 1ステップあたりのグリッド線数（デフォルト8）
        precision_px: 目標精度（物理ピクセル、デフォルト30）
        window: ウィンドウタイトルのキーワード（部分一致）。指定推奨。
        model: 使用する Claude モデル
    """
    import time as _time
    import ctypes
    import ctypes.wintypes

    t0 = _time.time()

    # API キー取得
    api_key = _grid_find_get_api_key()
    if not api_key:
        return "エラー: ANTHROPIC_API_KEY が見つかりません（環境変数 or ~/.claude.json）"

    import cv2 as _cv2
    from PIL import Image as PILImage
    import mss

    win_offset_x, win_offset_y = 0, 0  # ウィンドウの物理座標オフセット

    if window:
        # ウィンドウ検索
        user32 = ctypes.windll.user32
        target_hwnd = None
        target_title = ""

        def enum_cb(hwnd, _):
            nonlocal target_hwnd, target_title
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if window in buf.value:
                    target_hwnd = hwnd
                    target_title = buf.value
                    return False
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
        )
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

        if not target_hwnd:
            return f"grid_find: ウィンドウ「{window}」が見つかりません"

        # ウィンドウ矩形取得
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(target_hwnd, ctypes.byref(rect))
        win_left = max(rect.left, 0)
        win_top = max(rect.top, 0)
        win_w = rect.right - rect.left
        win_h = rect.bottom - rect.top
        if rect.left < 0:
            win_w += rect.left
        if rect.top < 0:
            win_h += rect.top

        # タイトルバー除外: クライアント領域との差分で高さを推定
        client_rect = ctypes.wintypes.RECT()
        user32.GetClientRect(target_hwnd, ctypes.byref(client_rect))
        client_pt = ctypes.wintypes.POINT(0, 0)
        user32.ClientToScreen(target_hwnd, ctypes.byref(client_pt))
        titlebar_h = client_pt.y - rect.top
        if titlebar_h <= 0:
            # カスタムタイトルバー（Win11 モダンアプリ等）: SM_CYCAPTION で推定
            SM_CYCAPTION = 4
            titlebar_h = user32.GetSystemMetrics(SM_CYCAPTION)
        # タイトルバー分をクロップ
        win_top += titlebar_h
        win_h -= titlebar_h

        win_offset_x, win_offset_y = win_left, win_top

        # ウィンドウが属するモニター検出
        with mss.mss() as sct:
            for i in range(1, len(sct.monitors)):
                mon = sct.monitors[i]
                cx = win_left + win_w // 2
                cy = win_top + win_h // 2
                if (mon['left'] <= cx < mon['left'] + mon['width'] and
                        mon['top'] <= cy < mon['top'] + mon['height']):
                    monitor = i
                    break

        # ウィンドウ領域キャプチャ（フル解像度で撮影）
        cap = ScreenCapture(
            region=(win_left, win_top, win_w, win_h), resize_width=0)
        img_bgr = cap.capture()
        native_w, native_h = img_bgr.shape[1], img_bgr.shape[0]
        # 物理サイズ = ウィンドウサイズ
        mon_phys_w, mon_phys_h = win_w, win_h

        # LLM がアイコンを判別しやすいように拡大
        # width パラメータをLLM用拡大サイズとして使用（デフォルト1920）
        upscale_w = max(width, native_w) if width > 0 else native_w
        if upscale_w > native_w:
            scale = upscale_w / native_w
            upscale_h = int(native_h * scale)
            img_bgr = _cv2.resize(img_bgr, (upscale_w, upscale_h),
                                  interpolation=_cv2.INTER_CUBIC)
        shot_w, shot_h = img_bgr.shape[1], img_bgr.shape[0]
    else:
        # モニター全体キャプチャ
        cap = ScreenCapture(monitor_index=monitor, resize_width=width)
        img_bgr = cap.capture()
        shot_w, shot_h = img_bgr.shape[1], img_bgr.shape[0]
        _last_screenshot_info[monitor] = (shot_w, shot_h)

        mon_phys_w, mon_phys_h = shot_w, shot_h
        try:
            with mss.mss() as sct:
                if monitor < len(sct.monitors):
                    m = sct.monitors[monitor]
                    mon_phys_w, mon_phys_h = m["width"], m["height"]
        except Exception:
            pass

    pil_img = PILImage.fromarray(_cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB))

    # X 軸探索
    x_shot, x_phys, x_log = _grid_find_search_axis(
        pil_img, target, "x", num_lines, precision_px,
        api_key, model, shot_w, mon_phys_w,
    )
    if x_shot < 0:
        elapsed = _time.time() - t0
        return (f"grid_find: ターゲット「{target}」のX座標が見つかりません "
                f"({elapsed:.1f}s)\n" + "\n".join(x_log))

    # Y 軸探索: X確定位置の周辺を縦スライスとして切り出し
    # ターゲット周辺のみに絞ることでLLMの判断精度を向上させる
    slice_half = max(shot_w // 6, 200)  # 幅の1/6 or 最低200px
    slice_left = max(0, x_shot - slice_half)
    slice_right = min(shot_w, x_shot + slice_half)
    y_slice = pil_img.crop((slice_left, 0, slice_right, shot_h))
    y_shot, y_phys, y_log = _grid_find_search_axis(
        y_slice, target, "y", num_lines, precision_px,
        api_key, model, shot_h, mon_phys_h,
    )
    if y_shot < 0:
        elapsed = _time.time() - t0
        return (f"grid_find: ターゲット「{target}」のY座標が見つかりません "
                f"({elapsed:.1f}s)\n" + "\n".join(x_log + y_log))

    # 物理座標に変換
    if window:
        # ウィンドウキャプチャ: shot座標→ウィンドウ内物理座標→画面物理座標
        phys_x = win_offset_x + int(x_shot * mon_phys_w / shot_w)
        phys_y = win_offset_y + int(y_shot * mon_phys_h / shot_h)
        # _screen_to_physical 用に情報を記録
        _last_screenshot_info[monitor] = (shot_w, shot_h)
        _last_window_capture[monitor] = (
            win_offset_x, win_offset_y, mon_phys_w, mon_phys_h)
    else:
        phys_x, phys_y = _screen_to_physical(monitor, x_shot, y_shot)

    _record_verified_coord(phys_x, phys_y, "grid_find")
    elapsed = _time.time() - t0

    # peek 画像を生成して結果と一緒に返す
    peek_size = 200
    peek_left = max(0, phys_x - peek_size // 2)
    peek_top = max(0, phys_y - peek_size // 2)
    peek_cap = ScreenCapture(region=(peek_left, peek_top, peek_size, peek_size))
    peek_pil = peek_cap.capture_as_pil()
    peek_enlarged = peek_pil.resize(
        (peek_size * 2, peek_size * 2), resample=0)  # NEAREST

    # 十字線描画（ターゲット位置）
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(peek_enlarged)
    cx, cy = peek_size, peek_size  # 拡大後の中央
    draw.line([(cx - 20, cy), (cx + 20, cy)], fill="red", width=2)
    draw.line([(cx, cy - 20), (cx, cy + 20)], fill="red", width=2)

    # インフォバー描画（下部に座標とクリックコマンド）
    try:
        info_font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        info_font = ImageFont.load_default()
    # ステップログを構築
    log_lines = (
        [f"grid_find: ({phys_x},{phys_y}) {elapsed:.1f}s",
         f"click(x={x_shot}, y={y_shot}, monitor={monitor})"]
        + [f"X: {l.strip()}" for l in x_log]
        + [f"Y: {l.strip()}" for l in y_log]
    )
    info_h = 14 * len(log_lines) + 8
    result_img = PILImage.new("RGB",
                              (peek_size * 2, peek_size * 2 + info_h),
                              (0, 0, 0))
    result_img.paste(peek_enlarged, (0, 0))
    info_draw = ImageDraw.Draw(result_img)
    for li, line in enumerate(log_lines):
        info_draw.text((8, peek_size * 2 + 4 + li * 14), line,
                       fill=(255, 255, 255), font=info_font)

    buf = BytesIO()
    result_img.save(buf, format="PNG")
    import json as _json
    coord_meta = {"tool": "grid_find", "x": phys_x, "y": phys_y,
                  "monitor": monitor, "elapsed": round(elapsed, 2)}
    return [Image(data=buf.getvalue(), format="png"),
            f"COORDINATE_META:{_json.dumps(coord_meta)}"]


# ---------------------------------------------------------------------------
# SoM (Set-of-Mark) — 番号付きUI要素検出
# ---------------------------------------------------------------------------
# OpenCV で画面上のUI要素を検出し、番号を振った画像を LLM に見せて
# 「どの番号？」と聞く。ピンポイント精度が必要な場合に使用。


def _som_build_thumbnail_grid(
    img_bgr,
    elements: list[dict],
    thumb_size: int = 64,
    cols: int = 10,
) -> "np.ndarray":
    """検出された要素のサムネイル一覧画像を生成する。

    各要素を切り出し、thumb_size に拡大し、番号ラベル付きで
    グリッド状に並べた画像を返す。
    """
    import cv2 as _cv2
    import numpy as np

    n = len(elements)
    rows = (n + cols - 1) // cols
    cell_w = thumb_size + 4  # 余白
    cell_h = thumb_size + 20  # 番号ラベル用スペース
    grid_w = cols * cell_w
    grid_h = rows * cell_h
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid[:] = 40  # ダークグレー背景

    h, w = img_bgr.shape[:2]
    for i, el in enumerate(elements):
        rx, ry, rw, rh = el["rect"]
        # 切り出し（padding 付き）
        pad = 4
        x1 = max(0, rx - pad)
        y1 = max(0, ry - pad)
        x2 = min(w, rx + rw + pad)
        y2 = min(h, ry + rh + pad)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # thumb_size に拡大（アスペクト比維持）
        ch, cw = crop.shape[:2]
        scale = min(thumb_size / cw, thumb_size / ch)
        new_w = max(1, int(cw * scale))
        new_h = max(1, int(ch * scale))
        thumb = _cv2.resize(crop, (new_w, new_h),
                            interpolation=_cv2.INTER_CUBIC)

        # グリッド上の位置
        col = i % cols
        row = i // cols
        gx = col * cell_w + (cell_w - new_w) // 2
        gy = row * cell_h + 16 + (thumb_size - new_h) // 2  # 16px = ラベル高さ

        # サムネイルを配置
        grid[gy:gy + new_h, gx:gx + new_w] = thumb

        # 番号ラベル（セル上部）
        label = str(i + 1)
        font_scale = 0.45
        (tw, th), _ = _cv2.getTextSize(
            label, _cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lx = col * cell_w + (cell_w - tw) // 2
        ly = row * cell_h + 12
        _cv2.putText(grid, label, (lx, ly),
                     _cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                     (0, 255, 255), 1, _cv2.LINE_AA)

    return grid


def _som_ask_llm(
    b64_overview: str,
    b64_thumbnails: str,
    target: str,
    num_elements: int,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> int | None:
    """番号付き SoM 画像 + サムネイル一覧を LLM に見せてターゲット番号を聞く。"""
    import requests as _req

    system = (
        "You identify UI elements on screen. "
        "Image 1: Screenshot with numbered green rectangles marking detected elements. "
        "Image 2: Thumbnail grid showing each element enlarged with its number. "
        f"Elements are numbered 1-{num_elements}. "
        "Find the element matching the target. "
        "Reply with ONLY the number. If not found, reply NONE."
    )
    user = f"Target: {target}\nWhich number?"

    content = [
        {"type": "text", "text": "Image 1 - Overview with numbered elements:"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64_overview,
            },
        },
        {"type": "text", "text": "Image 2 - Enlarged thumbnails of each element:"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64_thumbnails,
            },
        },
        {"type": "text", "text": user},
    ]

    payload = {
        "model": model,
        "max_tokens": 8,
        "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers, timeout=30,
        )
        if resp.status_code != 200:
            logging.warning("som_find LLM error: %s", resp.status_code)
            return None
        text = resp.json()["content"][0]["text"].strip()
        logging.warning("som_find LLM response: %r", text)
        import re
        m = re.search(r"\d+", text)
        if m:
            n = int(m.group())
            if 1 <= n <= num_elements:
                return n
        return None
    except Exception as e:
        logging.warning("som_find LLM failed: %s", e)
        return None


@mcp.tool()
@_preserves_screenshot_state
def som_find(
    target: str,
    window: str = "",
    monitor: int = 3,
    width: int = 3840,
    min_size: int = 15,
    max_size: int = 300,
    model: str = "claude-sonnet-4-20250514",
) -> Image:
    """SoM (Set-of-Mark) 方式でUI要素の座標をピンポイントで特定する。

    OpenCV で画面上のUI要素を検出し、番号を振った画像を LLM に見せて
    「どの番号がターゲット？」と聞く。1回の LLM 呼び出しで完了。

    grid_find より精度が高い（要素の中心座標を直接取得）が、
    要素が検出できない場合は使えない。

    Args:
        target: 探したいUI要素の説明（例: "保存ボタン", "カラーピッカー"）
        window: ウィンドウタイトルのキーワード（部分一致）。指定推奨。
        monitor: モニター番号（window 指定時は自動検出）
        width: スクリーンショットの拡大幅
        min_size: 検出する最小要素サイズ (px)
        max_size: 検出する最大要素サイズ (px)
        model: 使用する Claude モデル
    """
    import time as _time
    import ctypes
    import ctypes.wintypes

    t0 = _time.time()

    api_key = _grid_find_get_api_key()
    if not api_key:
        return "エラー: ANTHROPIC_API_KEY が見つかりません"

    import cv2 as _cv2
    import numpy as np
    from PIL import Image as PILImage
    import mss

    win_offset_x, win_offset_y = 0, 0
    _som_pad_x, _som_pad_y = 0, 0  # ウィンドウキャプチャ時の余白オフセット

    if window:
        # ウィンドウ検索（grid_find と同じパターン）
        user32 = ctypes.windll.user32
        target_hwnd = None

        def enum_cb(hwnd, _):
            nonlocal target_hwnd
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if window in buf.value:
                    target_hwnd = hwnd
                    return False
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

        if not target_hwnd:
            return f"som_find: ウィンドウ「{window}」が見つかりません"

        # ウィンドウ矩形 + タイトルバー除外
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(target_hwnd, ctypes.byref(rect))
        win_left = max(rect.left, 0)
        win_top = max(rect.top, 0)
        win_w = rect.right - rect.left
        win_h = rect.bottom - rect.top
        if rect.left < 0:
            win_w += rect.left
        if rect.top < 0:
            win_h += rect.top

        client_rect = ctypes.wintypes.RECT()
        user32.GetClientRect(target_hwnd, ctypes.byref(client_rect))
        client_pt = ctypes.wintypes.POINT(0, 0)
        user32.ClientToScreen(target_hwnd, ctypes.byref(client_pt))
        titlebar_h = client_pt.y - rect.top
        if titlebar_h <= 0:
            SM_CYCAPTION = 4
            titlebar_h = user32.GetSystemMetrics(SM_CYCAPTION)
        win_top += titlebar_h
        win_h -= titlebar_h

        win_offset_x, win_offset_y = win_left, win_top

        with mss.mss() as sct:
            for i in range(1, len(sct.monitors)):
                mon = sct.monitors[i]
                cx = win_left + win_w // 2
                cy = win_top + win_h // 2
                if (mon['left'] <= cx < mon['left'] + mon['width'] and
                        mon['top'] <= cy < mon['top'] + mon['height']):
                    monitor = i
                    break

        # ウィンドウをキャプチャ（フル解像度）
        cap = ScreenCapture(
            region=(win_left, win_top, win_w, win_h), resize_width=0)
        img_bgr = cap.capture()
        mon_phys_w, mon_phys_h = win_w, win_h

        # 端のUI要素の番号ラベルが見切れないように余白を追加
        # （検出は余白なし画像で行い、表示用に余白付き画像を作る）
        _som_pad = 30
        _som_pad_x, _som_pad_y = _som_pad, _som_pad
    else:
        cap = ScreenCapture(monitor_index=monitor, resize_width=0)
        img_bgr = cap.capture()
        mon_phys_w, mon_phys_h = img_bgr.shape[1], img_bgr.shape[0]
        try:
            with mss.mss() as sct:
                if monitor < len(sct.monitors):
                    m = sct.monitors[monitor]
                    mon_phys_w, mon_phys_h = m["width"], m["height"]
        except Exception:
            pass

    native_h, native_w = img_bgr.shape[:2]

    # UI 要素検出（フル解像度で実行）
    from agent.template_matcher import detect_ui_elements
    elements = detect_ui_elements(img_bgr, min_size=min_size, max_size=max_size)

    if not elements:
        return "som_find: UI 要素が検出されませんでした"

    # 要素数が多すぎる場合は面積の小さいものに絞る（UI部品 > テキスト塊）
    MAX_ELEMENTS = 50
    if len(elements) > MAX_ELEMENTS:
        elements.sort(key=lambda e: e["rect"][2] * e["rect"][3])
        elements = elements[:MAX_ELEMENTS]
        # 位置順に再ソート
        elements.sort(key=lambda e: (e["rect"][1] // 30, e["rect"][0]))

    # LLM 用に拡大
    upscale_w = max(width, native_w) if width > 0 else native_w
    if upscale_w > native_w:
        scale = upscale_w / native_w
        upscale_h = int(native_h * scale)
        display_bgr = _cv2.resize(img_bgr, (upscale_w, upscale_h),
                                  interpolation=_cv2.INTER_CUBIC)
    else:
        scale = 1.0
        display_bgr = img_bgr.copy()

    # 余白を追加（端の番号ラベルが見切れないように）
    pad_scaled = int(_som_pad_x * scale) if window else 0
    if pad_scaled > 0:
        dh, dw = display_bgr.shape[:2]
        padded = np.zeros((dh + pad_scaled * 2, dw + pad_scaled * 2, 3),
                          dtype=np.uint8)
        # 余白を周辺色で埋める（背景色推定）
        bg_color = display_bgr[0, 0].tolist()
        padded[:] = bg_color
        padded[pad_scaled:pad_scaled + dh,
               pad_scaled:pad_scaled + dw] = display_bgr
        display_bgr = padded

    # 番号付きマーカーを描画（矩形枠 + 小さな番号ラベル）
    for i, el in enumerate(elements, 1):
        rx, ry, rw, rh = el["rect"]
        # スケーリング + 余白オフセット
        sx = int(rx * scale) + pad_scaled
        sy = int(ry * scale) + pad_scaled
        sw = int(rw * scale)
        sh = int(rh * scale)

        # 緑の矩形枠で要素を囲む
        _cv2.rectangle(display_bgr, (sx, sy), (sx + sw, sy + sh),
                       (0, 255, 0), 1)

        # 左上に小さな番号ラベル（赤背景 + 白数字）
        label = str(i)
        font_scale = 0.4
        (tw, th), baseline = _cv2.getTextSize(
            label, _cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lx, ly = sx, max(sy - 2, 0)
        # 背景矩形
        _cv2.rectangle(display_bgr,
                       (lx, ly - th - 2), (lx + tw + 4, ly + 2),
                       (0, 0, 200), -1)
        _cv2.putText(display_bgr, label,
                     (lx + 2, ly),
                     _cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                     (255, 255, 255), 1, _cv2.LINE_AA)

    # サムネイルグリッド生成
    thumb_grid = _som_build_thumbnail_grid(img_bgr, elements)

    # LLM に質問（全体画像 + サムネイル一覧の2枚）
    pil_display = PILImage.fromarray(
        _cv2.cvtColor(display_bgr, _cv2.COLOR_BGR2RGB))
    b64_overview = _grid_find_to_base64(pil_display)
    pil_thumbs = PILImage.fromarray(
        _cv2.cvtColor(thumb_grid, _cv2.COLOR_BGR2RGB))
    b64_thumbs = _grid_find_to_base64(pil_thumbs)
    selected = _som_ask_llm(b64_overview, b64_thumbs, target,
                            len(elements), api_key, model)

    if selected is None:
        elapsed = _time.time() - t0
        # 番号付き画像を返す（ユーザーが自分で選べるように）
        info_text = (f"som_find: LLM がターゲットを特定できませんでした "
                     f"({len(elements)}要素検出, {elapsed:.1f}s)")
        # インフォバー追加
        from PIL import ImageDraw, ImageFont
        try:
            info_font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            info_font = ImageFont.load_default()
        info_h = 30
        result_img = PILImage.new("RGB",
                                  (pil_display.width, pil_display.height + info_h),
                                  (0, 0, 0))
        result_img.paste(pil_display, (0, 0))
        ImageDraw.Draw(result_img).text(
            (8, pil_display.height + 4), info_text,
            fill=(255, 255, 255), font=info_font)
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        return Image(data=buf.getvalue(), format="png")

    # 選択された要素の座標を取得
    el = elements[selected - 1]
    ecx, ecy = el["center"]

    # 物理座標に変換（余白分を補正）
    if window:
        phys_x = win_offset_x + ecx - _som_pad_x
        phys_y = win_offset_y + ecy - _som_pad_y
    else:
        phys_x = ecx
        phys_y = ecy

    _record_verified_coord(phys_x, phys_y, "som_find")

    # peek 画像
    peek_size = 200
    peek_left = max(0, phys_x - peek_size // 2)
    peek_top = max(0, phys_y - peek_size // 2)
    peek_cap = ScreenCapture(region=(peek_left, peek_top, peek_size, peek_size))
    peek_pil = peek_cap.capture_as_pil()
    peek_enlarged = peek_pil.resize(
        (peek_size * 2, peek_size * 2), resample=0)

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(peek_enlarged)
    pc = peek_size
    draw.line([(pc - 20, pc), (pc + 20, pc)], fill="red", width=2)
    draw.line([(pc, pc - 20), (pc, pc + 20)], fill="red", width=2)

    try:
        info_font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        info_font = ImageFont.load_default()
    elapsed = _time.time() - t0
    info_h = 60
    result_img = PILImage.new("RGB",
                              (peek_size * 2, peek_size * 2 + info_h),
                              (0, 0, 0))
    result_img.paste(peek_enlarged, (0, 0))
    info_draw = ImageDraw.Draw(result_img)

    # screenshot座標を計算（click用）
    if window:
        _last_screenshot_info[monitor] = (native_w, native_h)
        _last_window_capture[monitor] = (
            win_offset_x, win_offset_y, mon_phys_w, mon_phys_h)
        ss_x = int(ecx * native_w / mon_phys_w) if mon_phys_w != native_w else ecx
        ss_y = int(ecy * native_h / mon_phys_h) if mon_phys_h != native_h else ecy
    else:
        ss_x, ss_y = ecx, ecy

    info_text = (
        f"som_find #{selected}: ({phys_x},{phys_y}) {elapsed:.1f}s "
        f"[{len(elements)}elements]\n"
        f"click(x={ss_x}, y={ss_y}, monitor={monitor})"
    )
    info_draw.text((8, peek_size * 2 + 4), info_text,
                   fill=(255, 255, 255), font=info_font)

    buf = BytesIO()
    result_img.save(buf, format="PNG")
    import json as _json
    coord_meta = {"tool": "som_find", "x": phys_x, "y": phys_y,
                  "monitor": monitor, "elapsed": round(elapsed, 2),
                  "selected": selected, "elements_count": len(elements)}
    return [Image(data=buf.getvalue(), format="png"),
            f"COORDINATE_META:{_json.dumps(coord_meta)}"]


# ---------------------------------------------------------------------------
# スマート検索 + テンプレートマッチング
# ---------------------------------------------------------------------------
# 階層的にUI要素を検索する:
#   1. テンプレートマッチング (OpenCV, ~50ms) — 保存済みテンプレート画像で検索
#   2. UI Automation (watcher_find, ~0.01ms) — ScreenWatcher が動作中なら
#   3. フォールバック: 座標なしで「見つかりません」を返す（Claude が screenshot で探す）
#
# テンプレートは app_knowledge/templates/{app}/ に保存され、
# app_knowledge 作成時のWeb検索や、操作中の screenshot から蓄積していく。


@mcp.tool()
@_preserves_screenshot_state
def smart_find(
    name: str,
    app: str = "",
    monitor: int = 3,
    threshold: float = 0.55,
    width: int = 1920,
) -> str:
    """UI要素を階層的に検索する（テンプレート → UI Automation → フォールバック）。

    どんなアプリでも動作する汎用検索。テンプレートが保存済みなら ~50ms、
    UI Automation が使えるアプリなら ~0.01ms で座標が返る。
    どちらも使えなければ「見つかりません」を返す（screenshot で探す必要あり）。

    テンプレートマッチングの信頼度:
      score >= 0.8: 高信頼（そのままクリック可能）
      0.55 <= score < 0.8: 低信頼（peek で目視確認を推奨）

    Args:
        name: 検索するUI要素の名前（テンプレート名 or UI要素名の部分一致）
        app: アプリ名（テンプレート検索のフィルタ）。空でフォアグラウンドアプリを自動推定
        monitor: スクリーンショット撮影用モニター番号
        threshold: テンプレートマッチングの一致度閾値（0-1）
        width: スクリーンショットのリサイズ幅
    """
    results_lines = []
    found = False

    # --- アプリ名の自動推定 ---
    if not app:
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
            title = buf.value
            # タイトルからアプリ名を推定（末尾の「 - アプリ名」パターン）
            if " - " in title:
                app = title.rsplit(" - ", 1)[-1].strip()
            elif title:
                app = title.strip()
        except Exception:
            pass
    app_safe = app.replace(" ", "_").replace("/", "_").replace("\\", "_") if app else ""

    # --- 1. テンプレートマッチング ---
    if app_safe:
        templates = _template_matcher.list_templates(app_safe)
        if app_safe in templates:
            # name に一致するテンプレートを検索
            matching_tmpls = [t for t in templates[app_safe]
                              if name.lower() in t["name"].lower()
                              or name.lower() in t.get("description", "").lower()]
            if matching_tmpls:
                # スクリーンショット撮影
                cap = ScreenCapture(monitor_index=monitor, resize_width=width)
                img = cap.capture()
                _last_screenshot_info[monitor] = (
                    img.shape[1], img.shape[0])

                for tmpl_info in matching_tmpls:
                    t0 = _time.perf_counter()
                    # まず等倍検索、見つからなければマルチスケール
                    matches = _template_matcher.find(
                        img, app_safe, tmpl_info["name"],
                        threshold=threshold)
                    if not matches:
                        matches = _template_matcher.find_multiscale(
                            img, app_safe, tmpl_info["name"],
                            threshold=max(threshold - 0.1, 0.3))
                    elapsed = (_time.perf_counter() - t0) * 1000

                    if matches:
                        m = matches[0]
                        # スクリーンショット座標 → 物理座標
                        phys_x, phys_y = _screen_to_physical(
                            monitor, m.center_x, m.center_y)

                        # 低信頼マッチは LLM で自動検証
                        if m.score < 0.8:
                            api_key = _grid_find_get_api_key()
                            if api_key:
                                from agent.template_matcher import verify_match_with_llm
                                verified = verify_match_with_llm(
                                    img, m, name, api_key)
                                if verified:
                                    confidence = "LLM検証済み"
                                else:
                                    # LLM が否定 → この候補をスキップ
                                    continue
                            else:
                                confidence = "低信頼"
                        else:
                            confidence = "高信頼"

                        _record_verified_coord(phys_x, phys_y, "smart_find")
                        line = (
                            f"[テンプレート] \"{tmpl_info['name']}\" "
                            f"score={m.score:.3f} ({elapsed:.1f}ms) [{confidence}]\n"
                            f"  screenshot: ({m.center_x},{m.center_y}) "
                            f"physical: ({phys_x},{phys_y})\n"
                            f"  → click(x={m.center_x}, y={m.center_y}, monitor={monitor})"
                        )
                        results_lines.append(line)
                        found = True

    # --- 2. UI Automation (ScreenWatcher) ---
    if not found and _screen_watcher and _screen_watcher.is_running():
        elements = _screen_watcher.find_elements(name=name)
        if elements:
            _record_verified_coord(elements[0].center_x, elements[0].center_y, "smart_find")
            for el in elements[:3]:
                results_lines.append(
                    f"[UI Automation] \"{el.name}\" [{el.control_type}]\n"
                    f"  physical: ({el.center_x},{el.center_y}) "
                    f"size={el.width}x{el.height}"
                )
                if el.value:
                    results_lines[-1] += f'\n  value="{el.value[:60]}"'
            found = True

    # --- 結果 ---
    if found:
        header = f"smart_find(\"{name}\", app=\"{app}\"): {len(results_lines)}件"
        return header + "\n\n" + "\n\n".join(results_lines)

    return (f"smart_find(\"{name}\", app=\"{app}\"): 見つかりません\n"
            f"  テンプレート: なし (template_save で登録可能)\n"
            f"  UI Automation: {'動作中だが該当なし' if _screen_watcher and _screen_watcher.is_running() else '未起動'}\n"
            f"  → screenshot で目視確認してください")


@mcp.tool()
@_preserves_screenshot_state
def template_save(
    app: str,
    name: str,
    monitor: int = 3,
    x: int = 0, y: int = 0, w: int = 0, h: int = 0,
    width: int = 1920,
    description: str = "",
    source_path: str = "",
) -> str:
    """スクリーンショットの領域をテンプレート画像として保存する。

    保存先: app_knowledge/templates/{app}/{name}.png
    保存後は smart_find で ~50ms で検索できるようになる。

    方法1: スクリーンショットから領域を切り出し
      template_save(app="notepad", name="save_button", monitor=3, x=100, y=50, w=30, h=30)

    方法2: 既存の画像ファイルを登録（Web検索で得た画像等）
      template_save(app="clip_studio", name="pen_tool", source_path="C:/path/to/pen.png")

    Args:
        app: アプリ名
        name: テンプレート名（英数字推奨）
        monitor: スクリーンショット撮影用モニター番号
        x, y, w, h: 切り出し領域（スクリーンショット座標）。全て0ならスクリーンショット全体
        width: スクリーンショットのリサイズ幅
        description: テンプレートの説明
        source_path: 既存画像ファイルのパス（指定時はスクリーンショットを撮らない）
    """
    # 方法2: 既存ファイルから
    if source_path:
        result = _template_matcher.save_template_from_file(
            app, name, source_path, description=description)
        _log_action("template_save", target=f"{app}/{name}",
                     value=f"from file: {source_path}")
        return result

    # 方法1: スクリーンショットから切り出し
    cap = ScreenCapture(monitor_index=monitor, resize_width=width)
    img = cap.capture()
    _last_screenshot_info[monitor] = (img.shape[1], img.shape[0])

    if w <= 0 or h <= 0:
        # 領域指定なし → 全体を保存（テスト用）
        h_img, w_img = img.shape[:2]
        x, y, w, h = 0, 0, w_img, h_img

    result = _template_matcher.save_template(
        app, name, img, x=x, y=y, w=w, h=h,
        description=description)
    _log_action("template_save", target=f"{app}/{name}",
                 value=f"region ({x},{y},{w}x{h})")
    return result


@mcp.tool()
def template_list(app: str = "") -> str:
    """保存済みテンプレートの一覧を表示する。

    Args:
        app: アプリ名でフィルタ（空で全アプリ）
    """
    templates = _template_matcher.list_templates(app)
    if not templates:
        return "テンプレートはまだ保存されていません"

    lines = []
    for app_name, tmpls in templates.items():
        lines.append(f"[{app_name}] ({len(tmpls)}個)")
        for t in tmpls:
            size = t.get("size", [0, 0])
            desc = f' — {t["description"]}' if t.get("description") else ""
            tags = f' #{" #".join(t["tags"])}' if t.get("tags") else ""
            lines.append(f"  {t['name']} ({size[0]}x{size[1]}){desc}{tags}")
    return "\n".join(lines)


@mcp.tool()
def template_delete(app: str, name: str) -> str:
    """テンプレートを削除する。

    Args:
        app: アプリ名
        name: テンプレート名
    """
    if _template_matcher.delete_template(app, name):
        return f"テンプレート '{app}/{name}' を削除しました"
    return f"テンプレート '{app}/{name}' が見つかりません"


@mcp.tool()
def template_learn(
    target: str,
    app: str,
    name: str,
    monitor: int = 3,
    width: int = 1920,
    grid_size: int = 5,
    max_steps: int = 4,
    target_px: int = 60,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """LLM の視覚で画面を段階的にズームし、ターゲットのテンプレートを自動取得・保存する。

    APIもアクセシビリティも不要 — 画面のピクセルだけでUI要素を見つける。
    画面全体を5×5グリッドに分け、LLMが「どのセルにターゲットがあるか」を判定。
    セルを拡大して繰り返し、十分小さくなったらテンプレートとして保存。

    初回は ~5-10秒かかるが、保存後は smart_find で ~120ms で検索可能。

    使用例:
      template_learn(target="歯車アイコンの設定ボタン", app="メモ帳", name="settings_button")
      → LLM が3-4回画面を見て設定ボタンを特定 → テンプレート保存
      → 以降 smart_find("settings", app="メモ帳") で ~120ms で座標取得

    Args:
        target: 探すUI要素の説明（自然言語）。例: "歯車アイコンの設定ボタン", "赤い丸の録画ボタン"
        app: アプリ名（テンプレート保存先のフォルダ名）
        name: テンプレート名（英数字推奨）
        monitor: スクリーンショット撮影用モニター番号（0=全画面）
        width: スクリーンショットのリサイズ幅
        grid_size: グリッド分割数（3-5、大きいほど少ないステップで収束）
        max_steps: 最大ステップ数
        target_px: 目標セルサイズ（物理ピクセル）。これ以下で停止
        model: 使用する Claude モデル（速度優先なら Sonnet 推奨）
    """
    from agent.template_matcher import learn_template

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # フォールバック: ~/.claude.json の MCP env から読む
        try:
            claude_json = Path.home() / ".claude.json"
            if claude_json.exists():
                import json as _json
                cfg = _json.loads(claude_json.read_text(encoding="utf-8"))
                api_key = (cfg.get("mcpServers", {})
                           .get("video2ai-desktop", {})
                           .get("env", {})
                           .get("ANTHROPIC_API_KEY", ""))
        except Exception:
            pass
    if not api_key:
        return "エラー: ANTHROPIC_API_KEY が見つかりません（環境変数 or ~/.claude.json）"

    # スクリーンショット撮影（画面全体）
    cap = ScreenCapture(monitor_index=monitor, resize_width=width)
    img = cap.capture()
    _last_screenshot_info[monitor] = (img.shape[1], img.shape[0])

    # モニター物理幅を取得
    mon_phys_w = 3840  # デフォルト
    try:
        import mss
        with mss.mss() as sct:
            if 0 < monitor < len(sct.monitors):
                mon_phys_w = sct.monitors[monitor]['width']
    except Exception:
        pass

    result = learn_template(
        screenshot=img,
        target_description=target,
        app=app,
        name=name,
        api_key=api_key,
        matcher=_template_matcher,
        grid_size=grid_size,
        max_steps=max_steps,
        target_px=target_px,
        model=model,
        monitor_phys_w=mon_phys_w,
        screenshot_w=img.shape[1],
    )

    _log_action("template_learn", target=f"{app}/{name}",
                 value=f"target='{target}' steps={result['steps']}",
                 success=result["success"], error=result.get("error", ""))

    if result["success"]:
        rx, ry, rw, rh = result["region"]
        return (
            f"テンプレート学習成功!\n"
            f"  保存先: {result['path']}\n"
            f"  ステップ: {result['steps']}回 ({result['elapsed']:.1f}秒)\n"
            f"  経路: {' → '.join(result['history'])}\n"
            f"  領域: ({rx},{ry}) {rw}x{rh}px\n"
            f"  → smart_find(\"{name}\", app=\"{app}\") で検索可能"
        )
    return (
        f"テンプレート学習失敗\n"
        f"  エラー: {result['error']}\n"
        f"  ステップ: {result['steps']}回 ({result['elapsed']:.1f}秒)\n"
        f"  経路: {' → '.join(result['history'])}"
    )


# ---------------------------------------------------------------------------
# UIA 座標ガイドのテンプレート学習（方法1）
# ---------------------------------------------------------------------------

@mcp.tool()
def template_learn_uia(
    app: str,
    element_name: str = "",
    control_type: str = "",
    template_name: str = "",
    padding: int = 8,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """ScreenWatcher（UI Automation）の座標を使ってテンプレートを正確に学習する。

    20px 未満の小さなアイコンやボタンも確実にキャプチャできる。
    ScreenWatcher が起動中であること（watcher_start 済み）が必要。

    使用例:
      watcher_start("メモ帳")
      template_learn_uia(app="メモ帳", element_name="設定")
      template_learn_uia(app="メモ帳", element_name="表", control_type="Button")
      → UI Automation で座標取得 → フル解像度で切り出し → 検証 → 保存
      → 以降 smart_find("settings", app="メモ帳") で検索可能

    Args:
        app: アプリ名（テンプレート保存先）
        element_name: UI要素名（watcher_find で検索、部分一致）
        control_type: コントロール型フィルタ（例: "Button", "MenuItem"）
        template_name: テンプレート名（省略時は element_name を英語変換）
        padding: 切り出し時の周囲余白ピクセル
        model: 検証用 Claude モデル
    """
    from agent.template_matcher import learn_template_from_uia

    # ScreenWatcher チェック
    if not _screen_watcher or not _screen_watcher.is_running():
        return "エラー: ScreenWatcher が動作していません。watcher_start() で開始してください"

    # API キー取得
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            claude_json = Path.home() / ".claude.json"
            if claude_json.exists():
                import json as _json
                cfg = _json.loads(claude_json.read_text(encoding="utf-8"))
                api_key = (cfg.get("mcpServers", {})
                           .get("video2ai-desktop", {})
                           .get("env", {})
                           .get("ANTHROPIC_API_KEY", ""))
        except Exception:
            pass
    if not api_key:
        return "エラー: ANTHROPIC_API_KEY が見つかりません"

    # watcher_find で要素検索
    elements = _screen_watcher.find_elements(name=element_name,
                                             control_type=control_type)
    if not elements:
        return f"エラー: UI要素 '{element_name}' (type={control_type or 'any'}) が見つかりません"

    el = elements[0]  # 最初にマッチした要素
    description = f"{el.control_type}: {el.name}"

    # テンプレート名の決定
    if not template_name:
        # LLM で英語名を生成する代わりに簡易変換
        template_name = (el.name.replace(" ", "_").replace("　", "_")
                        .lower().strip("_"))
        # ASCII以外を含む場合はコントロール型+ハッシュで命名
        if not template_name.isascii() or not template_name:
            import hashlib
            h = hashlib.md5(el.name.encode()).hexdigest()[:6]
            template_name = f"{el.control_type.lower()}_{h}"

    result = learn_template_from_uia(
        element_center=(el.center_x, el.center_y),
        element_size=(el.width, el.height),
        target_description=description,
        app=app,
        name=template_name,
        api_key=api_key,
        matcher=_template_matcher,
        padding=padding,
        model=model,
    )

    _log_action("template_learn_uia", target=f"{app}/{template_name}",
                value=f"element='{el.name}' size={el.width}x{el.height}",
                success=result["success"], error=result.get("error", ""))

    if result["success"]:
        w, h = result["size"]
        return (
            f"テンプレート学習成功 (UIA)\n"
            f"  要素: [{el.control_type}] \"{el.name}\" ({el.width}x{el.height}px)\n"
            f"  保存先: {result['path']}\n"
            f"  サイズ: {w}x{h}px ({result['elapsed']:.1f}秒)\n"
            f"  → smart_find(\"{template_name}\", app=\"{app}\") で検索可能"
        )
    return (
        f"テンプレート学習失敗 (UIA)\n"
        f"  要素: [{el.control_type}] \"{el.name}\" ({el.width}x{el.height}px)\n"
        f"  エラー: {result['error']}"
    )


# ---------------------------------------------------------------------------
# ツールバー自動分割のテンプレート学習（方法2）
# ---------------------------------------------------------------------------

@mcp.tool()
def template_learn_toolbar(
    app: str,
    toolbar_x: int,
    toolbar_y: int,
    toolbar_w: int,
    toolbar_h: int,
    monitor: int = 3,
    max_icons: int = 20,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """ツールバー領域からアイコンを自動分割して一括学習する。

    UIA 非対応アプリ向け。フル解像度でキャプチャしてツールバー領域を
    OpenCV でアイコンに分割し、LLM で各アイコンを同定してテンプレート保存。
    20px 未満の小さなアイコンも検出可能。

    使用例:
      # まず screenshot でツールバーの大まかな座標を確認
      screenshot(monitor=3, width=1920)
      # ツールバーの座標を指定（スクリーンショット座標ではなく物理座標）
      template_learn_toolbar(app="PhotoShop", toolbar_x=0, toolbar_y=50,
                             toolbar_w=40, toolbar_h=600, monitor=3)

    Args:
        app: アプリ名（テンプレート保存先）
        toolbar_x: ツールバー左端の物理 X 座標
        toolbar_y: ツールバー上端の物理 Y 座標
        toolbar_w: ツールバー幅（物理ピクセル）
        toolbar_h: ツールバー高さ（物理ピクセル）
        monitor: モニター番号
        max_icons: 最大学習アイコン数
        model: 使用する Claude モデル
    """
    from agent.template_matcher import learn_template_toolbar

    # API キー取得
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            claude_json = Path.home() / ".claude.json"
            if claude_json.exists():
                import json as _json
                cfg = _json.loads(claude_json.read_text(encoding="utf-8"))
                api_key = (cfg.get("mcpServers", {})
                           .get("video2ai-desktop", {})
                           .get("env", {})
                           .get("ANTHROPIC_API_KEY", ""))
        except Exception:
            pass
    if not api_key:
        return "エラー: ANTHROPIC_API_KEY が見つかりません"

    # フル解像度でツールバー領域をキャプチャ
    import mss
    import numpy as np
    try:
        with mss.mss() as sct:
            if 0 < monitor < len(sct.monitors):
                mon = sct.monitors[monitor]
                # 物理座標をモニター原点基準に変換
                abs_x = mon["left"] + toolbar_x
                abs_y = mon["top"] + toolbar_y
            else:
                abs_x = toolbar_x
                abs_y = toolbar_y

            region = {"left": abs_x, "top": abs_y,
                      "width": toolbar_w, "height": toolbar_h}
            grab = sct.grab(region)
            screenshot = np.array(grab)[:, :, :3].copy()
    except Exception as e:
        return f"エラー: ツールバーキャプチャ失敗: {e}"

    # app_knowledge を読み込み（アイコン名の精度向上）
    ak_text = _find_app_knowledge(app)

    result = learn_template_toolbar(
        screenshot=screenshot,
        toolbar_rect=(0, 0, toolbar_w, toolbar_h),
        app=app,
        api_key=api_key,
        matcher=_template_matcher,
        max_icons=max_icons,
        app_knowledge=ak_text,
        model=model,
    )

    _log_action("template_learn_toolbar",
                target=f"{app} toolbar ({toolbar_w}x{toolbar_h})",
                value=f"icons={result['total_icons']} learned={len(result['learned'])}",
                success=result["success"],
                error=result.get("error", ""))

    lines = [f"ツールバー学習: {app}"]
    lines.append(f"  領域: ({toolbar_x},{toolbar_y}) {toolbar_w}x{toolbar_h}px")
    lines.append(f"  検出アイコン: {result['total_icons']}個")
    lines.append(f"  時間: {result['elapsed']:.1f}秒")

    if result["learned"]:
        lines.append(f"\n✓ 学習成功 ({len(result['learned'])}件):")
        for name in result["learned"]:
            lines.append(f"  - {name}")

    if result["failed"]:
        lines.append(f"\n✗ 失敗 ({len(result['failed'])}件):")
        for name in result["failed"]:
            lines.append(f"  - {name}")

    if not result["success"] and result.get("error"):
        lines.append(f"\nエラー: {result['error']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# テンプレート一括自動学習
# ---------------------------------------------------------------------------

def _find_app_knowledge(app: str) -> str:
    """app_knowledge ディレクトリからアプリのマークダウンを探して読み込む。"""
    knowledge_dir = Path("D:/ClaudeProject/app_knowledge")
    if not knowledge_dir.exists():
        return ""
    candidates = [
        app,
        app.replace(" ", "_"),
        app.replace(" ", "_").lower(),
        app.lower(),
        app.replace(" ", "").lower(),
    ]
    for name in candidates:
        path = knowledge_dir / f"{name}.md"
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass
    return ""


@mcp.tool()
def template_auto_learn(
    app: str = "",
    window: str = "",
    monitor: int = 3,
    width: int = 1920,
    max_elements: int = 10,
    force: bool = False,
    grid_size: int = 5,
    max_steps: int = 4,
    target_px: int = 60,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """アプリの主要UIを一括でテンプレート学習する。
    スクリーンショットをLLMに見せて主要ボタン・ツール等を特定し、
    各要素に対して template_learn を順次実行する。

    Args:
        app: アプリ名（空ならフォアグラウンドウィンドウから推定）
        window: ウィンドウタイトルのキーワード。指定するとそのウィンドウだけをキャプチャ
        monitor: スクリーンショット撮影用モニター番号（window未指定時に使用）
        width: スクリーンショットのリサイズ幅
        max_elements: 学習する要素の最大数（API コスト制御）
        force: True なら既存テンプレートを再学習
        grid_size: グリッド分割数
        max_steps: 各要素の最大ステップ数
        target_px: 目標セルサイズ（物理ピクセル）
        model: 使用する Claude モデル
    """
    from agent.template_matcher import auto_learn_templates

    # API キー解決
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            claude_json = Path.home() / ".claude.json"
            if claude_json.exists():
                import json as _json
                cfg = _json.loads(claude_json.read_text(encoding="utf-8"))
                api_key = (cfg.get("mcpServers", {})
                           .get("video2ai-desktop", {})
                           .get("env", {})
                           .get("ANTHROPIC_API_KEY", ""))
        except Exception:
            pass
    if not api_key:
        return "エラー: ANTHROPIC_API_KEY が見つかりません"

    # アプリ名の推定
    if not app:
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
            title = buf.value
            if " - " in title:
                app = title.split(" - ")[-1].strip()
            elif title:
                app = title
            else:
                return "エラー: フォアグラウンドウィンドウが取得できません"
        except Exception as e:
            return f"エラー: ウィンドウタイトル取得失敗 — {e}"

    # スクリーンショット（ウィンドウ指定 or モニター全体）
    mon_phys_w = 3840
    if window:
        import ctypes
        import ctypes.wintypes
        import mss
        user32 = ctypes.windll.user32
        # ウィンドウ検索
        _found_hwnd = [None]
        def _enum_cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(512)
                user32.GetWindowTextW(hwnd, buf, 512)
                if window in buf.value:
                    _found_hwnd[0] = hwnd
                    return False
            return True
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)

        if _found_hwnd[0]:
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(_found_hwnd[0], ctypes.byref(rect))
            wl, wt = rect.left, rect.top
            ww, wh = rect.right - rect.left, rect.bottom - rect.top
            if wl < 0: ww += wl; wl = 0
            if wt < 0: wh += wt; wt = 0
            cap = ScreenCapture(region=(wl, wt, ww, wh), resize_width=width)
            img = cap.capture()
            mon_phys_w = ww  # ウィンドウ幅を物理幅として使用
            # app 名が空なら window タイトルから推定
            if not app:
                buf = ctypes.create_unicode_buffer(512)
                user32.GetWindowTextW(_found_hwnd[0], buf, 512)
                title = buf.value
                app = title.split(" - ")[-1].strip() if " - " in title else title
        else:
            return f"エラー: '{window}' を含むウィンドウが見つかりません"
    else:
        cap = ScreenCapture(monitor_index=monitor, resize_width=width)
        img = cap.capture()
        try:
            import mss
            with mss.mss() as sct:
                mon_phys_w = sct.monitors[monitor]["width"]
        except Exception:
            pass

    if img is None:
        return "エラー: スクリーンショットの撮影に失敗しました"

    # app_knowledge の読み込み
    knowledge = _find_app_knowledge(app)
    knowledge_info = f"app_knowledge: あり ({len(knowledge)}文字)" if knowledge else "app_knowledge: なし"

    # 一括学習実行
    result = auto_learn_templates(
        screenshot=img,
        app=app,
        api_key=api_key,
        matcher=_template_matcher,
        app_knowledge=knowledge,
        max_elements=max_elements,
        force=force,
        grid_size=grid_size,
        max_steps=max_steps,
        target_px=target_px,
        model=model,
        monitor_phys_w=mon_phys_w,
        screenshot_w=img.shape[1],
    )

    _log_action("template_auto_learn", target=app,
                value=f"identified={result['total_identified']} learned={len(result['learned'])}",
                success=result["success"])

    # 結果フォーマット
    lines = [f"テンプレート一括学習: {app}"]
    lines.append(f"  {knowledge_info}")
    lines.append(f"  特定: {result['total_identified']}要素, "
                 f"スキップ: {result['skipped_existing']}, "
                 f"学習: {len(result['learned'])}, "
                 f"失敗: {len(result['failed'])}")
    lines.append(f"  合計時間: {result['total_elapsed']:.1f}秒")

    if result["learned"]:
        lines.append("\n✓ 学習成功:")
        for item in result["learned"]:
            lines.append(f"  - {item['name']}: {item['description']} "
                        f"({item['steps']}steps, {item['elapsed']:.1f}s)")

    if result["failed"]:
        lines.append("\n✗ 学習失敗:")
        for item in result["failed"]:
            lines.append(f"  - {item['name']}: {item['error']}")

    if result["skipped_existing"] > 0:
        lines.append(f"\n（既存テンプレート {result['skipped_existing']}個をスキップ。force=True で再学習可）")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Video2AI 動画解析ツール
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_video(
    video_path: str,
    output_dir: str,
    sync_map: bool = True,
    output_width: int = 960,
    jpeg_quality: int = 80,
    threshold: float = 0.02,
    min_interval: float = 0.5,
    whisper_model: str = "base",
    language: str = "",
    privacy_blur_faces: bool = False,
    privacy_redact_text: bool = False,
    privacy_strip_raw: bool = False,
) -> str:
    """Video2AI で動画を解析する。スクリーンショット抽出 + 音声文字起こし + 同期マップ生成。

    Args:
        video_path: 入力動画ファイルのパス
        output_dir: 出力ディレクトリのパス
        sync_map: 同期マップ (フレーム↔発話の対応) を生成するか
        output_width: 出力画像の幅 (px)
        jpeg_quality: JPEG 品質 (1-95)
        threshold: 変化検知の閾値 (0-1, 小さいほど敏感)
        min_interval: 最小キャプチャ間隔 (秒)
        whisper_model: Whisper モデル (tiny/base/small/medium/large)
        language: 音声言語コード (例: "ja", "en")。空欄で自動検出
        privacy_blur_faces: 抽出フレーム内の顔を検出してぼかす
        privacy_redact_text: 抽出フレーム内の個人情報テキストを墨消しする
        privacy_strip_raw: フィルタ適用後に元のフレーム画像を削除する（フィルタ済みのみ保持）
    """
    cmd = [
        sys.executable, str(Path(_ROOT) / "main.py"),
        video_path, output_dir,
        "--threshold", str(threshold),
        "--min-interval", str(min_interval),
        "--jpeg-quality", str(jpeg_quality),
        "--output-width", str(output_width),
        "--whisper-model", whisper_model,
    ]
    if sync_map:
        cmd.append("--sync-map")
    if language:
        cmd.extend(["--language", language])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=_ROOT, timeout=600)
        output = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0:
            return f"解析失敗 (exit code {result.returncode})。\n\n{output}"

        # プライバシーフィルタ: フレーム抽出後に後処理
        if privacy_blur_faces or privacy_redact_text:
            from agent.privacy_filters import apply_privacy_filters
            frames_dir = Path(output_dir) / "frames"
            if frames_dir.exists():
                filtered_count = 0
                for img_path in sorted(frames_dir.glob("*.jpg")) + \
                                sorted(frames_dir.glob("*.png")):
                    pil_img = PILImage.open(img_path)
                    filtered = apply_privacy_filters(
                        pil_img,
                        blur_faces_enabled=privacy_blur_faces,
                        redact_text_enabled=privacy_redact_text,
                    )
                    filtered.save(img_path)
                    filtered_count += 1
                output += f"\n\nプライバシーフィルタ適用: {filtered_count}フレーム"

        # privacy_strip_raw: 元動画のコピーを出力ディレクトリから削除
        if privacy_strip_raw:
            for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"):
                for raw_file in Path(output_dir).glob(ext):
                    raw_file.unlink()
            output += "\n生データ削除済み（フィルタ済みフレームのみ保持）"

        return f"解析完了。\n\n{output}"
    except subprocess.TimeoutExpired:
        return "タイムアウト: 動画解析が10分以内に完了しませんでした"
    except Exception as e:
        return f"エラー: {e}"


# ---------------------------------------------------------------------------
# Monte Carlo Pinpoint — グリッド追い込み＋ランダムドットで座標特定
# ---------------------------------------------------------------------------


def _pinpoint_llm_call(
    b64_image: str,
    system: str,
    user: str,
    api_key: str,
    model: str,
    max_tokens: int = 10,
) -> str | None:
    """pinpoint 用の汎用 LLM 呼出し。テキスト応答を返す。"""
    import requests as _req

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64_image,
                    },
                },
                {"type": "text", "text": user},
            ],
        }],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers, timeout=30,
        )
        if resp.status_code != 200:
            logging.warning("pinpoint LLM API error: %s %s",
                            resp.status_code, resp.text[:200])
            return None
        text = resp.json()["content"][0]["text"].strip()
        logging.warning("pinpoint LLM response: %r", text)
        return text
    except Exception as e:
        logging.warning("pinpoint LLM API failed: %s", e)
        return None


_PINPOINT_ANCHOR_SYSTEM = (
    "You are a pixel-precision visual inspector. "
    "You will be asked whether a target is visible in a zoomed image. "
    "If it is, declare EXACTLY which pixel you consider the correct "
    "target point — not 'the gear icon' but 'the exact center pixel "
    "of the gear icon's hub circle'. "
    "If the target is NOT clearly present in the image, you must "
    "explicitly report its absence instead of inventing a location. "
    "This declaration is the strict standard for all subsequent judgments."
)


def _pinpoint_declare_anchor(
    b64_image: str,
    target: str,
    api_key: str,
    model: str,
) -> tuple[str, bool]:
    """Phase 1.5: LLM にターゲットの存在検証と狙い位置宣言を同時にさせる。

    GR-0 Anchor Validity Check: Mode B (Phase-1 が wrong neighborhood に
    着地) の cascading hallucination を declaration 時点で検出する。
    "Is target visible" を明示的に聞くことで VLM が description に
    commit する前に absence を catch する。faithfulness bias を断つ。

    Returns:
        (anchor_text, visible)
        - visible=True: ターゲットが見える → anchor_text は target pixel 記述
        - visible=False: ターゲット不在 → anchor_text は見えたものの記述
    """
    user = (
        f"Target: {target}\n\n"
        "Is this target clearly visible in the image? "
        "Respond in this EXACT format:\n"
        "VISIBLE: <one sentence specifying the exact target pixel "
        "(center, edge, or feature point)>\n"
        "or\n"
        "ABSENT: <one sentence describing what you see instead>\n\n"
        "Start your response with either 'VISIBLE:' or 'ABSENT:' — "
        "no other prefix. One sentence only."
    )
    text = _pinpoint_llm_call(b64_image, _PINPOINT_ANCHOR_SYSTEM, user,
                              api_key, model, max_tokens=120)
    if not text:
        return (target, True)  # API 失敗時は元の target をそのまま使う
    text = text.strip()
    upper = text.upper()
    if upper.startswith("ABSENT"):
        desc = text.split(":", 1)[1].strip() if ":" in text else text
        return (desc, False)
    if upper.startswith("VISIBLE"):
        desc = text.split(":", 1)[1].strip() if ":" in text else text
        return (desc, True)
    # VLM が拒否応答で返すケース (旧パス互換): ABSENT 扱い
    if (upper.startswith("I CANNOT") or upper.startswith("I DO NOT")
            or upper.startswith("I CAN'T") or upper.startswith("I'M UNABLE")):
        return (text, False)
    # 非構造応答: visible として扱う (下位互換)
    return (text, True)


def _pinpoint_verify_anchor(
    b64_image: str,
    target: str,
    original_anchor: str,
    api_key: str,
    model: str,
) -> tuple[str, bool]:
    """アンカーの一貫性チェック: 同じ画像でアンカーを再宣言させ、初回と比較する。

    Returns:
        (new_anchor, is_consistent)
        - new_anchor: 再宣言されたアンカー
        - is_consistent: 初回アンカーと意味的に一致するか
    """
    # Step 1: 再宣言 (visible フラグは consistency 判定では使わない)
    new_anchor, _ = _pinpoint_declare_anchor(b64_image, target, api_key, model)

    # Step 2: 一貫性を LLM に判定させる（アンカー同士の意味比較）
    system = (
        "You compare two descriptions of a target point in an image. "
        "Reply ONLY with:\n"
        "  SAME = both describe the same pixel-level target point\n"
        "  DIFFERENT = they describe different points\n"
        "Output ONLY the code."
    )
    user = (
        f"Description A: {original_anchor}\n"
        f"Description B: {new_anchor}\n\n"
        "Are they targeting the same pixel? Answer SAME or DIFFERENT only:"
    )
    result = _pinpoint_llm_call(b64_image, system, user, api_key, model,
                                max_tokens=10)
    is_consistent = result and "SAME" in result.upper() if result else True
    return (new_anchor, is_consistent)


def _pinpoint_ask_dots(
    b64_image: str,
    anchor: str,
    num_dots: int,
    api_key: str,
    model: str,
) -> int | None:
    """LLM にドット付き画像を見せて、アンカー位置上のドット番号を聞く。

    anchor: _pinpoint_declare_anchor で LLM が自己宣言した狙い位置の記述

    Returns:
        ドット番号 (1-based) or None (どのドットもターゲット上にない)
    """
    system = (
        "You are a pixel-precision visual inspector. "
        f"The image shows a highly zoomed-in area with {num_dots} small "
        "numbered colored dots (tiny circles with number labels). "
        "You previously declared the exact target point as:\n"
        f'  "{anchor}"\n\n'
        "TASK: Identify which numbered dot is CLOSEST to the declared "
        "target point. There is always a closest dot — you must pick one.\n\n"
        "Reply with ONLY: D<number> (e.g. D3)\n"
        "Output ONLY the code. No explanation."
    )
    user = "Which dot is closest to the target? Answer D<number>:"

    text = _pinpoint_llm_call(b64_image, system, user, api_key, model,
                              max_tokens=10)
    if not text:
        return None
    text = text.upper()
    import re
    m = re.match(r"D(\d+)", text)
    if m:
        num = int(m.group(1))
        if 1 <= num <= num_dots:
            return num
    return None


def _pinpoint_ask_dots_majority(
    b64_image: str,
    anchor: str,
    num_dots: int,
    api_key: str,
    model: str,
    votes: int = 3,
) -> tuple[int | None, bool]:
    """多数決でドット判定の信頼性を検証する。

    同じ画像を votes 回 LLM に聞き、過半数が一致した回答のみ採用する。
    ハルシネーション（偶発的な誤判定）を排除するためのガードレール。

    Returns:
        (dot_number_or_None, is_unanimous)
        - dot_number: 過半数が合意したドット番号。合意なしは None
        - is_unanimous: 全票一致か（ログ用）
    """
    from collections import Counter

    results = []
    for _ in range(votes):
        hit = _pinpoint_ask_dots(b64_image, anchor, num_dots, api_key, model)
        results.append(hit)

    counter = Counter(results)
    majority_threshold = votes // 2 + 1  # 3 votes → need 2+

    (most_common, count), = counter.most_common(1)
    if count >= majority_threshold:
        return (most_common, count == votes)

    # 合意なし: ハルシネーションの可能性が高いため棄却
    logging.info(
        f"pinpoint majority: 合意なし votes={results} → 棄却")
    return (None, False)


# ドットの色パレット（番号→色、視認性重視）
_PINPOINT_COLORS = [
    (255, 0, 0),      # 1: 赤
    (0, 200, 0),      # 2: 緑
    (0, 100, 255),    # 3: 青
    (255, 165, 0),    # 4: オレンジ
    (255, 0, 255),    # 5: マゼンタ
    (0, 200, 200),    # 6: シアン
    (255, 255, 0),    # 7: 黄
    (128, 0, 255),    # 8: 紫
    (255, 100, 100),  # 9: ライトレッド
    (100, 255, 100),  # 10: ライトグリーン
]


@mcp.tool()
@_preserves_screenshot_state
def pinpoint(
    target: str,
    monitor: int = 3,
    width: int = 1920,
    window: str = "",
    num_dots: int = 7,
    max_rounds: int = 8,
    grid_precision_px: int = 20,
    precision: int = 5,
    model: str = "claude-sonnet-4-20250514",
) -> Image:
    """グリッド追い込み＋ランダムドットでピクセル精度の座標を特定する。

    Phase 1: grid_find で ±grid_precision_px の領域に追い込む。
    Phase 2: その領域を拡大し、ランダムに番号付きドットを配置。
             LLM にどのドットがターゲット上にあるか聞く。
             一致するまで繰り返す（モンテカルロ探索）。
    Phase 3: (precision < 5 の場合のみ) Phase 2 の HIT 点を中心に
             極小領域を超拡大し、DOT_RADIUS=1 で再探索。
             precision=1 で理論上1px精度に到達する。

    Args:
        target: 探したい場所の説明（例: "キャラクターの鼻の先端"）
        monitor: モニター番号
        width: スクリーンショットのリサイズ幅
        window: ウィンドウタイトルのキーワード（部分一致、指定推奨）
        num_dots: 1ラウンドあたりのドット数（デフォルト7）
        max_rounds: 最大ラウンド数（デフォルト8）
        grid_precision_px: Phase 1 の目標精度（物理ピクセル、デフォルト20）
        precision: 最終目標精度（スクリーンショットピクセル、デフォルト5）。
                   5=Phase 2 で終了（高速）、1=Phase 3 追い込みで最高精度。
        model: 使用する Claude モデル
    """
    import time as _time
    import random
    import re as _re
    import ctypes
    import ctypes.wintypes

    t0 = _time.time()

    # API キー取得
    api_key = _grid_find_get_api_key()
    if not api_key:
        return "エラー: ANTHROPIC_API_KEY が見つかりません"

    import cv2 as _cv2
    from PIL import Image as PILImage, ImageDraw, ImageFont
    import mss

    # ---------------------------------------------------------------
    # Phase 1: grid_find で追い込み
    # ---------------------------------------------------------------
    # grid_find の内部ロジックを再利用して座標と領域を取得
    win_offset_x, win_offset_y = 0, 0
    mon_phys_w, mon_phys_h = 0, 0

    if window:
        user32 = ctypes.windll.user32
        target_hwnd = None

        def enum_cb(hwnd, _):
            nonlocal target_hwnd
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if window in buf.value:
                    target_hwnd = hwnd
                    return False
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        if not target_hwnd:
            return f"pinpoint: ウィンドウ「{window}」が見つかりません"

        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(target_hwnd, ctypes.byref(rect))
        win_left = max(rect.left, 0)
        win_top = max(rect.top, 0)
        win_w = rect.right - rect.left
        win_h = rect.bottom - rect.top
        if rect.left < 0:
            win_w += rect.left
        if rect.top < 0:
            win_h += rect.top

        client_rect = ctypes.wintypes.RECT()
        user32.GetClientRect(target_hwnd, ctypes.byref(client_rect))
        client_pt = ctypes.wintypes.POINT(0, 0)
        user32.ClientToScreen(target_hwnd, ctypes.byref(client_pt))
        titlebar_h = client_pt.y - rect.top
        if titlebar_h <= 0:
            titlebar_h = user32.GetSystemMetrics(4)  # SM_CYCAPTION
        win_top += titlebar_h
        win_h -= titlebar_h
        win_offset_x, win_offset_y = win_left, win_top

        with mss.mss() as sct:
            for i in range(1, len(sct.monitors)):
                mon = sct.monitors[i]
                cx = win_left + win_w // 2
                cy = win_top + win_h // 2
                if (mon['left'] <= cx < mon['left'] + mon['width'] and
                        mon['top'] <= cy < mon['top'] + mon['height']):
                    monitor = i
                    break

        cap = ScreenCapture(
            region=(win_left, win_top, win_w, win_h), resize_width=0)
        img_bgr = cap.capture()
        native_w, native_h = img_bgr.shape[1], img_bgr.shape[0]
        mon_phys_w, mon_phys_h = win_w, win_h

        upscale_w = max(width, native_w) if width > 0 else native_w
        if upscale_w > native_w:
            scale = upscale_w / native_w
            upscale_h = int(native_h * scale)
            img_bgr = _cv2.resize(img_bgr, (upscale_w, upscale_h),
                                  interpolation=_cv2.INTER_CUBIC)
        shot_w, shot_h = img_bgr.shape[1], img_bgr.shape[0]
    else:
        cap = ScreenCapture(monitor_index=monitor, resize_width=width)
        img_bgr = cap.capture()
        shot_w, shot_h = img_bgr.shape[1], img_bgr.shape[0]
        _last_screenshot_info[monitor] = (shot_w, shot_h)
        mon_phys_w, mon_phys_h = shot_w, shot_h
        try:
            with mss.mss() as sct:
                if monitor < len(sct.monitors):
                    m = sct.monitors[monitor]
                    mon_phys_w, mon_phys_h = m["width"], m["height"]
        except Exception:
            pass

    pil_img = PILImage.fromarray(_cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB))

    # X 軸探索
    x_shot, x_phys, x_log = _grid_find_search_axis(
        pil_img, target, "x", 8, grid_precision_px,
        api_key, model, shot_w, mon_phys_w,
    )
    phase1_x_ok = x_shot >= 0

    # Y 軸探索
    if phase1_x_ok:
        slice_half = max(shot_w // 6, 200)
        slice_left = max(0, x_shot - slice_half)
        slice_right = min(shot_w, x_shot + slice_half)
        y_slice = pil_img.crop((slice_left, 0, slice_right, shot_h))
    else:
        y_slice = pil_img
    y_shot, y_phys, y_log = _grid_find_search_axis(
        y_slice, target, "y", 8, grid_precision_px,
        api_key, model, shot_h, mon_phys_h,
    )
    phase1_y_ok = y_shot >= 0

    # Fallback: if grid failed on either axis, use direct VLM estimation
    if not phase1_x_ok or not phase1_y_ok:
        b64_full = _grid_find_to_base64(pil_img)
        direct_sys = (
            f"You are a precise visual coordinate estimator. "
            f"The image is {shot_w}x{shot_h} pixels. "
            f"Given a target description, output the exact pixel coordinate. "
            f"Reply with ONLY: x,y (two integers separated by comma). "
            f"No explanation."
        )
        direct_text = _grid_find_vlm_call(
            b64_full, direct_sys,
            f"Target: {target}\nCoordinate (x,y):",
            api_key, model)
        dx, dy = None, None
        if direct_text:
            dm = _re.match(r"(\d+)\s*,\s*(\d+)", direct_text)
            if dm:
                dx, dy = int(dm.group(1)), int(dm.group(2))
        if not phase1_x_ok:
            x_shot = dx if dx is not None else shot_w // 2
            x_phys = int(x_shot * mon_phys_w / shot_w) if mon_phys_w > 0 else x_shot
        if not phase1_y_ok:
            y_shot = dy if dy is not None else shot_h // 2
            y_phys = int(y_shot * mon_phys_h / shot_h) if mon_phys_h > 0 else y_shot

    phase1_time = _time.time() - t0

    # ---------------------------------------------------------------
    # Stage B: Post-Phase 1 OCR-snap (shared helper, decision 049)
    # ---------------------------------------------------------------
    # Single source of truth with benchmark/run_benchmark.py via
    # benchmark.stage_b.apply_stage_b_snap. miss 経路は byte-identical。
    from benchmark.stage_b import apply_stage_b_snap as _apply_stage_b_snap
    _sb = _apply_stage_b_snap(
        pil_img, x_shot, y_shot, phase1_x_ok, phase1_y_ok,
        target, shot_w, shot_h,
    )
    stage_b_hit = bool(_sb["hit"])
    stage_b_match_text = _sb["match"]
    stage_b_snap_dist = float(_sb["dist"])
    stage_b_time = float(_sb["time"])

    # Stage B HIT: terminal 座標を事前確定、以降の Phase 1.5/2/3 を structurally skip
    if stage_b_hit:
        found_sx, found_sy = int(_sb["x"]), int(_sb["y"])
        max_rounds = 0  # Phase 2 for-loop is a no-op
    # -- end Stage B --

    # Phase 1 の結果: スクリーンショット座標 (x_shot, y_shot), 精度 ±grid_precision_px
    # 探索領域: Phase 1 の median error (61px) をカバーするため sr=100px に拡大。
    # 8 rounds の半減で 100/256 = 0.39px まで収束するので精度は維持。
    search_radius_shot = max(100, int(grid_precision_px * shot_w / mon_phys_w) + 2)
    # Phase 1 が失敗した軸がある場合はさらに拡大
    if not phase1_x_ok or not phase1_y_ok:
        search_radius_shot = search_radius_shot * 2
    # コンテキスト範囲: 探索範囲の2倍（LLM がターゲットを認識するのに十分な周辺情報）
    context_radius_shot = search_radius_shot * 2

    # ---------------------------------------------------------------
    # Phase 1.5: アンカー宣言（ガードレール）
    # ---------------------------------------------------------------
    # 追い込んだ領域を LLM に見せて、正確な狙い位置を自己宣言させる。
    # この宣言が Phase 2 の全ラウンドで判定基準として使われる。
    num_dots = min(num_dots, len(_PINPOINT_COLORS))
    log_lines = [
        f"Phase 1: ({x_shot},{y_shot}) ±{grid_precision_px}px [{phase1_time:.1f}s]"
    ]
    if stage_b_hit:
        log_lines.append(
            f"Stage B: OCR-snap HIT '{stage_b_match_text}' "
            f"d={stage_b_snap_dist:.1f}px → ({found_sx},{found_sy}) "
            f"[{stage_b_time:.2f}s]"
        )
    else:
        log_lines.append(f"Stage B: OCR-snap MISS [{stage_b_time:.2f}s]")
    # Claude Sonnet は 1:1 アスペクトで ~1092x1092 を超える画像を silent
    # downsample する (official "1568px rule")。1024 は情報ロスなしの安全値。
    ZOOM_SIZE = 1024

    # コンテキスト範囲でクロップ（アンカー宣言用 — 広い範囲で認識可能にする）
    ctx_x1 = max(0, x_shot - context_radius_shot)
    ctx_y1 = max(0, y_shot - context_radius_shot)
    ctx_x2 = min(shot_w, x_shot + context_radius_shot)
    ctx_y2 = min(shot_h, y_shot + context_radius_shot)

    # ドット配置範囲（狭い精度範囲）
    crop_x1 = max(0, x_shot - search_radius_shot)
    crop_y1 = max(0, y_shot - search_radius_shot)
    crop_x2 = min(shot_w, x_shot + search_radius_shot)
    crop_y2 = min(shot_h, y_shot + search_radius_shot)
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    ctx_w = ctx_x2 - ctx_x1
    ctx_h = ctx_y2 - ctx_y1

    # Phase 1.5 anchor declaration (skipped on Stage B hit)
    phase_1_5_t0 = _time.time()
    anchor = ""
    anchor_b64 = b""
    original_anchor = ""
    ANCHOR_CHECK_INTERVAL = 4  # 4ラウンドごとにアンカー一貫性チェック
    if not stage_b_hit:
        anchor_cropped = pil_img.crop((ctx_x1, ctx_y1, ctx_x2, ctx_y2))
        anchor_scale = min(ZOOM_SIZE / max(ctx_w, 1), ZOOM_SIZE / max(ctx_h, 1))
        anchor_zoom_w = max(1, int(ctx_w * anchor_scale))
        anchor_zoom_h = max(1, int(ctx_h * anchor_scale))
        anchor_zoomed = anchor_cropped.resize(
            (anchor_zoom_w, anchor_zoom_h), PILImage.Resampling.BICUBIC)
        anchor_b64 = _grid_find_to_base64(anchor_zoomed)

        anchor, visible = _pinpoint_declare_anchor(
            anchor_b64, target, api_key, model)

        # GR-0: ABSENT が返れば Mode B (Phase-1 wrong neighborhood) の可能性が高い。
        # 段階的に広い crop で anchor 宣言をリトライし、どこで target が見えるかを探す。
        # これにより VLM が description に commit する前に absence を catch し、
        # 汚染されたアンカーによる cascading hallucination を断つ。
        if not visible:
            log_lines.append(f"Anchor ABSENT in initial crop: {anchor}")
            # リトライ 1: ~1/3 画面範囲
            wide_r = min(shot_w, shot_h) // 3
            wx1 = max(0, x_shot - wide_r); wy1 = max(0, y_shot - wide_r)
            wx2 = min(shot_w, x_shot + wide_r); wy2 = min(shot_h, y_shot + wide_r)
            ww, wh = wx2 - wx1, wy2 - wy1
            wide_crop = pil_img.crop((wx1, wy1, wx2, wy2))
            wsc = min(ZOOM_SIZE / max(ww, 1), ZOOM_SIZE / max(wh, 1))
            wide_zoomed = wide_crop.resize(
                (max(1, int(ww * wsc)), max(1, int(wh * wsc))),
                PILImage.Resampling.BICUBIC)
            anchor2, visible2 = _pinpoint_declare_anchor(
                _grid_find_to_base64(wide_zoomed), target, api_key, model)
            if visible2:
                anchor = anchor2
                anchor_b64 = _grid_find_to_base64(wide_zoomed)
                log_lines.append(f"Anchor retry (1/3 crop): {anchor}")
            else:
                # リトライ 2: 全画面 (Mode B の最後の救済チャンス)
                full_scale = min(ZOOM_SIZE / max(shot_w, 1),
                                 ZOOM_SIZE / max(shot_h, 1))
                full_zoomed = pil_img.resize(
                    (max(1, int(shot_w * full_scale)),
                     max(1, int(shot_h * full_scale))),
                    PILImage.Resampling.BICUBIC)
                anchor3, visible3 = _pinpoint_declare_anchor(
                    _grid_find_to_base64(full_zoomed), target, api_key, model)
                if visible3:
                    anchor = anchor3
                    anchor_b64 = _grid_find_to_base64(full_zoomed)
                    log_lines.append(f"Anchor retry (full image): {anchor}")
                else:
                    log_lines.append(
                        f"⚠ Anchor ABSENT at all scales — "
                        f"Phase-1 likely wrong. Best-effort: {anchor2 or anchor}")
                    anchor = anchor2 or anchor

        original_anchor = anchor
        log_lines.append(f"Anchor: {anchor}")
    phase_1_5_time = _time.time() - phase_1_5_t0

    # ---------------------------------------------------------------
    # Phase 2: 階層的縮小探索 (deterministic hex stencil + HIT-only halving)
    #
    # Bundle 2.15 / Mode T-2 C-lite restructure (Q-MISS-A2 LOCK):
    # - Replaces random k=7 sampling with a deterministic 7-point hexagonal
    #   half-net per round (pinpoint_core.hex_stencil_dots), which is an
    #   sr/2-cover of the disk of radius sr around the current center.
    # - On a majority HIT, recenters on the chosen dot and halves sr, so
    #   the inductive containment guarantee of paper Theorem 1 holds under
    #   closest-dot semantics.
    # - On MISS (no majority), retries with the SAME (center, sr); a
    #   per-round retry counter caps consecutive MISSes at 3. After the
    #   3rd MISS without a HIT, Phase 2 halts and the algorithm falls
    #   back to the Phase 1 estimate. Halving is HIT-only; this matches
    #   the discrete recurrence proved in Theorem 1/2 of the paper and
    #   avoids the boundary-case containment failure at target offsets
    #   above sr/2 that an unconditional MISS-halving would introduce.
    # - Matches the algorithm in benchmark/run_benchmark.py, which is the
    #   empirically evaluated implementation. A shared primitive
    #   (pinpoint_core) prevents future drift between the two paths.
    # ---------------------------------------------------------------
    phase2_t0 = _time.time()
    if not stage_b_hit:
        found_sx, found_sy = None, None
        center_x, center_y = x_shot, y_shot
        sr = search_radius_shot
        phase2_any_hit = False
    tried_positions: list[tuple[int, int]] = []
    round_idx = -1  # Stage B hit では Phase 2 loop が走らないため事前初期化

    if not stage_b_hit:
        # Outer iteration: each pass is one query batch (HIT advances a
        # logical round; MISS increments the per-round retry counter).
        # Bound = max_rounds * 4 = HIT budget + worst-case retry budget.
        miss_retry_count = 0
        hit_round_count = 0
        for round_idx in range(max_rounds * 4):
            if hit_round_count >= max_rounds:
                break  # HIT budget exhausted
            # 現在の center+sr から探索領域 (disk-clipped to image bounds)
            dx1 = max(0, center_x - sr)
            dy1 = max(0, center_y - sr)
            dx2 = min(shot_w, center_x + sr)
            dy2 = min(shot_h, center_y + sr)
            if dx2 - dx1 < 3 or dy2 - dy1 < 3:
                log_lines.append(
                    f"  round {round_idx + 1}: 探索領域収束 sr={sr}")
                break

            # コンテキスト範囲 (3x sr) で clip + zoom
            ctx_x1_r = max(0, center_x - sr * 3)
            ctx_y1_r = max(0, center_y - sr * 3)
            ctx_x2_r = min(shot_w, center_x + sr * 3)
            ctx_y2_r = min(shot_h, center_y + sr * 3)
            ctx_w_r = ctx_x2_r - ctx_x1_r
            ctx_h_r = ctx_y2_r - ctx_y1_r

            ctx_cropped = pil_img.crop(
                (ctx_x1_r, ctx_y1_r, ctx_x2_r, ctx_y2_r))
            scale = min(ZOOM_SIZE / max(ctx_w_r, 1),
                        ZOOM_SIZE / max(ctx_h_r, 1))
            zoom_w = max(1, int(ctx_w_r * scale))
            zoom_h = max(1, int(ctx_h_r * scale))
            zoomed = ctx_cropped.resize(
                (zoom_w, zoom_h), PILImage.Resampling.BICUBIC)

            # 決定論的ヘックスステンシル (1 中心 + 6 外周 at angles
            # 0°/60°/120°/180°/240°/300°, 半径 (√3/2)*sr)
            all_dots = clip_dots_to_bounds(
                hex_stencil_dots(center_x, center_y, sr),
                dx1, dy1, dx2, dy2,
            )
            dots = all_dots[:num_dots]
            if not dots:
                log_lines.append(
                    f"  round {round_idx + 1}: stencil empty")
                break

            # ドットを拡大画像に描画（小さいドット: 半径2px on zoomed）
            annotated = zoomed.copy()
            draw = ImageDraw.Draw(annotated)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()

            DOT_RADIUS = 2  # 拡大画像上のドット半径（ピクセル精度を保つため小さく）
            for i, (sx, sy) in enumerate(dots):
                # screenshot 座標 → context-crop 内座標 → zoom 座標
                zx = int((sx - ctx_x1_r) * scale)
                zy = int((sy - ctx_y1_r) * scale)
                color = _PINPOINT_COLORS[i % len(_PINPOINT_COLORS)]
                draw.ellipse(
                    [(zx - DOT_RADIUS, zy - DOT_RADIUS),
                     (zx + DOT_RADIUS, zy + DOT_RADIUS)],
                    fill=color, outline=(255, 255, 255), width=1)
                label = str(i + 1)
                draw.text((zx + DOT_RADIUS + 3, zy - 10), label,
                          fill=color, font=font,
                          stroke_fill=(0, 0, 0), stroke_width=2)

            # LLM にアンカー基準で質問（多数決で検証）
            b64 = _grid_find_to_base64(annotated)
            hit, unanimous = _pinpoint_ask_dots_majority(
                b64, anchor, len(dots), api_key, model)

            if hit is not None and 1 <= hit <= len(dots):
                # Q-MISS-A2 LOCK: HIT recenters + halves. Resets the
                # per-round MISS retry counter so subsequent MISSes are
                # measured against the new HIT round.
                center_x, center_y = dots[hit - 1]
                phase2_any_hit = True
                sr = max(2, sr // 2)
                hit_round_count += 1
                miss_retry_count = 0
                vote_info = "unanimous" if unanimous else "majority"
                log_lines.append(
                    f"  HIT round {hit_round_count}: dot {hit} "
                    f"({vote_info}) → center=({center_x},{center_y}), "
                    f"sr={sr}")
            else:
                for sx, sy in dots:
                    tried_positions.append((sx, sy))
                # Q-MISS-A2 LOCK: MISS retries with same (center, sr);
                # halving is HIT-only. The retry counter caps consecutive
                # MISSes at 3 (Theorem 1 anti-loop bound) and then breaks
                # to Phase 1 fallback. This avoids the boundary-case
                # containment failure that unconditional MISS-halving
                # introduces when the target lies in (sr/2, sr] from
                # the current center.
                miss_retry_count += 1
                log_lines.append(
                    f"  MISS retry {miss_retry_count}/3 "
                    f"(HIT round {hit_round_count + 1}, "
                    f"{len(dots)} dots, center=({center_x},{center_y}), "
                    f"sr={sr})")

                # アンカー一貫性チェック（per MISS retry budget exhaustion-side）
                if (miss_retry_count == ANCHOR_CHECK_INTERVAL // 2
                        or miss_retry_count > 3):
                    new_anchor, consistent = _pinpoint_verify_anchor(
                        anchor_b64, target, original_anchor, api_key, model)
                    if not consistent:
                        anchor = new_anchor
                        log_lines.append(
                            f"  ⚠ anchor drift @ MISS retry "
                            f"{miss_retry_count}: re-anchored: {anchor}")

                if miss_retry_count > 3:
                    log_lines.append(
                        f"  Phase 2 halt: max-3 MISS retries exhausted "
                        f"on HIT round {hit_round_count + 1} → Phase 1 "
                        f"fallback")
                    break

        # Phase 2 終了: 最終 center が回答（HIT 無しなら Phase 1 center に
        # 一致する初期値のまま, fallback も同値）
        if phase2_any_hit:
            found_sx, found_sy = center_x, center_y
            log_lines.append(
                f"  Phase 2 converged: final=({found_sx},{found_sy}), "
                f"sr={sr}")
        else:
            found_sx, found_sy = x_shot, y_shot
            log_lines.append(
                f"  fallback: Phase 1 center ({found_sx},{found_sy})")
    phase2_time = _time.time() - phase2_t0

    # ---------------------------------------------------------------
    # Phase 3: 精密追い込み（precision < 5 かつ Phase 2 で HIT した場合）
    # ---------------------------------------------------------------
    if (not stage_b_hit
            and precision < 5 and found_sx is not None
            and found_sy is not None):
        log_lines.append(f"Phase 3: refine to ±{precision}px")
        # HIT 点を中心に ±refine_radius の極小領域を探索
        refine_radius = max(precision + 2, 3)  # 探索範囲（precision=1→±3px）
        REFINE_ZOOM = 1024  # Claude の downsample 閾値に合わせた安全値
        REFINE_DOT_RADIUS = 1  # 最小ドット
        refine_max_rounds = 6
        refine_tried: list[tuple[int, int]] = [(found_sx, found_sy)]

        for refine_round in range(refine_max_rounds):
            # 探索領域（スクリーンショット座標）
            rx1 = max(0, found_sx - refine_radius)
            ry1 = max(0, found_sy - refine_radius)
            rx2 = min(shot_w, found_sx + refine_radius + 1)
            ry2 = min(shot_h, found_sy + refine_radius + 1)
            rw = rx2 - rx1
            rh = ry2 - ry1
            if rw <= 0 or rh <= 0:
                break

            # コンテキスト（探索領域の3倍）で拡大
            rctx_x1 = max(0, rx1 - rw)
            rctx_y1 = max(0, ry1 - rh)
            rctx_x2 = min(shot_w, rx2 + rw)
            rctx_y2 = min(shot_h, ry2 + rh)
            rctx_w = rctx_x2 - rctx_x1
            rctx_h = rctx_y2 - rctx_y1

            r_cropped = pil_img.crop((rctx_x1, rctx_y1, rctx_x2, rctx_y2))
            r_scale = min(REFINE_ZOOM / max(rctx_w, 1),
                          REFINE_ZOOM / max(rctx_h, 1))
            r_zoom_w = max(1, int(rctx_w * r_scale))
            r_zoom_h = max(1, int(rctx_h * r_scale))
            r_zoomed = r_cropped.resize(
                (r_zoom_w, r_zoom_h), PILImage.Resampling.BICUBIC)

            # ランダムドット生成（探索領域内）
            r_dots: list[tuple[int, int]] = []
            for _ in range(num_dots * 30):
                dx = random.randint(rx1, max(rx1, rx2 - 1))
                dy = random.randint(ry1, max(ry1, ry2 - 1))
                too_close = False
                for tx, ty in refine_tried:
                    if dx == tx and dy == ty:
                        too_close = True
                        break
                if not too_close:
                    for ex, ey in r_dots:
                        if dx == ex and dy == ey:
                            too_close = True
                            break
                if not too_close:
                    r_dots.append((dx, dy))
                if len(r_dots) >= num_dots:
                    break

            if not r_dots:
                log_lines.append(
                    f"  refine {refine_round + 1}: 候補位置枯渇")
                break

            # ドット描画（DOT_RADIUS=1）
            r_annotated = r_zoomed.copy()
            r_draw = ImageDraw.Draw(r_annotated)
            try:
                r_font = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                r_font = ImageFont.load_default()

            for i, (sx, sy) in enumerate(r_dots):
                zx = int((sx - rctx_x1) * r_scale)
                zy = int((sy - rctx_y1) * r_scale)
                color = _PINPOINT_COLORS[i % len(_PINPOINT_COLORS)]
                r_draw.ellipse(
                    [(zx - REFINE_DOT_RADIUS, zy - REFINE_DOT_RADIUS),
                     (zx + REFINE_DOT_RADIUS, zy + REFINE_DOT_RADIUS)],
                    fill=color, outline=(255, 255, 255), width=1)
                label = str(i + 1)
                r_draw.text((zx + REFINE_DOT_RADIUS + 3, zy - 12), label,
                            fill=color, font=r_font,
                            stroke_fill=(0, 0, 0), stroke_width=2)

            # 多数決で判定
            r_b64 = _grid_find_to_base64(r_annotated)
            r_hit, r_unanimous = _pinpoint_ask_dots_majority(
                r_b64, anchor, len(r_dots), api_key, model)

            if r_hit is not None and 1 <= r_hit <= len(r_dots):
                found_sx, found_sy = r_dots[r_hit - 1]
                vote_info = "unanimous" if r_unanimous else "majority"
                log_lines.append(
                    f"  refine {refine_round + 1}: HIT dot {r_hit} "
                    f"({vote_info}) → shot=({found_sx},{found_sy})")
                # 精度目標に到達: 探索半径を縮小して続行
                refine_radius = max(precision, 1)
                if refine_radius <= precision:
                    break
            else:
                for sx, sy in r_dots:
                    refine_tried.append((sx, sy))
                log_lines.append(
                    f"  refine {refine_round + 1}: MISS ({len(r_dots)} dots)")

    elapsed = _time.time() - t0

    # 物理座標に変換
    if window:
        phys_x = win_offset_x + int(found_sx * mon_phys_w / shot_w)
        phys_y = win_offset_y + int(found_sy * mon_phys_h / shot_h)
        _last_screenshot_info[monitor] = (shot_w, shot_h)
        _last_window_capture[monitor] = (
            win_offset_x, win_offset_y, mon_phys_w, mon_phys_h)
    else:
        phys_x, phys_y = _screen_to_physical(monitor, found_sx, found_sy)

    _record_verified_coord(phys_x, phys_y, "pinpoint")

    # peek 画像を生成
    peek_size = 200
    peek_left = max(0, phys_x - peek_size // 2)
    peek_top = max(0, phys_y - peek_size // 2)
    peek_cap = ScreenCapture(region=(peek_left, peek_top, peek_size, peek_size))
    peek_pil = peek_cap.capture_as_pil()
    peek_enlarged = peek_pil.resize(
        (peek_size * 2, peek_size * 2), resample=0)

    draw = ImageDraw.Draw(peek_enlarged)
    cx, cy = peek_size, peek_size
    draw.line([(cx - 20, cy), (cx + 20, cy)], fill="red", width=2)
    draw.line([(cx, cy - 20), (cx, cy + 20)], fill="red", width=2)

    # インフォバー
    try:
        info_font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        info_font = ImageFont.load_default()

    all_lines = (
        [f"pinpoint: ({phys_x},{phys_y}) {elapsed:.1f}s",
         f"click(x={found_sx}, y={found_sy}, monitor={monitor})"]
        + log_lines
    )
    info_h = 14 * len(all_lines) + 8
    result_img = PILImage.new(
        "RGB", (peek_size * 2, peek_size * 2 + info_h), (0, 0, 0))
    result_img.paste(peek_enlarged, (0, 0))
    info_draw = ImageDraw.Draw(result_img)
    for li, line in enumerate(all_lines):
        info_draw.text(
            (8, peek_size * 2 + 4 + li * 14), line,
            fill=(255, 255, 255), font=info_font)

    buf = BytesIO()
    result_img.save(buf, format="PNG")
    import json as _json
    coord_meta = {
        "tool": "pinpoint", "x": phys_x, "y": phys_y,
        "monitor": monitor, "elapsed": round(elapsed, 2),
        "phase2_rounds": (round_idx + 1) if (found_sx is not None and round_idx >= 0) else 0,
        "phase1_time": round(phase1_time, 3),
        "phase_1_5_time": round(phase_1_5_time, 3),
        "phase2_time": round(phase2_time, 3),
        "stage_b_hit": bool(stage_b_hit),
        "stage_b_time": round(stage_b_time, 3),
        "stage_b_match": stage_b_match_text if stage_b_hit else "",
        "stage_b_dist": round(stage_b_snap_dist, 2) if stage_b_hit else 0.0,
    }
    return [Image(data=buf.getvalue(), format="png"),
            f"COORDINATE_META:{_json.dumps(coord_meta)}"]


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
