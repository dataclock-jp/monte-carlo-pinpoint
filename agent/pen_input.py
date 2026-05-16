"""
pen_input.py
仮想ペンタブレット入力モジュール

Windows の InjectSyntheticPointerInput API を使い、
筆圧付きペンストロークをアプリケーション（CSP 等）に注入する。
ドライバインストール不要、ユーザーモード API のみ。

要件:
  - Windows 10 1809+
  - 対象アプリの設定で「タブレットPC」(WM_POINTER) モードを有効にすること
    例: CSP → ファイル > 環境設定 > タブレット → 「タブレットPC」

座標系:
  - 物理スクリーン座標（GetCursorPos / SetCursorPos と同じ絶対座標）
  - DPI-aware プロセス前提（mcp_server.py が起動時に設定済み）
"""

import ctypes
import ctypes.wintypes
import logging
import math
import time
from ctypes import Structure, Union, byref, c_int, c_int32, c_uint32, c_uint64
from ctypes.wintypes import DWORD, HANDLE, HWND, POINT, RECT
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# POINTER_INPUT_TYPE
PT_PEN = 3

# POINTER_FEEDBACK_MODE
POINTER_FEEDBACK_DEFAULT = 1
POINTER_FEEDBACK_INDIRECT = 2
POINTER_FEEDBACK_NONE = 3

# POINTER_FLAGS
POINTER_FLAG_NEW = 0x00000001
POINTER_FLAG_INRANGE = 0x00000002
POINTER_FLAG_INCONTACT = 0x00000004
POINTER_FLAG_FIRSTBUTTON = 0x00000010
POINTER_FLAG_PRIMARY = 0x00002000
POINTER_FLAG_CONFIDENCE = 0x00004000
POINTER_FLAG_DOWN = 0x00010000
POINTER_FLAG_UPDATE = 0x00020000
POINTER_FLAG_UP = 0x00040000

# PEN_FLAGS
PEN_FLAG_NONE = 0x00000000

# PEN_MASK
PEN_MASK_PRESSURE = 0x00000001
PEN_MASK_TILT_X = 0x00000004
PEN_MASK_TILT_Y = 0x00000008

# POINTER_BUTTON_CHANGE_TYPE
POINTER_CHANGE_NONE = 0
POINTER_CHANGE_FIRSTBUTTON_DOWN = 1
POINTER_CHANGE_FIRSTBUTTON_UP = 2


# ============================================================================
# Structures (x64)
# ============================================================================

class POINTER_INFO(Structure):
    """96 bytes on x64."""
    _fields_ = [
        ("pointerType", c_uint32),
        ("pointerId", c_uint32),
        ("frameId", c_uint32),
        ("pointerFlags", c_uint32),
        ("sourceDevice", HANDLE),
        ("hwndTarget", HWND),
        ("ptPixelLocation", POINT),
        ("ptHimetricLocation", POINT),
        ("ptPixelLocationRaw", POINT),
        ("ptHimetricLocationRaw", POINT),
        ("dwTime", DWORD),
        ("historyCount", c_uint32),
        ("InputData", c_int32),
        ("dwKeyStates", DWORD),
        ("PerformanceCount", c_uint64),
        ("ButtonChangeType", c_int),
    ]


class POINTER_PEN_INFO(Structure):
    """120 bytes on x64."""
    _fields_ = [
        ("pointerInfo", POINTER_INFO),
        ("penFlags", c_uint32),
        ("penMask", c_uint32),
        ("pressure", c_uint32),
        ("rotation", c_uint32),
        ("tiltX", c_int32),
        ("tiltY", c_int32),
    ]


class POINTER_TOUCH_INFO(Structure):
    """144 bytes on x64 (largest union member)."""
    _fields_ = [
        ("pointerInfo", POINTER_INFO),
        ("touchFlags", c_uint32),
        ("touchMask", c_uint32),
        ("rcContact", RECT),
        ("rcContactRaw", RECT),
        ("orientation", c_uint32),
        ("pressure", c_uint32),
    ]


class _POINTER_TYPE_UNION(Union):
    _fields_ = [
        ("pointerInfo", POINTER_INFO),
        ("touchInfo", POINTER_TOUCH_INFO),
        ("penInfo", POINTER_PEN_INFO),
    ]


class POINTER_TYPE_INFO(Structure):
    """152 bytes on x64. InjectSyntheticPointerInput の入力構造体。"""
    _fields_ = [
        ("type", c_uint32),
        ("DUMMYUNIONNAME", _POINTER_TYPE_UNION),
    ]


# ============================================================================
# API bindings
# ============================================================================

_user32 = ctypes.windll.user32

_user32.CreateSyntheticPointerDevice.restype = HANDLE
_user32.CreateSyntheticPointerDevice.argtypes = [c_uint32, ctypes.c_ulong, c_uint32]

_user32.InjectSyntheticPointerInput.restype = ctypes.c_bool
_user32.InjectSyntheticPointerInput.argtypes = [
    HANDLE, ctypes.POINTER(POINTER_TYPE_INFO), c_uint32
]

_user32.DestroySyntheticPointerDevice.restype = None
_user32.DestroySyntheticPointerDevice.argtypes = [HANDLE]


# ============================================================================
# Pen stroke injection
# ============================================================================

def _make_pen_frame(
    pti: POINTER_TYPE_INFO,
    x: int, y: int,
    pressure: int,
    flags: int,
    button_change: int,
    tilt_x: int = 0,
    tilt_y: int = 0,
) -> None:
    """POINTER_TYPE_INFO をペンフレーム1つ分にセットする。"""
    pen = pti.DUMMYUNIONNAME.penInfo
    pen.pointerInfo.ptPixelLocation.x = x
    pen.pointerInfo.ptPixelLocation.y = y
    pen.pointerInfo.ptPixelLocationRaw.x = x
    pen.pointerInfo.ptPixelLocationRaw.y = y
    pen.pointerInfo.pointerFlags = flags
    pen.pointerInfo.ButtonChangeType = button_change
    pen.pressure = max(0, min(1024, pressure))

    mask = PEN_MASK_PRESSURE
    if tilt_x != 0 or tilt_y != 0:
        mask |= PEN_MASK_TILT_X | PEN_MASK_TILT_Y
        pen.tiltX = max(-90, min(90, tilt_x))
        pen.tiltY = max(-90, min(90, tilt_y))
    pen.penMask = mask


def _get_virtual_screen_origin() -> tuple[int, int]:
    """仮想スクリーンの左上原点を取得する。
    InjectSyntheticPointerInput はこの原点基準の座標を要求する。"""
    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77
    vx = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    return (vx, vy)


def inject_pen_stroke(
    points: list[tuple[int, int]],
    pressures: list[int],
    interval_ms: float = 8.0,
    tilt_x: int = 0,
    tilt_y: int = 0,
) -> dict:
    """筆圧付きペンストロークを注入する。

    Args:
        points: [(x, y), ...] 物理スクリーン座標のリスト（2点以上）
        pressures: [0..1024, ...] 筆圧リスト（points と同じ長さ）
        interval_ms: ポイント間の待機時間（ミリ秒）。8ms以上推奨（短すぎると失敗）
        tilt_x: ペンの傾き X 軸（-90..+90 度）
        tilt_y: ペンの傾き Y 軸（-90..+90 度）

    Returns:
        {"success": bool, "points_injected": int, "error": str}

    Note:
        座標は物理スクリーン座標（GetCursorPos と同じ）で渡す。
        内部で仮想スクリーン原点基準に変換される（マルチモニター対応）。
        dwTime は 0 固定（非ゼロだと Windows がフレームを無視する）。
    """
    if len(points) < 2:
        return {"success": False, "points_injected": 0, "error": "2点以上必要"}
    if len(points) != len(pressures):
        return {"success": False, "points_injected": 0,
                "error": f"points ({len(points)}) と pressures ({len(pressures)}) の長さが不一致"}

    device = _user32.CreateSyntheticPointerDevice(PT_PEN, 1, POINTER_FEEDBACK_DEFAULT)
    if not device:
        err = ctypes.get_last_error()
        return {"success": False, "points_injected": 0,
                "error": f"CreateSyntheticPointerDevice failed (error={err})"}

    # 物理座標 → 仮想スクリーン左上基準に変換（マルチモニター対応）
    vx, vy = _get_virtual_screen_origin()
    api_points = [(x - vx, y - vy) for x, y in points]

    injected = 0
    frame_id = 0
    interval_s = interval_ms / 1000.0

    def _inject(pti, label=""):
        nonlocal injected, frame_id
        frame_id += 1
        pen = pti.DUMMYUNIONNAME.penInfo
        pen.pointerInfo.frameId = frame_id
        pen.pointerInfo.dwTime = 0           # MUST be zero
        pen.pointerInfo.historyCount = 1     # MUST be 1
        ok = _user32.InjectSyntheticPointerInput(device, byref(pti), 1)
        if not ok:
            err = ctypes.get_last_error()
            logger.warning(f"Inject {label} failed (error={err})")
            return False
        injected += 1
        return True

    try:
        pti = POINTER_TYPE_INFO()
        ctypes.memset(byref(pti), 0, ctypes.sizeof(pti))
        pti.type = PT_PEN
        pen = pti.DUMMYUNIONNAME.penInfo
        pen.pointerInfo.pointerType = PT_PEN
        pen.pointerInfo.pointerId = 0
        pen.penFlags = PEN_FLAG_NONE

        # --- HOVER (ペンが範囲内に入る — DOWN の前に必要) ---
        flags_hover = (POINTER_FLAG_NEW | POINTER_FLAG_INRANGE |
                        POINTER_FLAG_PRIMARY | POINTER_FLAG_CONFIDENCE)
        _make_pen_frame(pti, api_points[0][0], api_points[0][1], 0,
                        flags_hover, POINTER_CHANGE_NONE, tilt_x, tilt_y)
        if not _inject(pti, "HOVER"):
            return {"success": False, "points_injected": 0, "error": "Inject HOVER failed"}
        time.sleep(interval_s)

        # --- DOWN (ペンが接触) ---
        flags_down = (POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT |
                       POINTER_FLAG_DOWN | POINTER_FLAG_FIRSTBUTTON |
                       POINTER_FLAG_PRIMARY | POINTER_FLAG_CONFIDENCE)
        _make_pen_frame(pti, api_points[0][0], api_points[0][1], pressures[0],
                        flags_down, POINTER_CHANGE_FIRSTBUTTON_DOWN, tilt_x, tilt_y)
        if not _inject(pti, "DOWN"):
            return {"success": False, "points_injected": 0, "error": "Inject DOWN failed"}
        time.sleep(interval_s)

        # --- UPDATE (ペンが動く) ---
        flags_update = (POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT |
                         POINTER_FLAG_UPDATE | POINTER_FLAG_FIRSTBUTTON |
                         POINTER_FLAG_PRIMARY | POINTER_FLAG_CONFIDENCE)
        for i in range(1, len(api_points) - 1):
            _make_pen_frame(pti, api_points[i][0], api_points[i][1], pressures[i],
                            flags_update, POINTER_CHANGE_NONE, tilt_x, tilt_y)
            if not _inject(pti, f"UPDATE[{i}]"):
                break
            time.sleep(interval_s)

        # --- UP (ペンが離れる) ---
        flags_up = (POINTER_FLAG_INRANGE | POINTER_FLAG_UP |
                     POINTER_FLAG_PRIMARY | POINTER_FLAG_CONFIDENCE)
        _make_pen_frame(pti, api_points[-1][0], api_points[-1][1], 0,
                        flags_up, POINTER_CHANGE_FIRSTBUTTON_UP, tilt_x, tilt_y)
        _inject(pti, "UP")
        time.sleep(interval_s)

        # --- END HOVER (ペンが範囲外に出る) ---
        flags_end = POINTER_FLAG_UPDATE | POINTER_FLAG_PRIMARY
        _make_pen_frame(pti, api_points[-1][0], api_points[-1][1], 0,
                        flags_end, POINTER_CHANGE_NONE, tilt_x, tilt_y)
        _inject(pti, "ENDHOVER")

    finally:
        _user32.DestroySyntheticPointerDevice(device)

    logger.info(f"pen_stroke: {injected} frames injected ({len(points)} stroke points)")
    return {"success": injected >= len(points), "points_injected": injected, "error": ""}


# ============================================================================
# Pressure curve helpers
# ============================================================================

def pressure_curve_linear(n: int, start: int = 100, end: int = 100) -> list[int]:
    """線形の筆圧カーブ。start から end へ均等変化。"""
    if n <= 1:
        return [start]
    return [int(start + (end - start) * i / (n - 1)) for i in range(n)]


def pressure_curve_bell(n: int, peak: int = 800, ends: int = 50) -> list[int]:
    """釣鐘型（入り抜き）。開始と終了が細く、中央が太い。"""
    if n <= 1:
        return [peak]
    result = []
    for i in range(n):
        t = i / (n - 1)  # 0.0 .. 1.0
        # sin curve: 0 → 1 → 0
        factor = math.sin(t * math.pi)
        p = int(ends + (peak - ends) * factor)
        result.append(max(0, min(1024, p)))
    return result


def pressure_curve_attack(n: int, peak: int = 900, tail: int = 100) -> list[int]:
    """入り重視（筆圧高く始まり、徐々に抜ける）。"""
    if n <= 1:
        return [peak]
    result = []
    for i in range(n):
        t = i / (n - 1)
        p = int(peak - (peak - tail) * t)
        result.append(max(0, min(1024, p)))
    return result


def interpolate_points(
    waypoints: list[tuple[int, int]],
    step_px: float = 3.0,
) -> list[tuple[int, int]]:
    """ウェイポイント間を step_px ピクセル間隔で補間して滑らかな座標列を生成。"""
    if len(waypoints) < 2:
        return list(waypoints)

    result = [waypoints[0]]
    for i in range(1, len(waypoints)):
        x0, y0 = waypoints[i - 1]
        x1, y1 = waypoints[i]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < step_px:
            result.append((x1, y1))
            continue
        steps = int(dist / step_px)
        for s in range(1, steps + 1):
            t = s / steps
            result.append((int(x0 + dx * t), int(y0 + dy * t)))
    # 最後の点を確実に含める
    if result[-1] != waypoints[-1]:
        result.append(waypoints[-1])
    return result


def bezier_points(
    p0: tuple[int, int],
    p1: tuple[int, int],
    p2: tuple[int, int],
    p3: tuple[int, int],
    num_points: int = 50,
) -> list[tuple[int, int]]:
    """3次ベジェ曲線の座標列を生成。

    Args:
        p0: 始点
        p1: 制御点1
        p2: 制御点2
        p3: 終点
        num_points: 分割数
    """
    result = []
    for i in range(num_points):
        t = i / (num_points - 1)
        u = 1 - t
        x = (u**3 * p0[0] + 3 * u**2 * t * p1[0] +
             3 * u * t**2 * p2[0] + t**3 * p3[0])
        y = (u**3 * p0[1] + 3 * u**2 * t * p1[1] +
             3 * u * t**2 * p2[1] + t**3 * p3[1])
        result.append((int(x), int(y)))
    return result
