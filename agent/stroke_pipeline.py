"""
stroke_pipeline.py
ストローク抽出・整列・平滑化・座標変換・筆圧生成パイプライン

OpenCV の輪郭データを描画可能なストロークに変換し、
筆圧カーブを生成して pen_stroke に渡せる形にする。

Neo ノード仕様:
  I13  Contour to Strokes     — 輪郭 → ストローク変換
  I14  Stroke Ordering         — 描画順の最適化
  I15  Stroke Smoothing        — ベジェフィッティング + 手描き揺れ
  I16  Stroke to Coordinates   — キャンバス相対座標 → 物理座標
  Iext1 Pressure Curve Generate — bell/linear/attack 筆圧カーブ
  Iext2 Pressure from Curvature — 曲率ベース筆圧
  Iext3 Pressure Merge          — 複数筆圧カーブの合成

データ型:
  Stroke = {
    "points": [[x, y], ...],
    "is_closed": bool,
    "length_px": float,
    "bbox": {"x": int, "y": int, "w": int, "h": int}
  }
  PressureCurve = {
    "values": [float, ...],    # 0.0-1.0
    "curve_type": str
  }
"""

import math
import random
from typing import Optional

import cv2
import numpy as np


# ============================================================================
# I13: Contour to Strokes
# ============================================================================

def contour_to_strokes(
    contours: list[np.ndarray],
    min_length_px: float = 10.0,
    simplify_epsilon: float = 2.0,
) -> dict:
    """OpenCV 輪郭をストロークリストに変換する。

    Args:
        contours: cv2.findContours の出力（list of np.ndarray）
        min_length_px: この長さ未満のストロークを除去
        simplify_epsilon: approxPolyDP の近似精度（小=忠実、大=滑らか）

    Returns:
        {"strokes": [Stroke, ...], "stroke_count": int}
    """
    strokes = []
    for contour in contours:
        length = cv2.arcLength(contour, closed=False)
        if length < min_length_px:
            continue

        # 近似
        epsilon = simplify_epsilon
        approx = cv2.approxPolyDP(contour, epsilon, closed=False)
        points = approx.reshape(-1, 2).tolist()

        if len(points) < 2:
            continue

        # 開閉判定: 始点と終点が十分近ければ閉曲線
        dist_start_end = math.hypot(
            points[0][0] - points[-1][0],
            points[0][1] - points[-1][1],
        )
        is_closed = dist_start_end < max(5.0, length * 0.03)

        # bbox
        pts_arr = np.array(points)
        x_min, y_min = pts_arr.min(axis=0)
        x_max, y_max = pts_arr.max(axis=0)

        strokes.append({
            "points": points,
            "is_closed": is_closed,
            "length_px": round(length, 1),
            "bbox": {
                "x": int(x_min), "y": int(y_min),
                "w": int(x_max - x_min), "h": int(y_max - y_min),
            },
        })

    return {"strokes": strokes, "stroke_count": len(strokes)}


# ============================================================================
# I14: Stroke Ordering
# ============================================================================

def stroke_ordering(
    strokes: list[dict],
    start_position: Optional[list[int]] = None,
    strategy: str = "nearest",
) -> dict:
    """ストロークを描画順に並べ替える。

    Args:
        strokes: Stroke のリスト
        start_position: [x, y] 描画開始位置（省略で [0, 0]）
        strategy: "nearest" / "top-to-bottom" / "left-to-right"

    Returns:
        {"ordered_strokes": [...], "total_draw_length": float, "total_lift_length": float}
    """
    if not strokes:
        return {"ordered_strokes": [], "total_draw_length": 0, "total_lift_length": 0}

    if strategy == "top-to-bottom":
        ordered = sorted(strokes, key=lambda s: s["bbox"]["y"])
    elif strategy == "left-to-right":
        ordered = sorted(strokes, key=lambda s: s["bbox"]["x"])
    else:
        # nearest-neighbor greedy
        ordered = _order_nearest(strokes, start_position or [0, 0])

    total_draw = sum(s["length_px"] for s in ordered)
    total_lift = _calc_lift_distance(ordered, start_position or [0, 0])

    return {
        "ordered_strokes": ordered,
        "total_draw_length": round(total_draw, 1),
        "total_lift_length": round(total_lift, 1),
    }


def _order_nearest(strokes: list[dict], start: list[int]) -> list[dict]:
    """最近傍法でストローク描画順を決定。"""
    remaining = list(range(len(strokes)))
    ordered = []
    current = start

    while remaining:
        best_idx = -1
        best_dist = float("inf")
        best_reverse = False

        for idx in remaining:
            s = strokes[idx]
            pts = s["points"]
            # 始点への距離
            d_start = math.hypot(pts[0][0] - current[0], pts[0][1] - current[1])
            # 終点への距離（逆順で描くケース）
            d_end = math.hypot(pts[-1][0] - current[0], pts[-1][1] - current[1])

            if d_start < best_dist:
                best_dist = d_start
                best_idx = idx
                best_reverse = False
            if d_end < best_dist:
                best_dist = d_end
                best_idx = idx
                best_reverse = True

        stroke = dict(strokes[best_idx])
        if best_reverse and not stroke["is_closed"]:
            stroke["points"] = list(reversed(stroke["points"]))

        ordered.append(stroke)
        current = stroke["points"][-1]
        remaining.remove(best_idx)

    return ordered


def _calc_lift_distance(ordered_strokes: list[dict], start: list[int]) -> float:
    """ストローク間の空中移動距離の合計。"""
    if not ordered_strokes:
        return 0.0
    total = 0.0
    current = start
    for s in ordered_strokes:
        pts = s["points"]
        total += math.hypot(pts[0][0] - current[0], pts[0][1] - current[1])
        current = pts[-1]
    return total


# ============================================================================
# I15: Stroke Smoothing
# ============================================================================

def stroke_smoothing(
    strokes: list[dict],
    method: str = "bezier",
    smoothness: float = 0.5,
    jitter: float = 0.0,
) -> dict:
    """ストロークの点列を平滑化する。

    Args:
        strokes: Stroke のリスト
        method: "bezier" / "moving_average" / "catmull_rom"
        smoothness: 平滑度 0.0-1.0
        jitter: 手描き揺れ量(px)。0 = 揺れなし

    Returns:
        {"smoothed_strokes": [...]}
    """
    smoothed = []
    for stroke in strokes:
        pts = stroke["points"]
        if len(pts) < 3:
            smoothed.append(stroke)
            continue

        if method == "moving_average":
            new_pts = _smooth_moving_average(pts, smoothness)
        elif method == "catmull_rom":
            new_pts = _smooth_catmull_rom(pts, smoothness)
        else:
            new_pts = _smooth_bezier(pts, smoothness)

        if jitter > 0:
            new_pts = _add_jitter(new_pts, jitter)

        # 再計算
        length = _polyline_length(new_pts)
        pts_arr = np.array(new_pts)
        x_min, y_min = pts_arr.min(axis=0)
        x_max, y_max = pts_arr.max(axis=0)

        smoothed.append({
            "points": new_pts,
            "is_closed": stroke["is_closed"],
            "length_px": round(length, 1),
            "bbox": {
                "x": int(x_min), "y": int(y_min),
                "w": int(x_max - x_min), "h": int(y_max - y_min),
            },
        })

    return {"smoothed_strokes": smoothed}


def _smooth_bezier(pts: list[list[int]], smoothness: float) -> list[list[int]]:
    """ベジェフィッティングで平滑化。

    隣接3点のセグメントごとに二次ベジェで補間し、
    smoothness で元の点列とのブレンド比率を制御する。
    """
    if len(pts) < 3:
        return pts

    n = len(pts)
    result = [pts[0]]

    # 各セグメントをベジェ曲線として補間
    for i in range(n - 2):
        p0 = np.array(pts[i], dtype=float)
        p1 = np.array(pts[i + 1], dtype=float)
        p2 = np.array(pts[i + 2], dtype=float)

        # 制御点は中点を smoothness でブレンド
        ctrl = p1  # 元の中間点が制御点
        mid_01 = (p0 + p1) / 2
        mid_12 = (p1 + p2) / 2

        # セグメント間の点数
        seg_len = math.hypot(*(p1 - p0)) + math.hypot(*(p2 - p1))
        num_pts = max(3, int(seg_len / 3))

        for j in range(1, num_pts):
            t = j / num_pts
            # 二次ベジェ: B(t) = (1-t)²·P0 + 2(1-t)t·P1 + t²·P2
            bezier_pt = ((1 - t) ** 2 * mid_01 +
                         2 * (1 - t) * t * ctrl +
                         t ** 2 * mid_12)
            # 線形補間の点
            linear_pt = mid_01 + (mid_12 - mid_01) * t
            # smoothness でブレンド
            blended = linear_pt + (bezier_pt - linear_pt) * smoothness
            result.append([int(round(blended[0])), int(round(blended[1]))])

    result.append(pts[-1])
    return result


def _smooth_moving_average(pts: list[list[int]], smoothness: float) -> list[list[int]]:
    """移動平均による平滑化。"""
    window = max(3, int(smoothness * 10) | 1)  # 奇数に
    arr = np.array(pts, dtype=float)
    n = len(arr)
    if n <= window:
        return pts

    result = arr.copy()
    half = window // 2
    for i in range(half, n - half):
        result[i] = arr[i - half:i + half + 1].mean(axis=0)

    # 始点・終点は保持
    result[0] = arr[0]
    result[-1] = arr[-1]
    return [[int(round(x)), int(round(y))] for x, y in result]


def _smooth_catmull_rom(pts: list[list[int]], smoothness: float) -> list[list[int]]:
    """Catmull-Rom スプライン補間。"""
    if len(pts) < 4:
        return pts

    alpha = 0.5 * smoothness + 0.25  # 0.25-0.75 (centripetal range)
    result = [pts[0]]

    for i in range(len(pts) - 1):
        p0 = np.array(pts[max(0, i - 1)], dtype=float)
        p1 = np.array(pts[i], dtype=float)
        p2 = np.array(pts[min(len(pts) - 1, i + 1)], dtype=float)
        p3 = np.array(pts[min(len(pts) - 1, i + 2)], dtype=float)

        seg_len = math.hypot(*(p2 - p1))
        num_pts = max(2, int(seg_len / 3))

        for j in range(1, num_pts):
            t = j / num_pts
            # Catmull-Rom basis
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            result.append([int(round(pt[0])), int(round(pt[1]))])

    result.append(pts[-1])
    return result


def _add_jitter(pts: list[list[int]], amount: float) -> list[list[int]]:
    """手描き風の微細な揺れを付加。始点・終点は揺らさない。"""
    if len(pts) <= 2:
        return pts
    result = [pts[0]]
    for p in pts[1:-1]:
        dx = random.gauss(0, amount)
        dy = random.gauss(0, amount)
        result.append([int(round(p[0] + dx)), int(round(p[1] + dy))])
    result.append(pts[-1])
    return result


def _polyline_length(pts: list[list[int]]) -> float:
    """ポリラインの総長を計算。"""
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return total


# ============================================================================
# I16: Stroke to Coordinates
# ============================================================================

def stroke_to_coordinates(
    strokes: list[dict],
    canvas_rect: dict,
    image_size: dict,
) -> dict:
    """ストロークを物理座標に変換する。

    画像上の相対座標 → 画面上の物理座標へ変換。
    pen_stroke に渡せる形式で出力。

    Args:
        strokes: Stroke のリスト
        canvas_rect: {"x": int, "y": int, "w": int, "h": int} — キャンバスの画面上の位置（物理座標）
        image_size: {"w": int, "h": int} — 元画像のサイズ

    Returns:
        {"stroke_coords": [...], "stroke_count": int}
          stroke_coords の各要素: {"points": [[phys_x, phys_y], ...], "is_closed": bool}
    """
    cx, cy = canvas_rect["x"], canvas_rect["y"]
    cw, ch = canvas_rect["w"], canvas_rect["h"]
    iw, ih = image_size["w"], image_size["h"]

    scale_x = cw / iw if iw > 0 else 1.0
    scale_y = ch / ih if ih > 0 else 1.0

    stroke_coords = []
    for stroke in strokes:
        phys_pts = []
        for px, py in stroke["points"]:
            phys_x = int(round(cx + px * scale_x))
            phys_y = int(round(cy + py * scale_y))
            phys_pts.append([phys_x, phys_y])
        stroke_coords.append({
            "points": phys_pts,
            "is_closed": stroke.get("is_closed", False),
        })

    return {"stroke_coords": stroke_coords, "stroke_count": len(stroke_coords)}


# ============================================================================
# Iext1: Pressure Curve Generate
# ============================================================================

def pressure_curve_generate(
    num_points: int,
    curve_type: str = "bell",
    base_pressure: float = 0.6,
    variation: float = 0.3,
) -> dict:
    """ストロークの点数に対応する筆圧カーブを生成する。

    Args:
        num_points: ストロークの点数
        curve_type: "bell"(入り抜き) / "linear" / "attack"(入り強め) / "constant"
        base_pressure: 基本筆圧 0.0-1.0
        variation: 筆圧変動幅 0.0-0.5

    Returns:
        PressureCurve: {"values": [float, ...], "curve_type": str}
    """
    n = max(1, num_points)
    base = max(0.0, min(1.0, base_pressure))
    var = max(0.0, min(0.5, variation))

    if curve_type == "bell":
        values = _gen_bell(n, base, var)
    elif curve_type == "attack":
        values = _gen_attack(n, base, var)
    elif curve_type == "constant":
        values = [base] * n
    else:  # linear
        values = _gen_linear(n, base, var)

    return {"values": values, "curve_type": curve_type}


def _gen_bell(n: int, base: float, var: float) -> list[float]:
    """釣鐘型: 入り抜きで中央がピーク。"""
    if n <= 1:
        return [base]
    peak = min(1.0, base + var)
    ends = max(0.0, base - var)
    return [
        round(ends + (peak - ends) * math.sin(i / (n - 1) * math.pi), 3)
        for i in range(n)
    ]


def _gen_attack(n: int, base: float, var: float) -> list[float]:
    """入り重視: 高筆圧で始まり徐々に抜ける。"""
    if n <= 1:
        return [base]
    start = min(1.0, base + var)
    end = max(0.0, base - var)
    return [
        round(start - (start - end) * (i / (n - 1)), 3)
        for i in range(n)
    ]


def _gen_linear(n: int, base: float, var: float) -> list[float]:
    """線形: start → end。"""
    if n <= 1:
        return [base]
    start = max(0.0, base - var / 2)
    end = min(1.0, base + var / 2)
    return [
        round(start + (end - start) * (i / (n - 1)), 3)
        for i in range(n)
    ]


# ============================================================================
# Iext2: Pressure from Curvature
# ============================================================================

def pressure_from_curvature(
    stroke: dict,
    min_pressure: float = 0.1,
    max_pressure: float = 0.9,
    curvature_sensitivity: float = 0.5,
) -> dict:
    """ストロークの曲率に基づいて筆圧を生成する。

    直線部分は太く（高筆圧）、急カーブは細く（低筆圧）。
    漫画の G ペンの自然な描き味を再現。

    Args:
        stroke: Stroke dict
        min_pressure: 最小筆圧 0.0-1.0
        max_pressure: 最大筆圧 0.0-1.0
        curvature_sensitivity: 曲率感度 0.0-1.0（高いほどメリハリ）

    Returns:
        PressureCurve: {"values": [float, ...], "curve_type": "curvature"}
    """
    pts = stroke["points"]
    n = len(pts)
    if n < 3:
        mid = (min_pressure + max_pressure) / 2
        return {"values": [round(mid, 3)] * n, "curve_type": "curvature"}

    # 各点の曲率を計算
    curvatures = _compute_curvatures(pts)

    # 曲率を 0-1 に正規化
    max_curv = max(curvatures) if max(curvatures) > 0 else 1.0
    norm_curvatures = [c / max_curv for c in curvatures]

    # 曲率 → 筆圧: 曲率が高い → 低筆圧、曲率が低い → 高筆圧
    pressure_range = max_pressure - min_pressure
    values = []
    for nc in norm_curvatures:
        # sensitivity で曲率の影響度を制御
        influence = nc * curvature_sensitivity
        p = max_pressure - pressure_range * influence
        values.append(round(max(min_pressure, min(max_pressure, p)), 3))

    # 終端テーパー（G ペンの「抜き」）
    taper_len = max(2, n // 8)
    for i in range(taper_len):
        t = (taper_len - i) / taper_len
        idx = n - 1 - i
        values[idx] = round(values[idx] * t * 0.5 + min_pressure * (1 - t * 0.5), 3)

    return {"values": values, "curve_type": "curvature"}


def _compute_curvatures(pts: list[list[int]]) -> list[float]:
    """各点の曲率（Menger curvature）を計算する。"""
    n = len(pts)
    curvatures = [0.0] * n

    for i in range(1, n - 1):
        p0 = np.array(pts[i - 1], dtype=float)
        p1 = np.array(pts[i], dtype=float)
        p2 = np.array(pts[i + 1], dtype=float)

        # 三角形の面積 (外積の半分)
        area = abs(np.cross(p1 - p0, p2 - p0)) / 2.0
        # 三辺の長さ
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)

        denom = a * b * c
        if denom > 1e-10:
            curvatures[i] = 4.0 * area / denom
        else:
            curvatures[i] = 0.0

    # 端点は隣接点の曲率をコピー
    curvatures[0] = curvatures[1] if n > 1 else 0.0
    curvatures[-1] = curvatures[-2] if n > 1 else 0.0

    return curvatures


# ============================================================================
# Iext3: Pressure Merge
# ============================================================================

def pressure_merge(
    curves: list[dict],
    method: str = "multiply",
) -> dict:
    """複数の筆圧カーブを合成する。

    Args:
        curves: PressureCurve のリスト [{"values": [...], "curve_type": str}, ...]
        method: "multiply" / "average" / "max"

    Returns:
        PressureCurve: {"values": [float, ...], "curve_type": "merged_{method}"}
    """
    if not curves:
        return {"values": [], "curve_type": f"merged_{method}"}
    if len(curves) == 1:
        return {
            "values": list(curves[0]["values"]),
            "curve_type": f"merged_{method}",
        }

    # 全カーブを同じ長さに揃える（最長に合わせてリサンプル）
    max_len = max(len(c["values"]) for c in curves)
    resampled = [_resample(c["values"], max_len) for c in curves]

    if method == "multiply":
        merged = [1.0] * max_len
        for vals in resampled:
            for i in range(max_len):
                merged[i] *= vals[i]
    elif method == "max":
        merged = [0.0] * max_len
        for vals in resampled:
            for i in range(max_len):
                merged[i] = max(merged[i], vals[i])
    else:  # average
        merged = [0.0] * max_len
        for vals in resampled:
            for i in range(max_len):
                merged[i] += vals[i]
        merged = [v / len(curves) for v in merged]

    merged = [round(max(0.0, min(1.0, v)), 3) for v in merged]
    return {"values": merged, "curve_type": f"merged_{method}"}


def _resample(values: list[float], target_len: int) -> list[float]:
    """筆圧カーブを target_len にリサンプル（線形補間）。"""
    n = len(values)
    if n == target_len:
        return list(values)
    if n == 0:
        return [0.5] * target_len
    if n == 1:
        return [values[0]] * target_len

    result = []
    for i in range(target_len):
        t = i / (target_len - 1) * (n - 1)
        idx = int(t)
        frac = t - idx
        if idx >= n - 1:
            result.append(values[-1])
        else:
            result.append(values[idx] + (values[idx + 1] - values[idx]) * frac)
    return result
