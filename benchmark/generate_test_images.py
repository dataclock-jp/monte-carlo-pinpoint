"""テスト画像生成器 — pinpoint ベンチマーク用

5カテゴリ × 10枚 = 50枚の合成画像を生成し、
グラウンドトゥルス座標を ground_truth.json に保存する。

カテゴリ:
  1. ui_button    — 標準的な UI ボタン/アイコン（SoM の得意分野）
  2. ui_dense     — 密集 UI（ツールバー、タブバー）
  3. natural      — 非UI ターゲット（自然画像内の意味的な点）
  4. ambiguous    — 低コントラスト/小ターゲット/曖昧記述
  5. precision    — precision パラメータテスト用（1, 3, 5, 10, 20）

生成物:
  benchmark/images/  — PNG 画像
  benchmark/ground_truth.json — [{id, category, image, target, x, y}, ...]
"""

import json
import math
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 画像サイズ（4K 相当）
IMG_W, IMG_H = 1920, 1080
OUTPUT_DIR = Path(__file__).parent / "images"
GT_FILE = Path(__file__).parent / "ground_truth.json"

# 再現性
random.seed(42)

# フォント（Windows のシステムフォント）
def _get_font(size: int):
    for name in ["segoeui.ttf", "arial.ttf", "calibri.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ============================================================
# カテゴリ 1: UI ボタン / アイコン
# ============================================================
def _gen_ui_button(idx: int) -> dict:
    """各画像で異なるレイアウトのUIボタン"""
    # 10種類の異なるUIレイアウト
    layouts = [
        {"bg": (240, 240, 240), "bar_bg": (60, 60, 60), "btn_bg": (100, 130, 200), "bar_y": 0, "bar_h": 50, "btn_area": "top"},
        {"bg": (30, 30, 35), "bar_bg": (45, 45, 50), "btn_bg": (70, 130, 70), "bar_y": 0, "bar_h": 45, "btn_area": "top"},
        {"bg": (245, 245, 248), "bar_bg": (230, 230, 235), "btn_bg": (180, 80, 80), "bar_y": 60, "bar_h": 50, "btn_area": "mid"},
        {"bg": (250, 250, 250), "bar_bg": (240, 240, 245), "btn_bg": (100, 100, 170), "bar_y": 0, "bar_h": 80, "btn_area": "top_wide"},
        {"bg": (35, 38, 45), "bar_bg": (50, 53, 60), "btn_bg": (60, 140, 180), "bar_y": 0, "bar_h": 55, "btn_area": "top"},
        {"bg": (240, 238, 235), "bar_bg": (220, 218, 215), "btn_bg": (150, 100, 60), "bar_y": 40, "bar_h": 45, "btn_area": "mid"},
        {"bg": (248, 248, 252), "bar_bg": (60, 60, 65), "btn_bg": (90, 150, 90), "bar_y": 0, "bar_h": 50, "btn_area": "top"},
        {"bg": (30, 32, 38), "bar_bg": (42, 44, 50), "btn_bg": (180, 130, 60), "bar_y": 0, "bar_h": 60, "btn_area": "top"},
        {"bg": (245, 242, 240), "bar_bg": (235, 232, 230), "btn_bg": (130, 70, 140), "bar_y": 50, "bar_h": 50, "btn_area": "mid"},
        {"bg": (255, 255, 255), "bar_bg": (245, 245, 250), "btn_bg": (80, 120, 170), "bar_y": 0, "bar_h": 70, "btn_area": "top_wide"},
    ]
    layout = layouts[idx]
    rng = random.Random(42 + idx * 11)

    img = Image.new("RGB", (IMG_W, IMG_H), layout["bg"])
    draw = ImageDraw.Draw(img)

    # バー背景
    draw.rectangle([0, layout["bar_y"], IMG_W, layout["bar_y"] + layout["bar_h"]], fill=layout["bar_bg"])

    # ボタンラベル — 各画像でシャッフル
    all_labels = ["File", "Edit", "View", "Help", "Settings", "Save", "Open", "Close",
                  "New", "Print", "Undo", "Redo", "Cut", "Copy", "Paste", "Search",
                  "Replace", "Zoom", "Export", "Import"]
    rng.shuffle(all_labels)

    # ボタン配置 — レイアウトごとにサイズ・間隔を変える
    buttons = []
    font = _get_font(14)
    num_btns = rng.randint(6, 12)
    btn_w = rng.randint(80, 130)
    btn_h = rng.randint(26, 36)
    gap = rng.randint(5, 20)
    start_x = rng.randint(10, 60)
    btn_y = layout["bar_y"] + (layout["bar_h"] - btn_h) // 2

    text_color = "white" if layout["bar_bg"][0] < 128 else (40, 40, 40)

    for i, label in enumerate(all_labels[:num_btns]):
        bx = start_x + i * (btn_w + gap)
        if bx + btn_w > IMG_W - 10:
            break
        draw.rounded_rectangle([bx, btn_y, bx + btn_w, btn_y + btn_h], radius=5, fill=layout["btn_bg"])
        draw.text((bx + 8, btn_y + (btn_h - 14) // 2), label, fill=text_color, font=font)
        buttons.append({"label": label, "cx": bx + btn_w // 2, "cy": btn_y + btn_h // 2})

    # コンテンツ領域にダミー要素を追加（画像を現実的に）
    content_y = layout["bar_y"] + layout["bar_h"] + 20
    for _ in range(rng.randint(3, 8)):
        rx = rng.randint(20, IMG_W - 200)
        ry = rng.randint(content_y, IMG_H - 80)
        rw, rh = rng.randint(60, 300), rng.randint(20, 60)
        c = tuple(max(0, min(255, layout["bg"][j] + rng.randint(-30, 30))) for j in range(3))
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=c)

    target_btn = buttons[idx % len(buttons)]
    return {
        "image": img,
        "target": f"the center of the '{target_btn['label']}' button in the toolbar",
        "x": target_btn["cx"],
        "y": target_btn["cy"],
    }


# ============================================================
# カテゴリ 2: 密集 UI
# ============================================================
def _gen_ui_dense(idx: int) -> dict:
    """密集したアイコン群 — 各画像で異なるレイアウト・スタイル"""
    # 10種類の異なる密集UIレイアウト
    styles = [
        {"bg": (245, 245, 248), "icon_bg": (235, 235, 240), "cols": 15, "spacing": 80, "icon_size": (60, 35), "y_start": 35, "label": "ribbon"},
        {"bg": (50, 50, 55), "icon_bg": (70, 70, 80), "cols": 12, "spacing": 50, "icon_size": (40, 40), "y_start": 10, "label": "dark toolbar"},
        {"bg": (240, 240, 240), "icon_bg": (220, 225, 235), "cols": 8, "spacing": 110, "icon_size": (90, 50), "y_start": 60, "label": "palette"},
        {"bg": (255, 255, 255), "icon_bg": (245, 245, 250), "cols": 20, "spacing": 55, "icon_size": (45, 30), "y_start": 5, "label": "compact bar"},
        {"bg": (35, 40, 50), "icon_bg": (55, 60, 70), "cols": 10, "spacing": 70, "icon_size": (55, 55), "y_start": 45, "label": "dark grid"},
        {"bg": (248, 248, 252), "icon_bg": (230, 235, 245), "cols": 6, "spacing": 140, "icon_size": (120, 40), "y_start": 30, "label": "wide buttons"},
        {"bg": (60, 60, 65), "icon_bg": (80, 85, 95), "cols": 14, "spacing": 65, "icon_size": (50, 35), "y_start": 20, "label": "IDE toolbar"},
        {"bg": (240, 238, 235), "icon_bg": (255, 255, 255), "cols": 10, "spacing": 90, "icon_size": (70, 45), "y_start": 50, "label": "card grid"},
        {"bg": (45, 48, 55), "icon_bg": (65, 68, 78), "cols": 16, "spacing": 60, "icon_size": (48, 32), "y_start": 15, "label": "status bar"},
        {"bg": (250, 250, 250), "icon_bg": (235, 238, 245), "cols": 11, "spacing": 85, "icon_size": (65, 38), "y_start": 40, "label": "menu strip"},
    ]
    style = styles[idx]

    img = Image.new("RGB", (IMG_W, IMG_H), style["bg"])
    draw = ImageDraw.Draw(img)

    # アイコンプール（各画像でシャッフル順を変える）— テキストラベルで描画（絵文字フォント不要）
    icon_names = [
        "Cut", "Edit", "Copy", "Attach", "Search", "Refresh", "Undo", "Redo",
        "Up", "Down", "Note", "Doc", "Folder", "Save", "Print", "Chart",
        "Graph", "Link", "Paint", "Image", "Ruler", "Grid", "Check", "Close",
        "Circle", "Play", "Pause", "Stop", "Bell", "Lock",
    ]
    rng = random.Random(42 + idx * 7)
    rng.shuffle(icon_names)

    # アイコンごとに簡単な図形を描画する色マップ
    icon_colors = [
        (180, 80, 80), (80, 130, 180), (80, 160, 80), (180, 140, 60),
        (140, 80, 160), (80, 160, 160), (160, 100, 60), (100, 100, 180),
        (160, 60, 120), (60, 140, 100),
    ]

    cols = style["cols"]
    iw, ih = style["icon_size"]
    sp = style["spacing"]
    y_start = style["y_start"]
    font_size = max(8, min(12, iw // 5))
    font_label = _get_font(font_size)
    text_color = (200, 200, 220) if style["bg"][0] < 128 else (40, 40, 60)

    icons = []
    for i, lbl in enumerate(icon_names):
        row = i // cols
        col = i % cols
        ix = 15 + col * sp
        iy = y_start + row * (ih + 10)
        if ix + iw > IMG_W - 10 or iy + ih > IMG_H - 10:
            continue
        draw.rounded_rectangle([ix, iy, ix + iw, iy + ih], radius=3, fill=style["icon_bg"])
        # アイコン風の小さい図形
        ic_color = icon_colors[i % len(icon_colors)]
        margin = max(4, min(iw, ih) // 5)
        draw.rounded_rectangle(
            [ix + margin, iy + margin, ix + iw - margin, iy + ih - margin],
            radius=2, fill=ic_color)
        # ラベルテキスト
        draw.text((ix + margin + 2, iy + margin + 1), lbl[:4], fill=(255, 255, 255), font=font_label)
        icons.append({"label": lbl, "cx": ix + iw // 2, "cy": iy + ih // 2})

    target = icons[idx % len(icons)]
    return {
        "image": img,
        "target": f"the center of the '{target['label']}' icon in the {style['label']}",
        "x": target["cx"],
        "y": target["cy"],
    }


# ============================================================
# カテゴリ 3: 非UI（自然画像）
# ============================================================
def _gen_natural(idx: int) -> dict:
    """自然画像風の合成画像 — 幾何学図形やグラデーションで「意味的な点」を作る"""
    img = Image.new("RGB", (IMG_W, IMG_H), (30, 60, 90))
    draw = ImageDraw.Draw(img)

    targets_spec = [
        ("tip of the red triangle", "red_triangle_tip"),
        ("center of the yellow circle", "yellow_circle_center"),
        ("intersection of the two diagonal lines", "line_intersection"),
        ("top of the green mountain peak", "mountain_peak"),
        ("center of the small white star", "star_center"),
        ("eye of the spiral pattern", "spiral_eye"),
        ("tip of the arrow pointing right", "arrow_tip"),
        ("center of the bullseye", "bullseye_center"),
        ("brightest point of the gradient", "gradient_peak"),
        ("corner where three shapes meet", "shape_corner"),
    ]

    spec = targets_spec[idx]
    target_desc, target_id = spec

    # 背景にノイズ風のランダム矩形
    for _ in range(50):
        rx = random.randint(0, IMG_W - 100)
        ry = random.randint(0, IMG_H - 100)
        rw = random.randint(30, 150)
        rh = random.randint(30, 150)
        c = (random.randint(20, 80), random.randint(40, 100), random.randint(60, 120))
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=c, outline=None)

    # ターゲット固有の描画
    tx, ty = 0, 0

    if target_id == "red_triangle_tip":
        cx, cy = 960, 600
        pts = [(cx, cy - 150), (cx - 130, cy + 100), (cx + 130, cy + 100)]
        draw.polygon(pts, fill=(220, 40, 40))
        tx, ty = cx, cy - 150  # 頂点

    elif target_id == "yellow_circle_center":
        cx, cy = 700, 400
        r = 80
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(240, 220, 50))
        tx, ty = cx, cy

    elif target_id == "line_intersection":
        draw.line([(200, 100), (1700, 900)], fill=(200, 200, 200), width=3)
        draw.line([(1700, 100), (200, 900)], fill=(200, 200, 200), width=3)
        tx, ty = 950, 500

    elif target_id == "mountain_peak":
        peak_x, peak_y = 800, 300
        pts = [(400, 800), (peak_x, peak_y), (1200, 800)]
        draw.polygon(pts, fill=(40, 150, 60))
        tx, ty = peak_x, peak_y

    elif target_id == "star_center":
        cx, cy = 1400, 350
        # 5-pointed star
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            r = 40 if i % 2 == 0 else 18
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(points, fill=(255, 255, 255))
        tx, ty = cx, cy

    elif target_id == "spiral_eye":
        cx, cy = 500, 700
        for i in range(200):
            angle = i * 0.15
            r = i * 0.8
            px = int(cx + r * math.cos(angle))
            py = int(cy + r * math.sin(angle))
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(150, 180, 220))
        tx, ty = cx, cy

    elif target_id == "arrow_tip":
        cx, cy = 1200, 550
        # 矢印の本体
        draw.line([(800, 550), (cx, cy)], fill=(255, 150, 50), width=5)
        # 矢じり
        draw.polygon([(cx, cy - 20), (cx + 40, cy), (cx, cy + 20)], fill=(255, 150, 50))
        tx, ty = cx + 40, cy  # 矢じりの先端

    elif target_id == "bullseye_center":
        cx, cy = 1000, 500
        for r in [100, 70, 40, 15]:
            color = (220, 50, 50) if r % 2 == 0 else (255, 255, 255)
            if r == 100:
                color = (220, 50, 50)
            elif r == 70:
                color = (255, 255, 255)
            elif r == 40:
                color = (220, 50, 50)
            else:
                color = (255, 255, 255)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(220, 50, 50))
        tx, ty = cx, cy

    elif target_id == "gradient_peak":
        # 放射状グラデーション
        cx, cy = 600, 300
        for r in range(200, 0, -1):
            brightness = int(255 * (1 - r / 200))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=(brightness, brightness, brightness + 30))
        tx, ty = cx, cy

    elif target_id == "shape_corner":
        # 3つの図形が角で接する
        cx, cy = 1100, 700
        draw.rectangle([cx, cy, cx + 120, cy + 120], fill=(60, 60, 180))
        draw.polygon([(cx, cy), (cx - 100, cy + 120), (cx, cy + 120)], fill=(60, 180, 60))
        draw.ellipse([cx - 40, cy - 80, cx + 40, cy], fill=(180, 60, 60))
        tx, ty = cx, cy

    return {
        "image": img,
        "target": f"the {target_desc}",
        "x": tx,
        "y": ty,
    }


# ============================================================
# カテゴリ 4: 曖昧 / 困難
# ============================================================
def _gen_ambiguous(idx: int) -> dict:
    """低コントラスト、小ターゲット、曖昧な記述"""
    img = Image.new("RGB", (IMG_W, IMG_H), (128, 128, 128))
    draw = ImageDraw.Draw(img)

    targets_spec = [
        # 低コントラスト
        ("the slightly darker square", (500, 400), lambda d: d.rectangle([480, 380, 540, 440], fill=(118, 118, 118))),
        ("the faint circle in the center", (960, 540), lambda d: d.ellipse([930, 510, 990, 570], fill=(135, 135, 135))),
        # 極小ターゲット
        ("the tiny red dot", (1200, 300), lambda d: d.ellipse([1197, 297, 1203, 303], fill=(200, 60, 60))),
        ("the small cross mark", (800, 700), lambda d: [d.line([(795, 700), (805, 700)], fill=(180, 80, 80), width=2),
                                                         d.line([(800, 695), (800, 705)], fill=(180, 80, 80), width=2)]),
        # テクスチャ上
        ("the single blue pixel among the noise", (650, 500), None),
        # 曖昧な記述
        ("the button on the right side", (1600, 60), lambda d: [
            d.rounded_rectangle([1550, 40, 1680, 80], radius=5, fill=(100, 130, 200)),
            d.text((1570, 50), "Apply", fill="white", font=_get_font(14))
        ]),
        ("the icon near the bottom", (960, 950), lambda d: d.rounded_rectangle([940, 930, 980, 970], radius=3, fill=(80, 160, 80))),
        # 同種要素の混在
        ("the second circle from the left", (400, 540), lambda d: [
            d.ellipse([180, 520, 240, 580], fill=(200, 200, 60)),
            d.ellipse([380, 520, 440, 580], fill=(200, 200, 60)),
            d.ellipse([580, 520, 640, 580], fill=(200, 200, 60)),
        ]),
        ("the green button, not the blue one", (300, 200), lambda d: [
            d.rounded_rectangle([260, 180, 360, 230], radius=5, fill=(60, 160, 60)),
            d.rounded_rectangle([460, 180, 560, 230], radius=5, fill=(60, 60, 160)),
            d.text((275, 195), "OK", fill="white", font=_get_font(14)),
            d.text((480, 195), "Cancel", fill="white", font=_get_font(14)),
        ]),
        # グレー on グレー
        ("the outlined rectangle on the gray background", (960, 540),
         lambda d: d.rectangle([910, 490, 1010, 590], outline=(145, 145, 145), width=2)),
    ]

    spec = targets_spec[idx]
    target_desc, (tx, ty), draw_fn = spec

    # ノイズテクスチャ
    if idx == 4:  # blue pixel among noise
        import numpy as np
        arr = np.random.randint(120, 140, (IMG_H, IMG_W, 3), dtype=np.uint8)
        arr[ty, tx] = [60, 60, 220]  # blue pixel
        img = Image.fromarray(arr)
    else:
        # 軽いノイズ
        import numpy as np
        arr = np.array(img)
        noise = np.random.randint(-5, 6, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        if draw_fn:
            draw_fn(draw)

    return {
        "image": img,
        "target": f"{target_desc}",
        "x": tx,
        "y": ty,
    }


# ============================================================
# カテゴリ 5: precision パラメータテスト
# ============================================================
def _gen_precision(idx: int) -> dict:
    """精度要求の異なるターゲット — 極小から中サイズまで"""
    img = Image.new("RGB", (IMG_W, IMG_H), (200, 200, 205))
    draw = ImageDraw.Draw(img)

    # 背景にランダムな図形でリアルな文脈を作る
    rng = random.Random(42 + idx * 13)
    for _ in range(30):
        rx, ry = rng.randint(0, IMG_W - 80), rng.randint(0, IMG_H - 80)
        rw, rh = rng.randint(20, 120), rng.randint(20, 120)
        c = (rng.randint(150, 220), rng.randint(150, 220), rng.randint(150, 220))
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=c)

    specs = [
        # (target_desc, draw_fn, tx, ty) — サイズ別: 2px ~ 60px
        ("the single red pixel",
         lambda d: d.point((1100, 400), fill=(220, 40, 40)),
         1100, 400),

        ("the tip of the thin blue needle pointing up",
         lambda d: d.polygon([(750, 280), (747, 330), (753, 330)], fill=(40, 60, 200)),
         750, 280),

        ("the center of the tiny green dot",
         lambda d: d.ellipse([898, 598, 906, 606], fill=(40, 180, 60)),
         902, 602),

        ("the exact center of the small orange circle",
         lambda d: d.ellipse([480, 340, 500, 360], fill=(230, 140, 30)),
         490, 350),

        ("the crosshair intersection",
         lambda d: [d.line([(1300, 680), (1340, 680)], fill=(200, 40, 40), width=1),
                    d.line([(1320, 660), (1320, 700)], fill=(200, 40, 40), width=1)],
         1320, 680),

        ("the center of the purple diamond shape",
         lambda d: d.polygon([(600, 730), (625, 705), (650, 730), (625, 755)], fill=(140, 60, 180)),
         625, 730),

        ("the top-left corner of the white square",
         lambda d: d.rectangle([350, 500, 400, 550], fill=(255, 255, 255), outline=(100, 100, 100)),
         350, 500),

        ("the right endpoint of the horizontal yellow line",
         lambda d: d.line([(800, 200), (920, 200)], fill=(220, 200, 40), width=3),
         920, 200),

        ("the center of the innermost ring of the target",
         lambda d: [d.ellipse([1400-r, 850-r, 1400+r, 850+r],
                              fill=(220, 50, 50) if i % 2 == 0 else (255, 255, 255))
                    for i, r in enumerate([50, 35, 20, 8])],
         1400, 850),

        ("the nose of the small arrow pointing right",
         lambda d: [d.line([(250, 150), (300, 150)], fill=(50, 50, 50), width=2),
                    d.polygon([(300, 140), (320, 150), (300, 160)], fill=(50, 50, 50))],
         320, 150),
    ]

    target_desc, draw_fn, tx, ty = specs[idx]
    draw_fn(draw)

    return {
        "image": img,
        "target": f"the {target_desc}" if not target_desc.startswith("the ") else target_desc,
        "x": tx,
        "y": ty,
    }


# ============================================================
# メイン生成
# ============================================================
CATEGORIES = {
    "ui_button": (_gen_ui_button, 10),
    "ui_dense": (_gen_ui_dense, 10),
    "natural": (_gen_natural, 10),
    "ambiguous": (_gen_ambiguous, 10),
    "precision": (_gen_precision, 10),
}


def generate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth = []
    total = 0

    for cat_name, (gen_fn, count) in CATEGORIES.items():
        for i in range(count):
            result = gen_fn(i)
            img_name = f"{cat_name}_{i:02d}.png"
            img_path = OUTPUT_DIR / img_name
            result["image"].save(img_path)

            gt_entry = {
                "id": f"{cat_name}_{i:02d}",
                "category": cat_name,
                "image": f"images/{img_name}",
                "target": result["target"],
                "x": result["x"],
                "y": result["y"],
            }
            ground_truth.append(gt_entry)
            total += 1
            desc = result['target'][:60].encode('ascii', 'replace').decode('ascii')
            print(f"  [{total:3d}] {cat_name}_{i:02d}: ({result['x']:4d}, {result['y']:4d}) - {desc}")

    with open(GT_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {total} images generated in {OUTPUT_DIR}")
    print(f"Ground truth: {GT_FILE}")
    return ground_truth


if __name__ == "__main__":
    generate_all()
