"""
mental_canvas.py
汎用メンタルキャンバス（脳内画像）

AIが「考える時に描く」ためのオフスクリーンキャンバスシステム。
画面には表示されず、AIだけが見ることができる視覚的ワーキングスペース。

用途:
  - 操作前のシミュレーション（ここをクリックしたらどうなるか）
  - 複数ステップの計画の視覚的ウォークスルー
  - 描画・配置のプレビュー
  - 思考の可視化（図示、マーキング、注釈）
  - before/after 比較

人間の認知科学における「心的イメージ」「運動イメージ」「ビジュオスパティアル・
スケッチパッド」（Baddeley のワーキングメモリモデル）に対応する。
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import copy
import time


# ---------------------------------------------------------------------------
# カラー定義
# ---------------------------------------------------------------------------

_COLOR_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (255, 165, 0),
    "gray": (128, 128, 128),
    "pink": (255, 192, 203),
}


def _parse_color(color) -> Tuple[int, int, int]:
    """色名または (R,G,B) タプルを RGB タプルに変換する。"""
    if isinstance(color, str):
        return _COLOR_MAP.get(color.lower(), (255, 0, 0))
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return tuple(int(c) for c in color[:3])
    return (255, 0, 0)


# フォントキャッシュ
_font_cache: Dict[int, ImageFont.FreeTypeFont] = {}

# 日本語対応フォントの検索順
_FONT_CANDIDATES = [
    "YuGothB.ttc",       # Yu Gothic Bold（Windows 10/11）
    "YuGothM.ttc",       # Yu Gothic Medium
    "meiryo.ttc",        # Meiryo
    "msgothic.ttc",      # MS Gothic
    "segoeui.ttf",       # Segoe UI（英語フォールバック）
    "arial.ttf",         # Arial（英語フォールバック）
]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """指定サイズのフォントを取得する（日本語対応、キャッシュ付き）。"""
    if size in _font_cache:
        return _font_cache[size]
    for name in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(name, size)
            _font_cache[size] = font
            return font
        except (OSError, IOError):
            continue
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


# ---------------------------------------------------------------------------
# CompareResult
# ---------------------------------------------------------------------------

@dataclass
class CompareResult:
    """キャンバス比較の結果。"""
    diff_score: float       # 正規化差分スコア (0=同一, 1=完全異差)
    changed: bool           # 閾値以上の差があるか
    diff_image: Image.Image # 差分を可視化した画像
    summary: str            # 人間可読なサマリ


# ---------------------------------------------------------------------------
# MentalCanvas
# ---------------------------------------------------------------------------

class MentalCanvas:
    """
    名前付きオフスクリーンキャンバス。

    描画・テキスト・画像合成・スナップショット（undo/比較用）をサポート。
    """

    def __init__(self, width: int, height: int, color=(255, 255, 255), name: str = ""):
        self.name = name
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), _parse_color(color))
        self._snapshots: Dict[str, Image.Image] = {}
        self.created_at = time.time()

    # ---- 描画 ----

    def draw_rect(self, x: int, y: int, w: int, h: int,
                  color="red", fill: bool = False, width: int = 2) -> None:
        """矩形を描画する。"""
        draw = ImageDraw.Draw(self.image)
        coords = [x, y, x + w, y + h]
        if fill:
            draw.rectangle(coords, fill=_parse_color(color))
        else:
            draw.rectangle(coords, outline=_parse_color(color), width=width)

    def draw_circle(self, cx: int, cy: int, r: int,
                    color="red", fill: bool = False, width: int = 2) -> None:
        """円を描画する。"""
        draw = ImageDraw.Draw(self.image)
        coords = [cx - r, cy - r, cx + r, cy + r]
        if fill:
            draw.ellipse(coords, fill=_parse_color(color))
        else:
            draw.ellipse(coords, outline=_parse_color(color), width=width)

    def draw_line(self, x1: int, y1: int, x2: int, y2: int,
                  color="red", width: int = 2) -> None:
        """線を描画する。"""
        draw = ImageDraw.Draw(self.image)
        draw.line([(x1, y1), (x2, y2)], fill=_parse_color(color), width=width)

    def draw_arrow(self, x1: int, y1: int, x2: int, y2: int,
                   color="red", width: int = 2, head_size: int = 12) -> None:
        """矢印を描画する。"""
        draw = ImageDraw.Draw(self.image)
        c = _parse_color(color)
        draw.line([(x1, y1), (x2, y2)], fill=c, width=width)
        # 矢印の先端
        import math
        angle = math.atan2(y2 - y1, x2 - x1)
        for da in [2.5, -2.5]:  # ±約143度
            ax = x2 - head_size * math.cos(angle + da)
            ay = y2 - head_size * math.sin(angle + da)
            draw.line([(x2, y2), (int(ax), int(ay))], fill=c, width=width)

    def draw_text(self, x: int, y: int, text: str,
                  color="black", size: int = 16, background: str = "") -> None:
        """テキストを描画する。"""
        draw = ImageDraw.Draw(self.image)
        font = _get_font(size)

        if background:
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle(bbox, fill=_parse_color(background))

        draw.text((x, y), text, fill=_parse_color(color), font=font)

    def draw_marker(self, x: int, y: int, label: str = "",
                    color="red", size: int = 16) -> None:
        """十字マーカー（+ラベル）を描画する。"""
        draw = ImageDraw.Draw(self.image)
        c = _parse_color(color)
        draw.line([(x - size, y), (x + size, y)], fill=c, width=2)
        draw.line([(x, y - size), (x, y + size)], fill=c, width=2)
        if label:
            self.draw_text(x + size + 4, y - 8, label, color=color, size=14,
                           background="white")

    def draw_numbered_marker(self, x: int, y: int, number: int,
                             color="blue", size: int = 24) -> None:
        """番号付き円マーカーを描画する（計画ステップの位置表示用）。
        塗りつぶし円の中央に白抜きの番号を描画する。
        """
        draw = ImageDraw.Draw(self.image)
        c = _parse_color(color)
        # 塗りつぶし円
        draw.ellipse([x - size, y - size, x + size, y + size], fill=c,
                     outline="white", width=2)
        # 白抜き番号テキスト
        label = str(number)
        font = _get_font(int(size * 1.2))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x - tw // 2, y - th // 2 - 2), label,
                  fill="white", font=font)

    # ---- 画像操作 ----

    def paste(self, source: Image.Image, x: int = 0, y: int = 0) -> None:
        """別の画像を貼り付ける。"""
        self.image.paste(source, (x, y))

    def paste_canvas(self, other: 'MentalCanvas', x: int = 0, y: int = 0) -> None:
        """別のキャンバスを貼り付ける。"""
        self.image.paste(other.image, (x, y))

    def crop(self, x: int, y: int, w: int, h: int) -> Image.Image:
        """領域を切り出して PIL Image として返す。"""
        return self.image.crop((x, y, x + w, y + h))

    def resize(self, new_width: int, new_height: int) -> None:
        """キャンバスをリサイズする。"""
        self.image = self.image.resize((new_width, new_height), Image.LANCZOS)
        self.width = new_width
        self.height = new_height

    def clear(self, color=(255, 255, 255)) -> None:
        """キャンバスを塗りつぶしてクリアする。"""
        self.image = Image.new("RGB", (self.width, self.height), _parse_color(color))

    # ---- 解析 (Analysis) ----

    def detect_white_region(
        self,
        white_threshold: int = 240,
        min_area_ratio: float = 0.05,
    ) -> Optional[Tuple[int, int, int, int]]:
        """キャンバス画像内の最大の明るい矩形領域を検出する。
        CSP等のペイントソフトのキャンバス領域を自動検出するために使う。
        白 (>240) で見つからない場合、閾値を段階的に下げて灰色キャンバスも検出する。

        Returns: (x, y, w, h) or None if not found.
        """
        import cv2
        arr = np.array(self.image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # 閾値を段階的に下げて試行（白→明灰→灰）
        thresholds = [white_threshold, 200, 160]
        for thresh in thresholds:
            result = self._detect_bright_region(gray, thresh, min_area_ratio)
            if result is not None:
                return result
        return None

    def _detect_bright_region(
        self,
        gray: "np.ndarray",
        threshold: int,
        min_area_ratio: float,
    ) -> Optional[Tuple[int, int, int, int]]:
        """指定閾値以上の明るい領域のうち最大の矩形を返す。"""
        import cv2
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        total_area = self.width * self.height
        best = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < total_area * min_area_ratio:
                continue
            if area > best_area:
                best_area = area
                best = cv2.boundingRect(cnt)
        return best

    @staticmethod
    def generate_shape_points(
        shape: str,
        cx: int, cy: int,
        r: int = 0,
        w: int = 0, h: int = 0,
        num_points: int = 36,
    ) -> List[Tuple[int, int]]:
        """図形の輪郭に沿った座標点列を生成する。

        Args:
            shape: "circle" or "ellipse"
            cx, cy: 中心座標
            r: 半径 (circle用)
            w, h: 幅と高さ (ellipse用)
            num_points: 点の数 (多いほど滑らか)

        Returns: [(x1,y1), (x2,y2), ...] — 閉じたループ (最後=最初)
        """
        import math
        points = []
        if shape == "circle":
            rx = ry = r
        elif shape == "ellipse":
            rx = w // 2
            ry = h // 2
        else:
            raise ValueError(f"未対応の形状: {shape}")
        for i in range(num_points):
            angle = 2.0 * math.pi * i / num_points
            x = int(cx + rx * math.cos(angle))
            y = int(cy + ry * math.sin(angle))
            points.append((x, y))
        # ループを閉じる
        points.append(points[0])
        return points

    # ---- スナップショット（状態保存/復元）----

    def snapshot(self, label: str = "default") -> None:
        """現在の状態をスナップショットとして保存する。"""
        self._snapshots[label] = self.image.copy()

    def restore(self, label: str = "default") -> bool:
        """スナップショットから復元する。"""
        if label not in self._snapshots:
            return False
        self.image = self._snapshots[label].copy()
        return True

    def list_snapshots(self) -> List[str]:
        """保存されたスナップショットのラベル一覧を返す。"""
        return list(self._snapshots.keys())

    # ---- 比較 ----

    def compare_with(self, other: 'MentalCanvas', threshold: float = 0.01) -> CompareResult:
        """別のキャンバスと比較する。"""
        return self.compare_images(self.image, other.image, threshold)

    def compare_with_snapshot(self, label: str = "default",
                              threshold: float = 0.01) -> Optional[CompareResult]:
        """スナップショットと現在の状態を比較する。"""
        if label not in self._snapshots:
            return None
        return self.compare_images(self._snapshots[label], self.image, threshold)

    @staticmethod
    def compare_images(img1: Image.Image, img2: Image.Image,
                       threshold: float = 0.01) -> CompareResult:
        """2つの画像を比較し、差分スコアと差分画像を返す。"""
        # サイズを揃える
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.LANCZOS)

        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        diff_arr = np.abs(arr1 - arr2)
        score = float(diff_arr.mean() / 255.0)

        # 差分可視化（差分を強調表示）
        diff_vis = np.clip(diff_arr * 3, 0, 255).astype(np.uint8)
        diff_image = Image.fromarray(diff_vis)

        changed = score > threshold
        if changed:
            summary = f"変化あり (score={score:.4f})"
        else:
            summary = f"変化なし (score={score:.4f})"

        return CompareResult(
            diff_score=score,
            changed=changed,
            diff_image=diff_image,
            summary=summary,
        )

    # ---- 出力 ----

    def get_image(self) -> Image.Image:
        """キャンバスの現在の画像を返す。"""
        return self.image.copy()

    def to_bytes(self, format: str = "PNG") -> bytes:
        """画像をバイト列として返す。"""
        buf = BytesIO()
        self.image.save(buf, format=format)
        return buf.getvalue()

    def __repr__(self) -> str:
        snaps = len(self._snapshots)
        return f"MentalCanvas('{self.name}', {self.width}x{self.height}, snapshots={snaps})"


# ---------------------------------------------------------------------------
# CanvasManager
# ---------------------------------------------------------------------------

class CanvasManager:
    """
    複数の名前付きキャンバスを管理するマネージャ。
    """

    def __init__(self):
        self._canvases: Dict[str, MentalCanvas] = {}

    def create(self, name: str, width: int = 800, height: int = 600,
               color="white") -> MentalCanvas:
        """新しいキャンバスを作成する。"""
        canvas = MentalCanvas(width, height, color, name=name)
        self._canvases[name] = canvas
        return canvas

    def from_image(self, name: str, image: Image.Image) -> MentalCanvas:
        """PIL Image からキャンバスを作成する。"""
        canvas = MentalCanvas(image.width, image.height, name=name)
        canvas.image = image.copy()
        self._canvases[name] = canvas
        return canvas

    def from_screenshot(self, name: str, monitor: int = 0,
                        region: Optional[Tuple[int, int, int, int]] = None) -> MentalCanvas:
        """スクリーンキャプチャからキャンバスを作成する。"""
        from agent.screen_capture import ScreenCapture
        cap = ScreenCapture(monitor_index=monitor, region=region)
        pil_img = cap.capture_as_pil()
        return self.from_image(name, pil_img)

    def from_file(self, name: str, path: str) -> MentalCanvas:
        """画像ファイルからキャンバスを作成する。"""
        pil_img = Image.open(path).convert("RGB")
        return self.from_image(name, pil_img)

    def get(self, name: str) -> Optional[MentalCanvas]:
        """名前でキャンバスを取得する。"""
        return self._canvases.get(name)

    def delete(self, name: str) -> bool:
        """キャンバスを削除する。"""
        if name in self._canvases:
            del self._canvases[name]
            return True
        return False

    def list(self) -> List[str]:
        """全キャンバス名を返す。"""
        return list(self._canvases.keys())

    def list_detail(self) -> List[dict]:
        """全キャンバスの詳細情報を返す。"""
        return [
            {
                "name": c.name,
                "size": f"{c.width}x{c.height}",
                "snapshots": len(c._snapshots),
            }
            for c in self._canvases.values()
        ]

    def clear_all(self) -> int:
        """全キャンバスを削除する。"""
        count = len(self._canvases)
        self._canvases.clear()
        return count

    def __len__(self) -> int:
        return len(self._canvases)

    def __repr__(self) -> str:
        return f"CanvasManager({len(self._canvases)} canvases)"
