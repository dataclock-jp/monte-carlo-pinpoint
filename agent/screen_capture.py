"""
screen_capture.py
画面キャプチャモジュール

mss（Multi-Screen Shot）ライブラリを使用してデスクトップ画面を
NumPy配列として取得する。Xvfb仮想ディスプレイにも対応。
"""
import os
import base64
import numpy as np
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image


class ScreenCapture:
    """
    デスクトップ画面をキャプチャするクラス。
    
    mssライブラリを使用し、仮想ディスプレイ（Xvfb）にも対応。
    Windows/Linux/macOS で動作する。
    """

    def __init__(
        self,
        monitor_index: int = 0,
        region: Optional[Tuple[int, int, int, int]] = None,
        resize_width: int = 0,
    ):
        """
        Parameters
        ----------
        monitor_index : int
            キャプチャするモニター番号（0=全画面, 1=プライマリ, ...）
        region : tuple (left, top, width, height) or None
            キャプチャ領域を限定する場合に指定（Noneで全画面）
        resize_width : int
            リサイズ後の幅（0でリサイズなし）
        """
        self.monitor_index = monitor_index
        self.region = region
        self.resize_width = resize_width
        self._sct = None

    def __enter__(self):
        import mss
        self._sct = mss.mss()
        return self

    def __exit__(self, *args):
        if self._sct:
            self._sct.close()
            self._sct = None

    def _get_sct(self):
        if self._sct is None:
            import mss
            self._sct = mss.mss()
        return self._sct

    def capture(self) -> np.ndarray:
        """
        画面をキャプチャしてBGR形式のNumPy配列を返す。
        
        Returns
        -------
        np.ndarray
            BGR形式の画像配列 (H, W, 3)
        """
        import cv2
        sct = self._get_sct()

        if self.region:
            left, top, width, height = self.region
            monitor = {"left": left, "top": top, "width": width, "height": height}
        else:
            monitor = sct.monitors[self.monitor_index]

        screenshot = sct.grab(monitor)
        # mssはBGRA形式で返すのでBGRに変換
        img = np.array(screenshot)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if self.resize_width > 0 and img_bgr.shape[1] != self.resize_width:
            h, w = img_bgr.shape[:2]
            new_h = int(h * self.resize_width / w)
            img_bgr = cv2.resize(img_bgr, (self.resize_width, new_h), interpolation=cv2.INTER_AREA)

        return img_bgr

    def capture_as_pil(self) -> Image.Image:
        """PIL Image形式でキャプチャを返す。"""
        import cv2
        bgr = self.capture()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def capture_as_base64(self, format: str = "JPEG", quality: int = 85) -> str:
        """
        Base64エンコードされた画像文字列を返す（LLM Vision API用）。
        
        Parameters
        ----------
        format : str
            画像フォーマット（"JPEG" or "PNG"）
        quality : int
            JPEG品質（1-95）
        
        Returns
        -------
        str
            data:image/jpeg;base64,... 形式の文字列
        """
        pil_img = self.capture_as_pil()
        buf = BytesIO()
        if format.upper() == "JPEG":
            pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
            mime = "image/jpeg"
        else:
            pil_img.save(buf, format="PNG")
            mime = "image/png"
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def get_screen_size(self) -> Tuple[int, int]:
        """スクリーンサイズ (width, height) を返す。"""
        sct = self._get_sct()
        monitor = sct.monitors[self.monitor_index]
        return monitor["width"], monitor["height"]


# ---------------------------------------------------------------------------
# Grid overlay utilities
# ---------------------------------------------------------------------------

GRID_COLS = 8   # A-H
GRID_ROWS = 5   # 1-5

def _col_label(col: int) -> str:
    """0-indexed column → 'A'-'H'."""
    return chr(ord("A") + col)


def grid_cell_to_image_coords(
    cell: str, img_w: int, img_h: int
) -> Tuple[int, int]:
    """
    グリッドセル名（例: "D3", "D3.7"）→ 画像上の座標 (x, y) を返す。

    サブグリッド（テンキー配列）対応:
        7(左上)  8(上)   9(右上)
        4(左)    5(中央)  6(右)
        1(左下)  2(下)   3(右下)

    サブグリッド省略時は ".5"（中央）と同じ。

    Parameters
    ----------
    cell : str
        "A1" ~ "H5" 形式のセル名、またはサブグリッド付き "A1.7" 形式
    img_w, img_h : int
        LLM送信画像のサイズ

    Returns
    -------
    (x, y) : Tuple[int, int]
        画像座標
    """
    cell = cell.strip().upper()

    # サブグリッド番号を分離（"D3.7" → "D3", 7）
    sub = 5  # デフォルト: 中央
    if "." in cell:
        parts = cell.split(".", 1)
        cell = parts[0]
        try:
            sub = int(parts[1])
            sub = max(1, min(sub, 9))
        except ValueError:
            sub = 5

    col = ord(cell[0]) - ord("A")          # 0-7
    row = int(cell[1:]) - 1                 # 0-4
    col = max(0, min(col, GRID_COLS - 1))
    row = max(0, min(row, GRID_ROWS - 1))
    cell_w = img_w / GRID_COLS
    cell_h = img_h / GRID_ROWS

    # サブグリッド: テンキー配列 → セル内の相対位置 (sx, sy)
    # sx: 0.0=左端, 0.5=中央, 1.0=右端
    # sy: 0.0=上端, 0.5=中央, 1.0=下端
    _SUB_OFFSETS = {
        7: (0.17, 0.17), 8: (0.50, 0.17), 9: (0.83, 0.17),
        4: (0.17, 0.50), 5: (0.50, 0.50), 6: (0.83, 0.50),
        1: (0.17, 0.83), 2: (0.50, 0.83), 3: (0.83, 0.83),
    }
    sx, sy = _SUB_OFFSETS.get(sub, (0.5, 0.5))

    cx = int(cell_w * col + cell_w * sx)
    cy = int(cell_h * row + cell_h * sy)
    return cx, cy


def draw_grid_overlay(frame: np.ndarray) -> np.ndarray:
    """
    BGR画像にグリッドオーバーレイ（罫線＋セルラベル）を描画して返す。
    元画像は変更せず、コピーを返す。

    Parameters
    ----------
    frame : np.ndarray
        BGR画像 (H, W, 3)

    Returns
    -------
    np.ndarray
        グリッド付きBGR画像
    """
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    cell_w = w / GRID_COLS
    cell_h = h / GRID_ROWS

    # 半透明のグリッド線（明るい緑、太さ1）
    line_color = (0, 255, 0)  # BGR: 緑
    # 縦線
    for c in range(1, GRID_COLS):
        x = int(cell_w * c)
        cv2.line(out, (x, 0), (x, h), line_color, 1, cv2.LINE_AA)
    # 横線
    for r in range(1, GRID_ROWS):
        y = int(cell_h * r)
        cv2.line(out, (0, y), (w, y), line_color, 1, cv2.LINE_AA)

    # セルラベルを各セルの左上隅に描画
    font = cv2.FONT_HERSHEY_SIMPLEX
    # フォントサイズは画像幅に比例（1920px → 0.7, 2560px → 0.9）
    font_scale = max(0.5, w / 2800)
    thickness = max(1, int(w / 1500))

    # サブグリッドドットの半径（画像サイズに比例）
    dot_radius = max(2, int(w / 960))

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            label = f"{_col_label(c)}{r + 1}"
            tx = int(cell_w * c + 6)
            ty = int(cell_h * r + 22 * font_scale + 8)
            # 背景矩形（ラベルを読みやすくする）
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(out, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(out, label, (tx, ty), font, font_scale,
                        (0, 255, 0), thickness, cv2.LINE_AA)

            # サブグリッドドット（3×3、中央の5は省略してラベルと被らないようにする）
            for sub in (1, 2, 3, 4, 6, 7, 8, 9):
                # テンキー配列 → セル内相対位置
                sc = (sub - 1) % 3          # 0,1,2
                sr = 2 - (sub - 1) // 3     # 2,1,0 → 上が0
                dx = int(cell_w * c + cell_w * (0.17 + 0.33 * sc))
                dy = int(cell_h * r + cell_h * (0.17 + 0.33 * sr))
                cv2.circle(out, (dx, dy), dot_radius, (0, 200, 0), -1, cv2.LINE_AA)

    return out
