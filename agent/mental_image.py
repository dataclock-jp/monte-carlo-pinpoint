"""
mental_image.py
脳内画像（メンタルイメージ）モジュール

人間の「運動イメージ」に着想を得た、操作前シミュレーション＆操作後検証システム。
オフスクリーンの PIL キャンバス上で操作を「脳内リハーサル」し、
実際の操作結果と比較して精度を検証・補正する。

主な用途:
  - クリック前にターゲット周辺をキャプチャし、クリック後に位置ずれを検出
  - ずれが閾値を超えた場合、正しい座標に補正移動
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image


@dataclass
class VerificationResult:
    """操作検証の結果。"""
    position_ok: bool           # カーソルが閾値内に着地したか
    actual_x: int               # 実際のカーソル X
    actual_y: int               # 実際のカーソル Y
    intended_x: int             # 意図した X
    intended_y: int             # 意図した Y
    distance_px: float          # 意図 vs 実際のユークリッド距離
    context_changed: bool       # 周辺ピクセルに変化があったか
    diff_score: float           # クロップ領域の差分スコア (0-1)
    corrected: bool = False     # 補正移動を実施したか
    assessment: str = ""        # 人間可読なサマリ


class MentalImage:
    """
    オフスクリーン画像による操作シミュレーション＆検証。

    Parameters
    ----------
    crop_size : int
        検証用クロップの一辺 (px)。128px でUI要素1つ分をカバー。
    position_threshold : int
        位置ずれの許容ピクセル数。
    """

    def __init__(
        self,
        crop_size: int = 128,
        position_threshold: int = 5,
    ):
        self.crop_size = crop_size
        self.position_threshold = position_threshold

    def capture_context(self, x: int, y: int, monitor_index: int = 0) -> Image.Image:
        """
        指定座標の周辺領域をスクリーンからクロップして返す。

        mss の region 指定で必要最小限だけキャプチャする。
        """
        half = self.crop_size // 2
        # mss で領域キャプチャ（絶対座標）
        import mss
        with mss.mss() as sct:
            # モニター情報からスクリーン境界を取得
            if monitor_index > 0 and monitor_index < len(sct.monitors):
                mon = sct.monitors[monitor_index]
            else:
                mon = sct.monitors[0]  # 全画面

            # クロップ領域をスクリーン境界内にクランプ
            left = max(mon["left"], x - half)
            top = max(mon["top"], y - half)
            right = min(mon["left"] + mon["width"], x + half)
            bottom = min(mon["top"] + mon["height"], y + half)
            width = right - left
            height = bottom - top

            if width <= 0 or height <= 0:
                return Image.new("RGB", (self.crop_size, self.crop_size), (0, 0, 0))

            region = {"left": left, "top": top, "width": width, "height": height}
            shot = sct.grab(region)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return img

    def get_cursor_position(self) -> Tuple[int, int]:
        """OS が報告するカーソル位置を返す。"""
        import pyautogui
        return pyautogui.position()

    def verify_position(self, intended_x: int, intended_y: int) -> VerificationResult:
        """
        カーソルが意図した位置にあるか検証する。

        Returns
        -------
        VerificationResult
            位置検証の結果（視覚コンテキスト比較は含まない）
        """
        actual_x, actual_y = self.get_cursor_position()
        dist = ((actual_x - intended_x) ** 2 + (actual_y - intended_y) ** 2) ** 0.5
        ok = dist <= self.position_threshold

        if ok:
            assessment = "OK"
        else:
            assessment = f"位置ずれ: {dist:.1f}px（意図({intended_x},{intended_y}) → 実際({actual_x},{actual_y})）"

        return VerificationResult(
            position_ok=ok,
            actual_x=actual_x,
            actual_y=actual_y,
            intended_x=intended_x,
            intended_y=intended_y,
            distance_px=dist,
            context_changed=False,
            diff_score=0.0,
            assessment=assessment,
        )

    def compare_context(
        self, before: Image.Image, after: Image.Image
    ) -> Tuple[bool, float]:
        """
        操作前後のクロップ画像を比較する。

        Returns
        -------
        (changed, diff_score) : Tuple[bool, float]
            changed: 視覚的変化があったか
            diff_score: 正規化された差分スコア (0-1)
        """
        # サイズを揃える
        if before.size != after.size:
            after = after.resize(before.size)
        arr_before = np.array(before, dtype=np.float32)
        arr_after = np.array(after, dtype=np.float32)
        diff = np.abs(arr_before - arr_after).mean() / 255.0
        return diff > 0.005, float(diff)

    def verify_click(
        self,
        intended_x: int,
        intended_y: int,
        before_crop: Image.Image,
        monitor_index: int = 0,
    ) -> VerificationResult:
        """
        クリック後の位置と視覚コンテキストを検証する。

        Parameters
        ----------
        intended_x, intended_y : int
            意図したクリック座標
        before_crop : PIL.Image
            クリック前にキャプチャしたターゲット周辺
        monitor_index : int
            キャプチャ対象モニター

        Returns
        -------
        VerificationResult
            位置 + 視覚コンテキストの総合検証結果
        """
        # 位置検証
        pos_result = self.verify_position(intended_x, intended_y)

        # 視覚コンテキスト検証
        after_crop = self.capture_context(intended_x, intended_y, monitor_index)
        changed, diff_score = self.compare_context(before_crop, after_crop)

        # 総合判定
        parts = []
        if pos_result.position_ok:
            parts.append("位置OK")
        else:
            parts.append(f"位置ずれ {pos_result.distance_px:.1f}px")
        if changed:
            parts.append(f"画面変化あり(score={diff_score:.3f})")
        else:
            parts.append("画面変化なし")

        return VerificationResult(
            position_ok=pos_result.position_ok,
            actual_x=pos_result.actual_x,
            actual_y=pos_result.actual_y,
            intended_x=intended_x,
            intended_y=intended_y,
            distance_px=pos_result.distance_px,
            context_changed=changed,
            diff_score=diff_score,
            assessment=" / ".join(parts),
        )

    def correct_position(self, intended_x: int, intended_y: int) -> bool:
        """
        カーソルが意図した位置からずれていた場合、補正移動する。

        Returns
        -------
        bool
            補正を実施した場合 True
        """
        actual_x, actual_y = self.get_cursor_position()
        dist = ((actual_x - intended_x) ** 2 + (actual_y - intended_y) ** 2) ** 0.5
        if dist <= self.position_threshold:
            return False

        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.moveTo(intended_x, intended_y, duration=0.1)
        return True
