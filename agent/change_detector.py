"""
change_detector.py
Video2AIの変化検知アルゴリズムをリアルタイムストリームに適用するモジュール

screenshot.py の diff/ssim/optical-flow アルゴリズムを
「動画フレーム」ではなく「リアルタイムキャプチャフレーム」に適用する。

method="or-all" を指定すると3方式を並列実行し、
どれか1つでも閾値超えで変化検出（OR条件）。
"""
import sys
import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Literal

# video2aiのルートディレクトリをパスに追加
_VIDEO2AI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VIDEO2AI_ROOT not in sys.path:
    sys.path.insert(0, _VIDEO2AI_ROOT)

from screenshot import _diff_score, _ssim_score, _optical_flow_score

ChangeMethod = Literal["diff", "ssim", "optical-flow", "or-all"]

# 3方式の構成要素
_OR_ALL_METHODS = ("diff", "ssim", "optical-flow")


class ChangeDetector:
    """
    リアルタイム画面変化検知クラス。

    Video2AIのコア変化検知アルゴリズム（diff/ssim/optical-flow）を
    連続するフレームペアに適用し、変化スコアと変化有無を返す。

    method="or-all" で3方式を並列実行し、どれか1つでも閾値超えで
    変化と判定（OR条件）。各方式の得意分野:
      diff=色変化、ssim=構造変化、optical-flow=動き

    Attributes
    ----------
    method : str
        変化検知手法（"diff", "ssim", "optical-flow", "or-all"）
    threshold : float
        変化と判定するスコア閾値（単一方式の場合）
    thresholds : dict
        方式ごとの個別閾値（or-all の場合に使用）
    resize_for_diff : int
        差分計算時のリサイズ幅（高速化のため小さくする）
    """

    # 手法ごとの推奨デフォルト閾値
    DEFAULT_THRESHOLDS = {
        "diff": 0.02,
        "ssim": 0.05,
        "optical-flow": 0.03,
    }

    def __init__(
        self,
        method: ChangeMethod = "diff",
        threshold: Optional[float] = None,
        resize_for_diff: int = 320,
        thresholds: Optional[dict[str, float]] = None,
    ):
        self.method = method
        self.resize_for_diff = resize_for_diff
        self._prev_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._change_count = 0
        # 最新の方式別スコア（or-all 時に参照可能）
        self._last_scores: dict[str, float] = {}

        if method == "or-all":
            # or-all: 方式ごとの個別閾値
            self.thresholds = {**self.DEFAULT_THRESHOLDS}
            if thresholds:
                self.thresholds.update(thresholds)
            self.threshold = 0.0  # or-all では使わない（個別閾値で判定）
        else:
            self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLDS.get(method, 0.02)
            self.thresholds = {method: self.threshold}

    def reset(self):
        """検知状態をリセットする（新セッション開始時に呼ぶ）。"""
        self._prev_frame = None
        self._frame_count = 0
        self._change_count = 0
        self._last_scores = {}

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """差分計算用にフレームをリサイズする。"""
        if self.resize_for_diff > 0 and frame.shape[1] > self.resize_for_diff:
            h, w = frame.shape[:2]
            new_h = int(h * self.resize_for_diff / w)
            return cv2.resize(frame, (self.resize_for_diff, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def _compute_single(self, method: str, prev_small: np.ndarray, curr_small: np.ndarray,
                         prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """指定した1方式のスコアを計算する。"""
        if method == "diff":
            return _diff_score(prev_small, curr_small)
        elif method == "ssim":
            return _ssim_score(prev_gray, curr_gray)
        elif method == "optical-flow":
            return _optical_flow_score(prev_gray, curr_gray)
        return 0.0

    def compute_score(self, frame: np.ndarray) -> float:
        """
        前フレームとの変化スコアを計算する。

        Parameters
        ----------
        frame : np.ndarray
            BGR形式の現在フレーム

        Returns
        -------
        float
            変化スコア（0.0〜1.0）。前フレームがない場合は0.0。
            or-all の場合は最大スコアを返す。
        """
        if self._prev_frame is None:
            self._prev_frame = self._resize(frame)
            self._last_scores = {}
            return 0.0

        curr_small = self._resize(frame)
        prev_small = self._prev_frame

        if self.method == "or-all":
            # 3方式を並列実行
            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {
                    m: pool.submit(self._compute_single, m, prev_small, curr_small, prev_gray, curr_gray)
                    for m in _OR_ALL_METHODS
                }
                self._last_scores = {m: f.result() for m, f in futures.items()}

            score = max(self._last_scores.values())
        else:
            if self.method == "diff":
                score = _diff_score(prev_small, curr_small)
            elif self.method == "ssim":
                prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
                score = _ssim_score(prev_gray, curr_gray)
            elif self.method == "optical-flow":
                prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
                score = _optical_flow_score(prev_gray, curr_gray)
            else:
                score = _diff_score(prev_small, curr_small)
            self._last_scores = {self.method: score}

        self._prev_frame = curr_small
        self._frame_count += 1
        return score

    def is_changed(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        フレームが変化しているかどうかを判定する。

        Parameters
        ----------
        frame : np.ndarray
            BGR形式の現在フレーム

        Returns
        -------
        tuple[bool, float]
            (変化あり/なし, 変化スコア)
            or-all の場合、どれか1方式でも閾値超えなら True。
            スコアは最大スコアを返す。
        """
        score = self.compute_score(frame)

        if self.method == "or-all":
            # OR条件: どれか1つでも閾値超えで変化と判定
            changed = any(
                self._last_scores.get(m, 0.0) >= self.thresholds[m]
                for m in _OR_ALL_METHODS
            )
        else:
            changed = score >= self.threshold

        if changed:
            self._change_count += 1
        return changed, score

    @property
    def last_scores(self) -> dict[str, float]:
        """最新の方式別スコアを返す（デバッグ・ログ用）。"""
        return dict(self._last_scores)

    @property
    def stats(self) -> dict:
        """検知統計を返す。"""
        result = {
            "method": self.method,
            "threshold": self.threshold,
            "frame_count": self._frame_count,
            "change_count": self._change_count,
            "change_rate": self._change_count / max(1, self._frame_count),
        }
        if self.method == "or-all":
            result["thresholds"] = dict(self.thresholds)
            result["last_scores"] = dict(self._last_scores)
        return result
