"""
screenshot.py
変化検知スクリーンショット抽出モジュール

動画フレームを逐次比較し、「変化が生じた瞬間」のフレームだけを
HH-MM-SS.SSS 形式のファイル名で保存する。

出力フォーマット: PNG（高品質・可逆）または JPEG（軽量・非可逆）
出力解像度: 元解像度のまま、または指定幅にリサイズ

変化検知手法:
  diff         : BGR全チャンネル絶対差分（高速・デフォルト）
  ssim         : 構造的類似度（SSIM）による変化検知（精度重視）
  optical-flow : Farneback法による密なオプティカルフロー（動体量で判断）
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def seconds_to_filename(seconds: float) -> str:
    """秒数を HH-MM-SS.SSS 形式の文字列に変換する。"""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}-{m:02d}-{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# 変化検知手法の実装
# ---------------------------------------------------------------------------

def _diff_score(prev: np.ndarray, curr: np.ndarray) -> float:
    """
    BGR差分スコアを計算する（現状の手法）。
    戻り値: 0.0〜1.0（大きいほど変化が大きい）
    """
    diff = cv2.absdiff(curr.astype(np.float32), prev.astype(np.float32))
    return float(diff.mean() / 255.0)


def _ssim_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    SSIM（Structural Similarity Index）による変化スコアを計算する。
    OpenCV + NumPy のみで実装（scikit-image 不要）。

    戻り値: 0.0〜1.0（大きいほど変化が大きい = 1 - SSIM）
    SSIM が 1.0 = 完全に同じ、0.0 = 全く異なる。
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    prev_f = prev_gray.astype(np.float64)
    curr_f = curr_gray.astype(np.float64)

    mu1 = cv2.GaussianBlur(prev_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(curr_f, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(prev_f * prev_f, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(curr_f * curr_f, (11, 11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(prev_f * curr_f, (11, 11), 1.5) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-10)
    ssim_val = float(ssim_map.mean())

    # 変化スコアに変換（SSIM が低い = 変化が大きい）
    return max(0.0, 1.0 - ssim_val)


def _optical_flow_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Farneback法による密なオプティカルフローで変化スコアを計算する。

    動きベクトルの大きさ（マグニチュード）の平均を正規化して返す。
    戻り値: 0.0〜1.0（大きいほど動きが大きい）
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    # フローの大きさ（ピクセル単位の移動量）を計算
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    # 画像の対角線長で正規化（0〜1 に収める）
    diag = np.sqrt(prev_gray.shape[0] ** 2 + prev_gray.shape[1] ** 2)
    return float(magnitude.mean() / (diag * 0.1 + 1e-10))


# ---------------------------------------------------------------------------
# メイン関数
# ---------------------------------------------------------------------------

def extract_changed_frames(
    video_path: str,
    output_dir: str,
    threshold: float = 0.02,
    min_interval: float = 0.5,
    resize_for_diff: int = 320,
    image_format: str = "jpg",
    jpeg_quality: int = 85,
    output_width: int = 0,
    interval_mode: float = 0.0,
    change_method: str = "diff",
    verbose: bool = True,
    max_seconds: float = 0.0,
    start_seconds: float = 0.0,
) -> list[dict]:
    """
    動画から変化検知フレームを抽出して保存する。

    Parameters
    ----------
    video_path : str
        入力動画ファイルのパス
    output_dir : str
        スクリーンショット保存先ディレクトリ
    threshold : float
        フレーム間変化スコアの閾値（0〜1）
        手法ごとの推奨デフォルト値:
          diff         : 0.02（2%）
          ssim         : 0.05（SSIM変化量 5%）
          optical-flow : 0.03（動きベクトル量 3%）
    min_interval : float
        変化検知モード時の最小撮影間隔（秒）。連続変化の重複を防ぐ
    resize_for_diff : int
        差分計算時にリサイズする幅（ピクセル）。処理速度向上のため
    image_format : str
        出力画像フォーマット。"jpg"（軽量）または "png"（高品質）
    jpeg_quality : int
        JPEG 保存時の品質（0〜100）。PNG 保存時は無視される
    output_width : int
        出力画像の幅（ピクセル）。0 の場合は元解像度のまま
    interval_mode : float
        0.0 の場合は変化検知モード（デフォルト）
        正の値（秒）を指定すると一定間隔モードに切り替わる
    change_method : str
        変化検知手法:
          "diff"         : BGR差分（高速・デフォルト）
          "ssim"         : 構造的類似度（精度重視・スライド動画に最適）
          "optical-flow" : オプティカルフロー（動体量で判断・スポーツ動画に最適）
          "or-all"       : 3方式OR（diff/ssim/optical-flowを並列実行、どれか1つ閾値超えで検出）
    max_seconds : float
        0.0 の場合は動画全体を処理。正の値を指定するとその秒数まで処理して停止。
        閾値チューニング用のプレビューに使用。
    start_seconds : float
        0.0 の場合は先頭から処理。正の値を指定するとその秒数まで読み飛ばしてから開始。
        動画途中のサンプリングに使用（冒頭が静止画面の場合等）。
    verbose : bool
        進捗バーを表示するか

    Returns
    -------
    list[dict]
        抽出されたフレームの情報リスト
        各要素: {"timestamp": float, "filename": str, "filepath": str}
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 手法の検証
    valid_methods = ("diff", "ssim", "optical-flow", "or-all")
    if change_method not in valid_methods:
        raise ValueError(f"change_method は {valid_methods} のいずれかを指定してください: {change_method}")

    # or-all 用の方式別閾値
    _or_all_thresholds = {"diff": 0.02, "ssim": 0.05, "optical-flow": 0.03}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"動画ファイルを開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # 出力フォーマット設定
    fmt = image_format.lower().strip(".")
    if fmt not in ("jpg", "jpeg", "png"):
        raise ValueError(f"image_format は 'jpg' または 'png' を指定してください: {fmt}")
    ext = ".jpg" if fmt in ("jpg", "jpeg") else ".png"

    # 保存パラメータ
    if ext == ".jpg":
        save_params = [cv2.IMWRITE_JPEG_QUALITY, max(0, min(100, jpeg_quality))]
    else:
        png_level = max(0, min(9, (100 - jpeg_quality) // 10))
        save_params = [cv2.IMWRITE_PNG_COMPRESSION, png_level]

    if verbose:
        mode_str = f"一定間隔 {interval_mode:.1f}s" if interval_mode > 0 else f"変化検知（手法={change_method}, 閾値={threshold:.3f}）"
        fmt_str = f"{ext.upper()} 品質={jpeg_quality}" if ext == ".jpg" else f"{ext.upper()}"
        res_str = f"→ 幅 {output_width}px にリサイズ" if output_width > 0 else "元解像度"
        print(f"[screenshot] 動画情報: {total_frames} フレーム / {fps:.2f} fps / {duration:.1f} 秒")
        print(f"[screenshot] モード: {mode_str} / フォーマット: {fmt_str} / 解像度: {res_str}")

    results = []
    prev_small = None          # diff 用（BGR float32）
    prev_gray_small = None     # ssim / optical-flow 用（グレースケール uint8）
    last_saved_time = -(min_interval if interval_mode <= 0 else interval_mode)

    # start_seconds: 指定秒数まで読み飛ばし
    if start_seconds > 0:
        start_ms = start_seconds * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        if verbose:
            print(f"[screenshot] 開始位置: {start_seconds:.0f}秒からサンプリング")
        # max_seconds は start_seconds からの相対ではなく絶対秒数に変換
        if max_seconds > 0:
            max_seconds = start_seconds + max_seconds

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="フレーム解析", unit="f", disable=not verbose)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # max_seconds 超過で停止
        if max_seconds > 0 and timestamp > max_seconds:
            if verbose:
                print(f"\n[screenshot] プレビュー停止: {max_seconds:.0f}秒まで処理 ({len(results)}枚抽出)")
            break

        should_save = False

        if interval_mode > 0:
            # ---- 一定間隔モード ----
            if (timestamp - last_saved_time) >= interval_mode:
                should_save = True
        else:
            # ---- 変化検知モード ----
            small = cv2.resize(frame, (resize_for_diff, resize_for_diff // 2))

            if change_method == "diff":
                color_small = small.astype(np.float32)
                if prev_small is None:
                    should_save = True
                else:
                    score = _diff_score(prev_small, color_small)
                    if score >= threshold and (timestamp - last_saved_time) >= min_interval:
                        should_save = True
                prev_small = color_small

            elif change_method == "ssim":
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray_small is None:
                    should_save = True
                else:
                    score = _ssim_score(prev_gray_small, gray_small)
                    if score >= threshold and (timestamp - last_saved_time) >= min_interval:
                        should_save = True
                prev_gray_small = gray_small

            elif change_method == "optical-flow":
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray_small is None:
                    should_save = True
                else:
                    score = _optical_flow_score(prev_gray_small, gray_small)
                    if score >= threshold and (timestamp - last_saved_time) >= min_interval:
                        should_save = True
                prev_gray_small = gray_small

            elif change_method == "or-all":
                # 3方式OR: diff=色変化、ssim=構造変化、optical-flow=動き
                color_small = small.astype(np.float32)
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_small is None or prev_gray_small is None:
                    should_save = True
                else:
                    s_diff = _diff_score(prev_small, color_small)
                    s_ssim = _ssim_score(prev_gray_small, gray_small)
                    s_flow = _optical_flow_score(prev_gray_small, gray_small)
                    score = max(s_diff, s_ssim, s_flow)
                    # OR条件: どれか1つでも個別閾値超えで変化と判定
                    or_changed = (
                        s_diff >= _or_all_thresholds["diff"]
                        or s_ssim >= _or_all_thresholds["ssim"]
                        or s_flow >= _or_all_thresholds["optical-flow"]
                    )
                    if or_changed and (timestamp - last_saved_time) >= min_interval:
                        should_save = True
                prev_small = color_small
                prev_gray_small = gray_small

        if should_save:
            # 出力解像度のリサイズ
            out_frame = frame
            if output_width > 0 and frame.shape[1] != output_width:
                orig_h, orig_w = frame.shape[:2]
                out_h = int(orig_h * output_width / orig_w)
                out_frame = cv2.resize(frame, (output_width, out_h), interpolation=cv2.INTER_AREA)

            filename = seconds_to_filename(timestamp) + ext
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), out_frame, save_params)

            results.append({
                "timestamp": timestamp,
                "filename": filename,
                "filepath": str(filepath),
            })
            last_saved_time = timestamp

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if verbose:
        total_bytes = sum(Path(r["filepath"]).stat().st_size for r in results)
        total_mb = total_bytes / (1024 * 1024)
        print(f"[screenshot] 完了: {len(results)} 枚 / 合計 {total_mb:.1f} MB → {output_dir}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("使い方: python screenshot.py <動画ファイル> <出力ディレクトリ> [diff|ssim|optical-flow]")
        sys.exit(1)
    method = sys.argv[3] if len(sys.argv) > 3 else "diff"
    frames = extract_changed_frames(sys.argv[1], sys.argv[2], change_method=method)
    for f in frames:
        print(f["filename"])
