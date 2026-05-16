"""
ocr_windows.py
Windows OCR (WinRT) を使ったテキスト認識

Windows 10/11 に標準搭載されている OCR エンジンを使用。
追加のインストール不要で日本語・英語に対応。
"""
import json as _json
import os
import sys
import subprocess
import tempfile
from typing import Optional, List, Dict, Any
from PIL import Image


# PowerShell スクリプトのテンプレート
_PS_SCRIPT = r'''
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Storage.StorageFile,Windows.Foundation,ContentType=WindowsRuntime]

function Await($WinRtTask, $ResultType) {
    $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

$imagePath = $args[0]
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($imagePath)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
Write-Output $result.Text
'''

# bbox 取得用 PowerShell スクリプト（word 単位の BoundingRect を JSON で出力）
_PS_SCRIPT_BBOX = r'''
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Storage.StorageFile,Windows.Foundation,ContentType=WindowsRuntime]

function Await($WinRtTask, $ResultType) {
    $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

$imagePath = $args[0]
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($imagePath)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$words = @()
foreach ($line in $result.Lines) {
    foreach ($word in $line.Words) {
        $rect = $word.BoundingRect
        $words += [pscustomobject]@{
            text = $word.Text
            x = [double]$rect.X
            y = [double]$rect.Y
            w = [double]$rect.Width
            h = [double]$rect.Height
        }
    }
}
$words | ConvertTo-Json -Compress -Depth 3
'''

# スクリプトファイルのパス（初回に作成）
_script_path: Optional[str] = None
_script_bbox_path: Optional[str] = None


def _ensure_script() -> str:
    """PowerShell スクリプトファイルを作成して返す。"""
    global _script_path
    if _script_path and os.path.exists(_script_path):
        return _script_path
    _script_path = os.path.join(tempfile.gettempdir(), "video2ai_ocr.ps1")
    with open(_script_path, "w", encoding="utf-8") as f:
        f.write(_PS_SCRIPT)
    return _script_path


def _ensure_script_bbox() -> str:
    """bbox 付き OCR 用 PowerShell スクリプトファイルを作成して返す。"""
    global _script_bbox_path
    if _script_bbox_path and os.path.exists(_script_bbox_path):
        return _script_bbox_path
    _script_bbox_path = os.path.join(tempfile.gettempdir(), "video2ai_ocr_bbox.ps1")
    with open(_script_bbox_path, "w", encoding="utf-8") as f:
        f.write(_PS_SCRIPT_BBOX)
    return _script_bbox_path


def _preprocess(image: Image.Image) -> Image.Image:
    """
    OCR 精度を上げるための前処理パイプライン。

    1. 小画像なら 2x に拡大（小文字の認識精度向上）
    2. グレースケール化
    3. ダークテーマ検出 → 反転（暗い背景の白文字を白背景の黒文字に）
    4. CLAHE コントラスト強調

    二値化は Windows OCR との相性が悪く精度を下げるため行わない。
    """
    import cv2
    import numpy as np

    # 小画像は拡大（幅 1000px 未満なら 2x）
    if image.width < 1000:
        image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)

    arr = np.array(image)

    # グレースケール化
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    # ダークテーマ検出: 平均輝度が128未満なら暗い背景 → 反転して白背景にする
    if gray.mean() < 128:
        gray = cv2.bitwise_not(gray)

    # CLAHE（コントラスト制限付き適応ヒストグラム均等化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return Image.fromarray(enhanced)


def ocr_image(image: Image.Image, timeout: int = 15,
              preprocess: bool = True) -> str:
    """
    PIL Image から Windows OCR でテキストを認識する。

    Parameters
    ----------
    image : PIL.Image
        認識する画像
    timeout : int
        タイムアウト（秒）
    preprocess : bool
        前処理（グレースケール、コントラスト強調、二値化）を適用するか

    Returns
    -------
    str
        認識されたテキスト（認識失敗時は空文字列）
    """
    if sys.platform != "win32":
        return ""

    if preprocess:
        image = _preprocess(image)

    tmp_path = os.path.join(tempfile.gettempdir(), "video2ai_ocr_input.png")
    try:
        image.save(tmp_path, format="PNG")
        script = _ensure_script()
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", script, tmp_path],
            capture_output=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.decode("utf-8", errors="replace").strip()
        return ""
    except (subprocess.TimeoutExpired, Exception):
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def ocr_image_with_bboxes(image: Image.Image, timeout: int = 15,
                          preprocess: bool = True) -> List[Dict[str, Any]]:
    """
    PIL Image から Windows OCR で word 単位の bbox 付きテキストを認識する。

    Returns
    -------
    list of dict
        [{"text": str, "bbox": {"x": float, "y": float, "w": float, "h": float}}, ...]
        bbox 座標は **入力画像の座標系**（preprocess による 2x upscale は逆変換済）。
        認識失敗時は空リスト。
    """
    if sys.platform != "win32":
        return []

    # _preprocess と同じ条件で upscale 係数を追跡（座標逆変換用）
    scale = 1.0
    if preprocess:
        if image.width < 1000:
            scale = 2.0
        image = _preprocess(image)

    tmp_path = os.path.join(tempfile.gettempdir(), "video2ai_ocr_bbox_input.png")
    try:
        image.save(tmp_path, format="PNG")
        script = _ensure_script_bbox()
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", script, tmp_path],
            capture_output=True, timeout=timeout,
        )
        if result.returncode != 0:
            return []
        raw = result.stdout.decode("utf-8", errors="replace").strip()
        if not raw:
            return []
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError:
            return []
        # ConvertTo-Json は 1 要素だと配列でなく object を返す
        if isinstance(parsed, dict):
            parsed = [parsed]
        out: List[Dict[str, Any]] = []
        for w in parsed:
            if not isinstance(w, dict):
                continue
            out.append({
                "text": w.get("text", ""),
                "bbox": {
                    "x": float(w.get("x", 0)) / scale,
                    "y": float(w.get("y", 0)) / scale,
                    "w": float(w.get("w", 0)) / scale,
                    "h": float(w.get("h", 0)) / scale,
                },
            })
        return out
    except (subprocess.TimeoutExpired, Exception):
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def ocr_region(monitor: int = 0, left: int = 0, top: int = 0,
               width: int = 0, height: int = 0) -> str:
    """
    画面の指定領域を OCR する。

    Parameters
    ----------
    monitor : int
        モニター番号
    left, top, width, height : int
        キャプチャ領域（0,0,0,0 で全画面）

    Returns
    -------
    str
        認識されたテキスト
    """
    from agent.screen_capture import ScreenCapture
    region = (left, top, width, height) if width > 0 and height > 0 else None
    cap = ScreenCapture(monitor_index=monitor, region=region)
    pil_img = cap.capture_as_pil()
    return ocr_image(pil_img)
