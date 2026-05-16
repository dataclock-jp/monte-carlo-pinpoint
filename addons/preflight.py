"""
AIプリフライトアドオン
======================
動画の代表フレームをマルチモーダルAIに渡し、最適なVideo2AIの処理パラメータを
自動判断させるモジュールです。

このモジュールはコアエンジンに依存しますが、コアエンジン側はこのモジュールを
一切参照しません。

対応AIバックエンド:
    - openai    : GPT-4o / GPT-4o-mini (OpenAI API)
    - claude    : Claude 3.5 Sonnet / Haiku (Anthropic API)
    - grok      : Grok-2-vision (xAI API, OpenAI互換)
    - gemini    : Gemini 1.5 Pro / Flash (Google Generative AI API)

依存パッケージ（requirements-ai.txt）:
    - openai>=1.0.0          (openai / grok バックエンド)
    - anthropic>=0.25.0      (claude バックエンド)
    - google-generativeai    (gemini バックエンド)

使い方:
    from addons.preflight import analyze_video_and_suggest_params

    params = analyze_video_and_suggest_params(
        video_path="input.mp4",
        backend="claude",          # or "openai", "grok", "gemini"
        model=None,                # None でバックエンドのデフォルトモデルを使用
    )
    # params は argparse.Namespace 互換の dict
    # {"interval": 2.0, "jpeg_quality": 80, "output_width": 640, ...}
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# デフォルトモデル定義
# ---------------------------------------------------------------------------

DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "claude": "claude-3-5-haiku-20241022",
    "grok":   "grok-2-vision-1212",
    "gemini": "gemini-2.5-flash",
}

# ---------------------------------------------------------------------------
# プリフライト用プロンプト（全バックエンド共通）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
あなたは動画処理ツール「Video2AI」のパラメータ設定アドバイザーです。
ユーザーから動画の基本情報と代表フレーム画像が渡されます。
動画の内容・種類を分析し、最適な処理パラメータをJSON形式で返してください。
"""

USER_PROMPT_TEMPLATE = """\
以下の動画を処理するための最適なパラメータを判断してください。

## 動画情報
- ファイル名: {filename}
- 長さ: {duration_str}
- 解像度: {width}x{height}
- ファイルサイズ: {size_mb:.1f} MB

## 添付画像
動画の冒頭・中盤・末尾から抽出した代表フレーム（{n_frames}枚）です。

## 出力形式
以下のJSONのみを返してください（説明文は不要）:

```json
{{
  "video_type": "tutorial|lecture|sports|game|vlog|animation|other",
  "reason": "判断理由を1〜2文で",
  "mode": "change|interval",
  "interval": 0.0,
  "threshold": 0.02,
  "min_interval": 0.5,
  "jpeg_quality": 85,
  "output_width": 0,
  "whisper_model": "base",
  "language": "ja"
}}
```

## パラメータ選択の基準

| 動画の種類 | mode | interval | jpeg_quality | output_width |
|---|---|---|---|---|
| スライド・チュートリアル・講義 | change | 0.0 | 85 | 0（元解像度） |
| スポーツ・レース・アクション | interval | 2.0〜5.0 | 75〜80 | 640〜960 |
| ゲームプレイ | interval | 1.0〜3.0 | 80 | 960 |
| Vlog・インタビュー | change | 0.0 | 85 | 0 |
| アニメ・映画 | interval | 3.0〜10.0 | 85 | 0 |

- `mode`: "change"=変化検知モード, "interval"=一定間隔モード
- `interval`: intervalモード時の撮影間隔（秒）。changeモードでは0.0
- `threshold`: 変化検知の感度（0〜1、小さいほど敏感）
- `output_width`: 0で元解像度のまま
- `whisper_model`: tiny/base/small/medium/large（動画が長いほど小さいモデルを推奨）
- `language`: 音声の言語コード（不明なら"auto"）
"""

# ---------------------------------------------------------------------------
# 代表フレーム抽出
# ---------------------------------------------------------------------------

def extract_representative_frames(
    video_path: str | Path,
    n_frames: int = 5,
) -> list[Path]:
    """
    動画から代表フレームを抽出して一時ファイルに保存し、パスのリストを返す。
    冒頭・均等間隔・末尾から n_frames 枚を抽出する。
    """
    try:
        import cv2
    except ImportError:
        print("[preflight] OpenCV が未インストールです。", file=sys.stderr)
        return []

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[preflight] 動画を開けませんでした: {video_path}", file=sys.stderr)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps

    # 抽出するタイムスタンプ（秒）: 冒頭5秒・均等間隔・末尾5秒前
    if n_frames <= 1:
        timestamps = [duration * 0.5]
    else:
        # 冒頭は5秒、末尾は5秒前、中間は均等分割
        start = min(5.0, duration * 0.05)
        end = max(duration - 5.0, duration * 0.95)
        if n_frames == 2:
            timestamps = [start, end]
        else:
            mid_count = n_frames - 2
            step = (end - start) / (mid_count + 1)
            mid_times = [start + step * (i + 1) for i in range(mid_count)]
            timestamps = [start] + mid_times + [end]

    saved_paths = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="video2ai_preflight_"))

    for i, ts in enumerate(timestamps):
        frame_no = int(ts * fps)
        frame_no = max(0, min(frame_no, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            continue
        # 幅を最大 960px に縮小（APIへの転送量削減）
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            frame = cv2.resize(frame, (960, int(h * scale)))
        out_path = tmp_dir / f"preflight_{i:02d}_{ts:.1f}s.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        saved_paths.append(out_path)

    cap.release()
    print(f"[preflight] 代表フレーム {len(saved_paths)} 枚を抽出しました。")
    return saved_paths


def image_to_base64(path: Path) -> str:
    """画像ファイルを base64 文字列に変換する。"""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# AIバックエンド別の呼び出し
# ---------------------------------------------------------------------------

def _call_openai(
    frames: list[Path],
    user_prompt: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> str:
    """OpenAI API（GPT-4o / Grok）を呼び出す。"""
    from openai import OpenAI

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)

    content: list[dict] = [{"type": "text", "text": user_prompt}]
    for frame in frames:
        b64 = image_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def _call_claude(
    frames: list[Path],
    user_prompt: str,
    model: str,
    api_key: str | None,
) -> str:
    """Anthropic Claude API を呼び出す。"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    content: list[dict] = []
    for frame in frames:
        b64 = image_to_base64(frame)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({"type": "text", "text": user_prompt})

    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
    )
    return response.content[0].text if response.content else ""


def _call_gemini(
    frames: list[Path],
    user_prompt: str,
    model: str,
    api_key: str | None,
) -> str:
    """Google Gemini API を呼び出す。"""
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
    )

    parts: list[Any] = [user_prompt]
    for frame in frames:
        parts.append(Image.open(frame))

    response = gemini_model.generate_content(parts)
    return response.text or ""


# ---------------------------------------------------------------------------
# JSON パース
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict:
    """AIの応答からJSONブロックを抽出してパースする。"""
    # ```json ... ``` ブロックを探す
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        # ブロックなしで直接JSONが返ってきた場合
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[preflight] JSONパースエラー: {e}\n応答内容:\n{raw}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# デフォルトパラメータ
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "video_type": "unknown",
    "reason": "AI判断なし（デフォルト設定を使用）",
    "mode": "change",
    "interval": 0.0,
    "threshold": 0.02,
    "min_interval": 0.5,
    "jpeg_quality": 85,
    "output_width": 0,
    "whisper_model": "base",
    "language": "ja",
}


# ---------------------------------------------------------------------------
# メインAPI
# ---------------------------------------------------------------------------

def analyze_video_and_suggest_params(
    video_path: str | Path,
    backend: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    n_frames: int = 5,
) -> dict:
    """
    動画の代表フレームをAIに渡し、最適な処理パラメータを返す。

    Parameters
    ----------
    video_path : str | Path
        処理対象の動画ファイルパス。
    backend : str
        使用するAIバックエンド: "openai" / "claude" / "grok" / "gemini"
    model : str | None
        使用するモデル名。None でバックエンドのデフォルトを使用。
    api_key : str | None
        APIキー。None の場合は環境変数から読む。
    n_frames : int
        抽出する代表フレーム数（デフォルト: 5）。

    Returns
    -------
    dict
        処理パラメータの辞書。AIが失敗した場合はデフォルト値を返す。
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[preflight] 動画ファイルが見つかりません: {video_path}", file=sys.stderr)
        return DEFAULT_PARAMS.copy()

    # 動画情報の取得
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = total_frames / fps
    except Exception as e:
        print(f"[preflight] 動画情報の取得に失敗: {e}", file=sys.stderr)
        return DEFAULT_PARAMS.copy()

    size_mb = video_path.stat().st_size / (1024 * 1024)
    h = int(duration // 3600)
    m = int((duration % 3600) // 60)
    s = int(duration % 60)
    duration_str = f"{h:02d}:{m:02d}:{s:02d}"

    print(f"[preflight] 動画情報: {duration_str} / {width}x{height} / {size_mb:.1f}MB")
    print(f"[preflight] AIバックエンド: {backend} / モデル: {model or DEFAULT_MODELS.get(backend, '?')}")

    # 代表フレームの抽出
    frames = extract_representative_frames(video_path, n_frames=n_frames)
    if not frames:
        print("[preflight] フレーム抽出失敗。デフォルト設定を使用します。", file=sys.stderr)
        return DEFAULT_PARAMS.copy()

    # ユーザープロンプトの構築
    user_prompt = USER_PROMPT_TEMPLATE.format(
        filename=video_path.name,
        duration_str=duration_str,
        width=width,
        height=height,
        size_mb=size_mb,
        n_frames=len(frames),
    )

    # 使用モデルの決定
    resolved_model = model or DEFAULT_MODELS.get(backend, "")

    # AIへの問い合わせ
    raw_response = ""
    try:
        if backend == "openai":
            key = api_key or os.environ.get("OPENAI_API_KEY")
            raw_response = _call_openai(frames, user_prompt, resolved_model, key, None)

        elif backend == "claude":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            raw_response = _call_claude(frames, user_prompt, resolved_model, key)

        elif backend == "grok":
            key = api_key or os.environ.get("XAI_API_KEY")
            raw_response = _call_openai(
                frames, user_prompt, resolved_model, key,
                base_url="https://api.x.ai/v1",
            )

        elif backend == "gemini":
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            raw_response = _call_gemini(frames, user_prompt, resolved_model, key)

        else:
            print(f"[preflight] 未知のバックエンド: {backend}", file=sys.stderr)
            return DEFAULT_PARAMS.copy()

    except ImportError as e:
        print(f"[preflight] ライブラリ未インストール: {e}\n  pip install -r requirements-ai.txt を実行してください。", file=sys.stderr)
        return DEFAULT_PARAMS.copy()
    except Exception as e:
        print(f"[preflight] AI呼び出しエラー ({type(e).__name__}): {e}", file=sys.stderr)
        return DEFAULT_PARAMS.copy()

    # JSONパース
    params = _parse_json_response(raw_response)
    if not params:
        print("[preflight] AIの応答をパースできませんでした。デフォルト設定を使用します。", file=sys.stderr)
        return DEFAULT_PARAMS.copy()

    # デフォルト値で補完
    result = DEFAULT_PARAMS.copy()
    result.update(params)

    print(f"[preflight] 判断結果: {result.get('video_type', '?')} - {result.get('reason', '')}")
    print(f"[preflight] 設定: mode={result['mode']}, interval={result['interval']}, "
          f"jpeg_quality={result['jpeg_quality']}, output_width={result['output_width']}, "
          f"whisper_model={result['whisper_model']}")

    # 一時ファイルの削除
    for f in frames:
        try:
            f.unlink()
            f.parent.rmdir()
        except Exception:
            pass

    return result
