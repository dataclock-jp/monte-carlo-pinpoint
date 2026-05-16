"""
transcribe.py
タイムスタンプ付き文字起こしモジュール

OpenAI Whisper を使用して動画の音声を文字起こしし、
各セグメントに [HH:MM:SS] 形式のタイムスタンプを付与したテキストを生成する。
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def seconds_to_timestamp(seconds: float) -> str:
    """秒数を [HH:MM:SS] 形式の文字列に変換する。"""
    total_s = int(seconds)
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def extract_audio(video_path: str, audio_path: str, verbose: bool = True) -> None:
    """
    FFmpeg を使用して動画から音声を抽出する。

    Parameters
    ----------
    video_path : str
        入力動画ファイルのパス
    audio_path : str
        出力音声ファイルのパス（.wav 推奨）
    verbose : bool
        FFmpeg の出力を表示するか
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                    # 映像を除外
        "-acodec", "pcm_s16le",   # 16bit PCM WAV
        "-ar", "16000",           # Whisper 推奨サンプリングレート
        "-ac", "1",               # モノラル
        str(audio_path),
    ]
    stderr = None if verbose else subprocess.DEVNULL
    result = subprocess.run(cmd, stderr=stderr)
    if result.returncode != 0:
        raise RuntimeError(f"音声抽出に失敗しました: {video_path}")


def transcribe_video(
    video_path: str,
    output_dir: str,
    model_name: str = "base",
    language: Optional[str] = None,
    output_formats: list[str] = None,
    verbose: bool = True,
) -> dict:
    """
    動画を文字起こしし、タイムスタンプ付きテキストを保存する。

    Parameters
    ----------
    video_path : str
        入力動画ファイルのパス
    output_dir : str
        出力ファイル保存先ディレクトリ
    model_name : str
        Whisper モデル名（tiny / base / small / medium / large）
        精度と速度のバランス: base が推奨デフォルト
    language : str, optional
        音声言語コード（例: "ja", "en"）。None で自動検出
    output_formats : list[str]
        出力フォーマット（"txt", "json", "srt" の組み合わせ）
    verbose : bool
        進捗メッセージを表示するか

    Returns
    -------
    dict
        {
            "segments": [...],   # Whisper セグメントリスト
            "text": str,         # タイムスタンプ付き全文
            "files": dict        # 保存されたファイルパスの辞書
        }
    """
    import whisper

    if output_formats is None:
        output_formats = ["txt", "json"]

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 音声抽出（一時ファイル）
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        if verbose:
            print(f"[transcribe] 音声を抽出中: {video_path.name}")
        extract_audio(str(video_path), audio_path, verbose=False)

        if verbose:
            print(f"[transcribe] Whisper モデル '{model_name}' を読み込み中...")
        model = whisper.load_model(model_name)

        if verbose:
            lang_str = language if language else "自動検出"
            print(f"[transcribe] 文字起こし中（言語: {lang_str}）...")

        transcribe_options = {"verbose": False}
        if language:
            transcribe_options["language"] = language

        result = model.transcribe(audio_path, **transcribe_options)

    finally:
        Path(audio_path).unlink(missing_ok=True)

    segments = result.get("segments", [])

    # タイムスタンプ付きテキストを生成
    lines = []
    for seg in segments:
        ts = seconds_to_timestamp(seg["start"])
        text = seg["text"].strip()
        if text:
            lines.append(f"{ts} {text}")

    timestamped_text = "\n".join(lines)

    # 保存されたファイルパスを追跡
    saved_files = {}
    stem = video_path.stem

    # TXT 形式（タイムスタンプ付き読みやすいテキスト）
    if "txt" in output_formats:
        txt_path = output_dir / f"{stem}_transcript.txt"
        txt_path.write_text(timestamped_text, encoding="utf-8")
        saved_files["txt"] = str(txt_path)
        if verbose:
            print(f"[transcribe] TXT 保存: {txt_path}")

    # JSON 形式（セグメント詳細データ）
    if "json" in output_formats:
        json_data = {
            "video": str(video_path),
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "timestamp": seconds_to_timestamp(seg["start"]),
                    "text": seg["text"].strip(),
                }
                for seg in segments
            ],
        }
        json_path = output_dir / f"{stem}_transcript.json"
        json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved_files["json"] = str(json_path)
        if verbose:
            print(f"[transcribe] JSON 保存: {json_path}")

    # SRT 形式（字幕ファイル）
    if "srt" in output_formats:
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start_srt = _seconds_to_srt_time(seg["start"])
            end_srt = _seconds_to_srt_time(seg["end"])
            srt_lines.append(str(i))
            srt_lines.append(f"{start_srt} --> {end_srt}")
            srt_lines.append(seg["text"].strip())
            srt_lines.append("")
        srt_path = output_dir / f"{stem}_transcript.srt"
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        saved_files["srt"] = str(srt_path)
        if verbose:
            print(f"[transcribe] SRT 保存: {srt_path}")

    if verbose:
        print(f"[transcribe] 完了: {len(segments)} セグメント / 検出言語: {result.get('language', 'unknown')}")

    return {
        "segments": segments,
        "text": timestamped_text,
        "files": saved_files,
        "language": result.get("language", "unknown"),
    }


def _seconds_to_srt_time(seconds: float) -> str:
    """秒数を SRT タイムコード形式（HH:MM:SS,mmm）に変換する。"""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("使い方: python transcribe.py <動画ファイル> <出力ディレクトリ> [モデル名] [言語コード]")
        sys.exit(1)
    model = sys.argv[3] if len(sys.argv) > 3 else "base"
    lang = sys.argv[4] if len(sys.argv) > 4 else None
    result = transcribe_video(sys.argv[1], sys.argv[2], model_name=model, language=lang)
    print(result["text"])
