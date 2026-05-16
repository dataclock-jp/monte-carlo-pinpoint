"""
YouTube アドオン
================
Video2AI コアエンジンの周辺機能として、YouTube 動画を処理するためのモジュールです。

このモジュールはコアエンジン（screenshot.py / transcribe.py）に依存しますが、
コアエンジン側はこのモジュールを一切参照しません。

依存パッケージ（requirements-youtube.txt）:
    - yt-dlp        : YouTube 動画・字幕のダウンロード
    - youtube-transcript-api : YouTube 字幕の直接取得（yt-dlp より高速）

使い方:
    from addons.youtube import fetch_youtube_transcript, download_youtube_video

    # 字幕を取得（取得できなければ None を返す）
    transcript = fetch_youtube_transcript("xiET3sGuMLM", languages=["ja", "en"])

    # 動画をダウンロードしてローカルパスを返す
    video_path = download_youtube_video("https://www.youtube.com/watch?v=xiET3sGuMLM", "/tmp/out")
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def extract_video_id(url_or_id: str) -> str:
    """YouTube URL または動画 ID から動画 ID を抽出して返す。"""
    # すでに ID のみの場合（11文字の英数字）
    if re.fullmatch(r"[A-Za-z0-9_\-]{11}", url_or_id):
        return url_or_id
    # URL から抽出
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_\-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    raise ValueError(f"YouTube 動画 ID を抽出できませんでした: {url_or_id}")


# ---------------------------------------------------------------------------
# 字幕取得（youtube-transcript-api）
# ---------------------------------------------------------------------------

def fetch_youtube_transcript(
    url_or_id: str,
    languages: list[str] | None = None,
) -> Optional[list[dict]]:
    """
    YouTube の字幕データをタイムスタンプ付きで取得する。

    Parameters
    ----------
    url_or_id : str
        YouTube URL または動画 ID。
    languages : list[str], optional
        優先言語コードのリスト（例: ["ja", "en"]）。
        None の場合は ["ja", "en"] を使用。

    Returns
    -------
    list[dict] | None
        成功時: [{"start": float, "duration": float, "text": str}, ...]
        失敗時: None（ブロック・字幕なし・ライブラリ未インストール）
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print(
            "[youtube] youtube-transcript-api が未インストールです。\n"
            "  pip install -r requirements-youtube.txt を実行してください。",
            file=sys.stderr,
        )
        return None

    if languages is None:
        languages = ["ja", "en"]

    video_id = extract_video_id(url_or_id)

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # 手動字幕を優先し、なければ自動生成字幕を使用
        try:
            transcript = transcript_list.find_manually_created_transcript(languages)
            print(f"[youtube] 手動字幕を取得: {transcript.language} ({transcript.language_code})")
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(languages)
                print(f"[youtube] 自動生成字幕を取得: {transcript.language} ({transcript.language_code})")
            except Exception:
                # 指定言語がない場合は最初に見つかった字幕を使用
                available = list(transcript_list)
                if not available:
                    print("[youtube] 字幕が見つかりませんでした。", file=sys.stderr)
                    return None
                transcript = available[0]
                print(f"[youtube] 字幕を取得（フォールバック）: {transcript.language} ({transcript.language_code})")

        fetched = transcript.fetch()
        # FetchedTranscript を dict のリストに変換
        segments = [
            {"start": seg.start, "duration": seg.duration, "text": seg.text}
            for seg in fetched
        ]
        print(f"[youtube] {len(segments)} セグメントを取得しました。")
        return segments

    except Exception as e:
        error_type = type(e).__name__
        if "RequestBlocked" in error_type or "IpBlocked" in error_type:
            print(
                "[youtube] IP がブロックされています（クラウド環境では発生しやすい）。\n"
                "  Whisper によるフォールバックに切り替えます。",
                file=sys.stderr,
            )
        elif "TranscriptsDisabled" in error_type:
            print("[youtube] この動画は字幕が無効化されています。Whisper にフォールバックします。", file=sys.stderr)
        elif "VideoUnavailable" in error_type:
            print("[youtube] 動画が見つかりません。URL または ID を確認してください。", file=sys.stderr)
        else:
            print(f"[youtube] 字幕取得エラー ({error_type}): {e}", file=sys.stderr)
        return None


def segments_to_transcript_lines(segments: list[dict]) -> list[str]:
    """
    字幕セグメントリストを transcribe.py と同じ形式のテキスト行に変換する。

    出力形式: "[HH:MM:SS] テキスト"
    """
    lines = []
    for seg in segments:
        start = seg["start"]
        h = int(start // 3600)
        m = int((start % 3600) // 60)
        s = int(start % 60)
        timestamp = f"{h:02d}:{m:02d}:{s:02d}"
        lines.append(f"[{timestamp}] {seg['text']}")
    return lines


def save_youtube_transcript(
    segments: list[dict],
    output_dir: str | Path,
    stem: str,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """
    字幕セグメントを txt / json / srt 形式で保存する。

    Parameters
    ----------
    segments : list[dict]
        fetch_youtube_transcript() の戻り値。
    output_dir : str | Path
        保存先ディレクトリ。
    stem : str
        出力ファイルのベース名（拡張子なし）。
    formats : list[str], optional
        保存するフォーマット（"txt", "json", "srt"）。デフォルトは ["txt", "json"]。

    Returns
    -------
    dict[str, Path]
        {"txt": Path, "json": Path, ...} 形式の保存済みファイルパス。
    """
    import json

    if formats is None:
        formats = ["txt", "json"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    if "txt" in formats:
        lines = segments_to_transcript_lines(segments)
        path = output_dir / f"{stem}_transcript.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[youtube] TXT 保存: {path}")
        saved["txt"] = path

    if "json" in formats:
        path = output_dir / f"{stem}_transcript.json"
        path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[youtube] JSON 保存: {path}")
        saved["json"] = path

    if "srt" in formats:
        path = output_dir / f"{stem}_transcript.srt"
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start = seg["start"]
            end = start + seg.get("duration", 2.0)

            def to_srt_time(t: float) -> str:
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                ms = int((t % 1) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

            srt_lines.append(str(i))
            srt_lines.append(f"{to_srt_time(start)} --> {to_srt_time(end)}")
            srt_lines.append(seg["text"])
            srt_lines.append("")
        path.write_text("\n".join(srt_lines), encoding="utf-8")
        print(f"[youtube] SRT 保存: {path}")
        saved["srt"] = path

    return saved


# ---------------------------------------------------------------------------
# 動画ダウンロード（yt-dlp）
# ---------------------------------------------------------------------------

def download_youtube_video(
    url: str,
    output_dir: str | Path,
    filename: str = "video",
) -> Optional[Path]:
    """
    YouTube 動画をダウンロードしてローカルパスを返す。

    Parameters
    ----------
    url : str
        YouTube URL。
    output_dir : str | Path
        保存先ディレクトリ。
    filename : str
        保存ファイル名（拡張子なし）。デフォルトは "video"。

    Returns
    -------
    Path | None
        ダウンロードされた動画ファイルのパス。失敗時は None。
    """
    try:
        import yt_dlp
    except ImportError:
        print(
            "[youtube] yt-dlp が未インストールです。\n"
            "  pip install -r requirements-youtube.txt を実行してください。",
            file=sys.stderr,
        )
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{filename}.%(ext)s")

    ydl_opts = {
        "outtmpl": output_template,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": False,
        "no_warnings": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ext = info.get("ext", "mp4")
            downloaded = output_dir / f"{filename}.{ext}"
            if downloaded.exists():
                print(f"[youtube] 動画ダウンロード完了: {downloaded}")
                return downloaded
            # ファイル名が変わっている場合を探す
            for f in output_dir.glob(f"{filename}.*"):
                if f.suffix in {".mp4", ".mkv", ".webm"}:
                    print(f"[youtube] 動画ダウンロード完了: {f}")
                    return f
            print("[youtube] ダウンロードファイルが見つかりませんでした。", file=sys.stderr)
            return None
    except Exception as e:
        print(f"[youtube] ダウンロードエラー: {e}", file=sys.stderr)
        return None
