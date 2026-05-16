"""
Video2AI MCP サーバー
=====================
ClaudeなどのAIアシスタントが動画分析ツールを自律的に呼び出せるようにする
Model Context Protocol (MCP) サーバーです。

起動方法:
    # stdio モード（Claude Desktop / Claude Code 等）
    python addons/mcp_server.py

Claude Desktop の設定例 (claude_desktop_config.json):
    {
      "mcpServers": {
        "video2ai": {
          "command": "python",
          "args": ["/path/to/video2ai/addons/mcp_server.py"],
          "env": {
            "OPENAI_API_KEY": "sk-..."
          }
        }
      }
    }

公開ツール一覧:
    - analyze_video        : 動画を全処理（スクリーンショット＋文字起こし）
    - extract_screenshots  : スクリーンショット抽出のみ
    - transcribe_video     : 文字起こしのみ
    - preflight_video      : AIによるパラメータ提案のみ
    - analyze_youtube      : YouTube URLを受け取って全処理
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# コアエンジンのパスを通す
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="video2ai",
    instructions=(
        "動画ファイルをAIが理解しやすい形式（変化検知スクリーンショット＋タイムスタンプ付き文字起こし）に変換するツール群です。"
        "動画のパスを渡すと、画像ファイルと文字起こしテキストを生成します。"
        "まず preflight_video で動画を分析してパラメータを確認し、その後 analyze_video で処理するのが推奨フローです。"
    ),
)

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _read_image_as_base64(path: Path) -> str:
    """画像ファイルをbase64エンコードして返す。"""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _summarize_frames(frames_dir: Path, max_images: int = 10) -> list[dict]:
    """
    framesディレクトリから代表的な画像を選んでbase64リストを返す。
    全枚数が多い場合は均等間隔でサンプリングする。
    """
    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frames:
        return []

    # 均等間隔でmax_images枚を選ぶ
    if len(frames) <= max_images:
        selected = frames
    else:
        step = len(frames) / max_images
        selected = [frames[int(i * step)] for i in range(max_images)]

    result = []
    for f in selected:
        try:
            result.append({
                "timestamp": f.stem,  # ファイル名がタイムスタンプ（例: 00-01-23.050）
                "filename": f.name,
                "base64": _read_image_as_base64(f),
                "mime_type": "image/jpeg" if f.suffix == ".jpg" else "image/png",
            })
        except Exception:
            pass
    return result


def _read_transcript(output_dir: Path) -> str:
    """output_dir内の文字起こしTXTファイルを読んで返す。"""
    txts = list(output_dir.glob("*_transcript.txt"))
    if txts:
        return txts[0].read_text(encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# Tool 1: analyze_video（全処理）
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "動画ファイルを処理して、変化検知スクリーンショットとタイムスタンプ付き文字起こしを生成します。"
        "生成された画像（最大10枚）と文字起こしテキストを返します。"
        "処理時間は動画の長さに比例します（1分あたり約10〜30秒）。"
    )
)
def analyze_video(
    video_path: str,
    output_dir: Optional[str] = None,
    mode: str = "change",
    interval: float = 0.0,
    threshold: float = 0.02,
    jpeg_quality: int = 85,
    output_width: int = 0,
    whisper_model: str = "base",
    language: Optional[str] = None,
    max_images_to_return: int = 10,
) -> dict:
    """
    動画ファイルを全処理します。

    Args:
        video_path: 処理する動画ファイルの絶対パス
        output_dir: 出力ディレクトリ（省略時は動画と同じディレクトリに自動生成）
        mode: "change"=変化検知モード, "interval"=一定間隔モード
        interval: intervalモード時の撮影間隔（秒）。changeモードでは0.0
        threshold: 変化検知の感度（0〜1、小さいほど敏感）
        jpeg_quality: JPEG品質（0〜100）
        output_width: 出力画像の幅（px）。0で元解像度のまま
        whisper_model: Whisperモデルサイズ（tiny/base/small/medium/large）
        language: 音声言語コード（例: ja, en）。Noneで自動検出
        max_images_to_return: 返却する画像の最大枚数（デフォルト: 10）

    Returns:
        {
            "success": bool,
            "output_dir": str,
            "frame_count": int,
            "frames": [{"timestamp": str, "filename": str, "base64": str, "mime_type": str}],
            "transcript": str,
            "error": str  # エラー時のみ
        }
    """
    from screenshot import extract_changed_frames
    from transcribe import transcribe_video as _transcribe

    video_path = Path(video_path)
    if not video_path.exists():
        return {"success": False, "error": f"動画ファイルが見つかりません: {video_path}"}

    # 出力ディレクトリの決定
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = video_path.parent / f"{video_path.stem}_video2ai"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # スクリーンショット抽出
        frames_dir = out_dir / "frames"
        extract_changed_frames(
            video_path=str(video_path),
            output_dir=str(frames_dir),
            threshold=threshold,
            min_interval=0.5,
            image_format="jpg",
            jpeg_quality=jpeg_quality,
            output_width=output_width,
            interval_mode=interval if mode == "interval" else 0.0,
        )

        # 文字起こし
        _transcribe(
            video_path=str(video_path),
            output_dir=str(out_dir),
            model_name=whisper_model,
            language=language,
            output_formats=["txt", "json"],
        )

        # 結果の収集
        frames_data = _summarize_frames(frames_dir, max_images=max_images_to_return)
        transcript = _read_transcript(out_dir)
        frame_count = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png")))

        return {
            "success": True,
            "output_dir": str(out_dir),
            "frame_count": frame_count,
            "frames": frames_data,
            "transcript": transcript,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "output_dir": str(out_dir)}


# ---------------------------------------------------------------------------
# Tool 2: extract_screenshots（スクリーンショット抽出のみ）
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "動画から変化検知スクリーンショットのみを抽出します。"
        "文字起こしは行いません。高速に画像だけ欲しい場合に使います。"
    )
)
def extract_screenshots(
    video_path: str,
    output_dir: Optional[str] = None,
    mode: str = "change",
    interval: float = 0.0,
    threshold: float = 0.02,
    jpeg_quality: int = 85,
    output_width: int = 0,
    max_images_to_return: int = 10,
) -> dict:
    """
    動画からスクリーンショットのみを抽出します。

    Args:
        video_path: 処理する動画ファイルの絶対パス
        output_dir: 出力ディレクトリ（省略時は自動生成）
        mode: "change"=変化検知, "interval"=一定間隔
        interval: intervalモード時の撮影間隔（秒）
        threshold: 変化検知の感度（0〜1）
        jpeg_quality: JPEG品質（0〜100）
        output_width: 出力幅（px）。0で元解像度
        max_images_to_return: 返却する画像の最大枚数

    Returns:
        {"success": bool, "output_dir": str, "frame_count": int, "frames": [...]}
    """
    from screenshot import extract_changed_frames

    video_path = Path(video_path)
    if not video_path.exists():
        return {"success": False, "error": f"動画ファイルが見つかりません: {video_path}"}

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = video_path.parent / f"{video_path.stem}_video2ai"
    frames_dir = out_dir / "frames"

    try:
        extract_changed_frames(
            video_path=str(video_path),
            output_dir=str(frames_dir),
            threshold=threshold,
            min_interval=0.5,
            image_format="jpg",
            jpeg_quality=jpeg_quality,
            output_width=output_width,
            interval_mode=interval if mode == "interval" else 0.0,
        )

        frames_data = _summarize_frames(frames_dir, max_images=max_images_to_return)
        frame_count = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png")))

        return {
            "success": True,
            "output_dir": str(out_dir),
            "frame_count": frame_count,
            "frames": frames_data,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool 3: transcribe_video（文字起こしのみ）
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "動画の音声をWhisperで文字起こしします。スクリーンショット抽出は行いません。"
        "タイムスタンプ付きのテキストを返します。"
    )
)
def transcribe_video(
    video_path: str,
    output_dir: Optional[str] = None,
    whisper_model: str = "base",
    language: Optional[str] = None,
) -> dict:
    """
    動画の音声を文字起こしします。

    Args:
        video_path: 処理する動画ファイルの絶対パス
        output_dir: 出力ディレクトリ（省略時は自動生成）
        whisper_model: Whisperモデルサイズ（tiny/base/small/medium/large）
        language: 音声言語コード（例: ja, en）。Noneで自動検出

    Returns:
        {"success": bool, "output_dir": str, "transcript": str}
    """
    from transcribe import transcribe_video as _transcribe

    video_path = Path(video_path)
    if not video_path.exists():
        return {"success": False, "error": f"動画ファイルが見つかりません: {video_path}"}

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = video_path.parent / f"{video_path.stem}_video2ai"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        _transcribe(
            video_path=str(video_path),
            output_dir=str(out_dir),
            model_name=whisper_model,
            language=language,
            output_formats=["txt", "json"],
        )
        transcript = _read_transcript(out_dir)
        return {"success": True, "output_dir": str(out_dir), "transcript": transcript}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool 4: preflight_video（AIによるパラメータ提案）
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "動画の代表フレームをAIに渡して、最適な処理パラメータを提案させます。"
        "analyze_video を呼ぶ前にこのツールで設定を確認するのが推奨フローです。"
        "addons/preflight.py と requirements-ai.txt のインストールが必要です。"
    )
)
def preflight_video(
    video_path: str,
    backend: str = "openai",
    model: Optional[str] = None,
) -> dict:
    """
    動画の種類をAIに判断させ、最適な処理パラメータを返します。

    Args:
        video_path: 分析する動画ファイルの絶対パス
        backend: AIバックエンド（openai / claude / grok / gemini）
        model: 使用するモデル名（省略時はバックエンドのデフォルト）

    Returns:
        {
            "success": bool,
            "video_type": str,
            "reason": str,
            "suggested_params": {
                "mode": str, "interval": float, "threshold": float,
                "jpeg_quality": int, "output_width": int,
                "whisper_model": str, "language": str
            }
        }
    """
    try:
        from addons.preflight import analyze_video_and_suggest_params
    except ImportError:
        return {
            "success": False,
            "error": "addons/preflight.py が見つかりません。requirements-ai.txt をインストールしてください。"
        }

    video_path = Path(video_path)
    if not video_path.exists():
        return {"success": False, "error": f"動画ファイルが見つかりません: {video_path}"}

    try:
        result = analyze_video_and_suggest_params(
            video_path=video_path,
            backend=backend,
            model=model,
        )
        return {
            "success": True,
            "video_type": result.get("video_type", "unknown"),
            "reason": result.get("reason", ""),
            "suggested_params": {
                "mode": result.get("mode", "change"),
                "interval": result.get("interval", 0.0),
                "threshold": result.get("threshold", 0.02),
                "jpeg_quality": result.get("jpeg_quality", 85),
                "output_width": result.get("output_width", 0),
                "whisper_model": result.get("whisper_model", "base"),
                "language": result.get("language", "auto"),
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool 5: analyze_youtube（YouTube URL対応）
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "YouTube URLを受け取り、動画をダウンロードして全処理します。"
        "公式字幕が取得できる場合はWhisperをスキップするため高速です。"
        "addons/youtube.py と requirements-youtube.txt のインストールが必要です。"
    )
)
def analyze_youtube(
    youtube_url: str,
    output_dir: Optional[str] = None,
    mode: str = "change",
    interval: float = 0.0,
    jpeg_quality: int = 85,
    output_width: int = 0,
    whisper_model: str = "base",
    language: Optional[str] = None,
    force_whisper: bool = False,
    max_images_to_return: int = 10,
) -> dict:
    """
    YouTube URLから動画を取得して全処理します。

    Args:
        youtube_url: YouTube動画のURL
        output_dir: 出力ディレクトリ（省略時は一時ディレクトリを使用）
        mode: "change"=変化検知, "interval"=一定間隔
        interval: intervalモード時の撮影間隔（秒）
        jpeg_quality: JPEG品質（0〜100）
        output_width: 出力幅（px）。0で元解像度
        whisper_model: Whisperモデルサイズ
        language: 音声言語コード
        force_whisper: Trueの場合、字幕があってもWhisperを使用する
        max_images_to_return: 返却する画像の最大枚数

    Returns:
        analyze_video と同じ形式
    """
    try:
        from addons.youtube import YouTubeProcessor
    except ImportError:
        return {
            "success": False,
            "error": "addons/youtube.py が見つかりません。requirements-youtube.txt をインストールしてください。"
        }

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = None
    else:
        tmp_dir = tempfile.mkdtemp(prefix="video2ai_yt_")
        out_dir = Path(tmp_dir)

    try:
        processor = YouTubeProcessor(youtube_url)

        # 動画ダウンロード
        video_path = processor.download_video(str(out_dir))
        if not video_path:
            return {"success": False, "error": "動画のダウンロードに失敗しました。"}

        # 字幕取得（force_whisperでない場合）
        transcript = ""
        if not force_whisper:
            transcript_data = processor.get_transcript(
                language=language or "ja",
                output_dir=str(out_dir),
            )
            if transcript_data:
                transcript = "\n".join(
                    f"[{seg['start']:.1f}s] {seg['text']}" for seg in transcript_data
                )

        # スクリーンショット抽出
        from screenshot import extract_changed_frames
        frames_dir = out_dir / "frames"
        extract_changed_frames(
            video_path=str(video_path),
            output_dir=str(frames_dir),
            threshold=0.02,
            min_interval=0.5,
            image_format="jpg",
            jpeg_quality=jpeg_quality,
            output_width=output_width,
            interval_mode=interval if mode == "interval" else 0.0,
        )

        # 字幕がなければWhisperで文字起こし
        if not transcript:
            from transcribe import transcribe_video as _transcribe
            _transcribe(
                video_path=str(video_path),
                output_dir=str(out_dir),
                model_name=whisper_model,
                language=language,
                output_formats=["txt", "json"],
            )
            transcript = _read_transcript(out_dir)

        frames_data = _summarize_frames(frames_dir, max_images=max_images_to_return)
        frame_count = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png")))

        return {
            "success": True,
            "output_dir": str(out_dir),
            "frame_count": frame_count,
            "frames": frames_data,
            "transcript": transcript,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "output_dir": str(out_dir)}


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # デフォルトは stdio モード（Claude Desktop / Claude Code 対応）
    transport = os.environ.get("VIDEO2AI_TRANSPORT", "stdio")
    mcp.run(transport=transport)
