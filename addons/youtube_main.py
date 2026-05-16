#!/usr/bin/env python3
"""
YouTube 対応 CLI ラッパー
=========================
Video2AI コアエンジンを YouTube URL に対応させるアドオン CLI です。

使い方:
    # YouTube URL を直接渡す（字幕取得を優先し、失敗時は Whisper にフォールバック）
    python addons/youtube_main.py https://www.youtube.com/watch?v=VIDEO_ID output/

    # 字幕取得のみ（スクリーンショットなし）
    python addons/youtube_main.py https://youtu.be/VIDEO_ID output/ --skip-screenshot

    # Whisper を強制使用（字幕取得をスキップ）
    python addons/youtube_main.py URL output/ --force-whisper

    # 動画をダウンロードせずに字幕のみ取得
    python addons/youtube_main.py URL output/ --transcript-only

コアエンジンとの関係:
    このスクリプトはコアエンジン（main.py）を内部で呼び出します。
    コアエンジン側はこのスクリプトを一切参照しません。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

# コアエンジンをインポート（このスクリプトは addons/ 内にあるため親ディレクトリを追加）
sys.path.insert(0, str(Path(__file__).parent.parent))

from addons.youtube import (
    extract_video_id,
    fetch_youtube_transcript,
    save_youtube_transcript,
    download_youtube_video,
)


def run(args: argparse.Namespace) -> None:
    url = args.url
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Video2AI YouTube アドオン ===")
    print(f"入力: {url}")
    print(f"出力: {output_dir}")
    print("-" * 40)

    start_time = time.time()

    # 動画 ID を取得してファイル名のステムに使用
    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)

    stem = video_id

    # ------------------------------------------------------------------
    # Phase 1: 字幕取得
    # ------------------------------------------------------------------
    transcript_segments = None

    if not args.skip_transcript:
        if not args.force_whisper:
            print("[Phase 1] YouTube 字幕を取得中...")
            transcript_segments = fetch_youtube_transcript(
                url, languages=args.languages or ["ja", "en"]
            )
            if transcript_segments:
                formats = args.output_formats or ["txt", "json"]
                save_youtube_transcript(transcript_segments, output_dir, stem, formats=formats)
                print(f"[Phase 1] 字幕取得完了: {len(transcript_segments)} セグメント")
            else:
                print("[Phase 1] YouTube 字幕取得失敗 → Whisper にフォールバックします")
        else:
            print("[Phase 1] --force-whisper 指定のため字幕取得をスキップ")

    # ------------------------------------------------------------------
    # Phase 2: 動画ダウンロード（スクリーンショットまたは Whisper が必要な場合）
    # ------------------------------------------------------------------
    video_path = None
    need_video = (not args.skip_screenshot) or (transcript_segments is None and not args.skip_transcript)

    if args.transcript_only:
        need_video = False

    if need_video:
        print("[Phase 2] YouTube 動画をダウンロード中...")
        dl_dir = output_dir / "_download"
        video_path = download_youtube_video(url, dl_dir, filename=stem)
        if video_path is None:
            print("エラー: 動画のダウンロードに失敗しました。", file=sys.stderr)
            sys.exit(1)
        print(f"[Phase 2] ダウンロード完了: {video_path}")

    # ------------------------------------------------------------------
    # Phase 3: コアエンジン（main.py）を呼び出す
    # ------------------------------------------------------------------
    if video_path is not None:
        import importlib.util
        core_main_path = Path(__file__).parent.parent / "main.py"
        spec = importlib.util.spec_from_file_location("core_main", core_main_path)
        core_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_main)

        # コアエンジンの引数を構築
        core_args = argparse.Namespace(
            video_path=str(video_path),
            output_dir=str(output_dir),
            threshold=args.threshold,
            min_interval=args.min_interval,
            interval=args.interval,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            output_width=args.output_width,
            skip_screenshot=args.skip_screenshot,
            # Whisper フォールバックが必要な場合のみ文字起こしを実行
            skip_transcribe=(transcript_segments is not None) or args.skip_transcript,
            whisper_model=args.whisper_model,
            language=args.language,
            output_formats=args.output_formats or ["txt", "json"],
        )

        print("[Phase 3] コアエンジンを実行中...")
        core_main.run(core_args)

    elapsed = time.time() - start_time
    print("-" * 40)
    print(f"=== 完了 ({elapsed:.1f}秒) ===")
    print(f"出力ディレクトリ: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python addons/youtube_main.py",
        description="Video2AI YouTube アドオン: YouTube URL を直接処理します",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 必須引数
    parser.add_argument("url", help="YouTube URL または動画 ID")
    parser.add_argument("output_dir", help="出力ディレクトリ")

    # 字幕オプション
    transcript_group = parser.add_argument_group("字幕オプション")
    transcript_group.add_argument(
        "--languages", nargs="+", default=["ja", "en"],
        metavar="LANG",
        help="字幕の優先言語コード（例: ja en）"
    )
    transcript_group.add_argument(
        "--force-whisper", action="store_true",
        help="YouTube 字幕取得をスキップし、常に Whisper を使用する"
    )
    transcript_group.add_argument(
        "--skip-transcript", action="store_true",
        help="字幕・文字起こしを完全にスキップする"
    )
    transcript_group.add_argument(
        "--transcript-only", action="store_true",
        help="字幕取得のみ行い、動画のダウンロードとスクリーンショットをスキップする"
    )

    # スクリーンショットオプション（コアエンジンに委譲）
    ss_group = parser.add_argument_group("スクリーンショットオプション（コアエンジンに委譲）")
    ss_group.add_argument("--skip-screenshot", action="store_true", help="スクリーンショット抽出をスキップ")
    ss_group.add_argument("--threshold", type=float, default=0.02, help="変化検知の閾値（0〜1）")
    ss_group.add_argument("--min-interval", type=float, default=0.5, help="最小撮影間隔（秒）")
    ss_group.add_argument("--interval", type=float, default=0.0, help="一定間隔モード（秒）。0 で変化検知モード")
    ss_group.add_argument("--image-format", choices=["jpg", "png"], default="jpg", help="出力画像フォーマット")
    ss_group.add_argument("--jpeg-quality", type=int, default=85, help="JPEG 品質（0〜100）")
    ss_group.add_argument("--output-width", type=int, default=0, help="出力画像の幅（px）。0 で元解像度")

    # Whisper オプション（コアエンジンに委譲）
    whisper_group = parser.add_argument_group("Whisper オプション（コアエンジンに委譲）")
    whisper_group.add_argument("--whisper-model", default="base", help="Whisper モデルサイズ")
    whisper_group.add_argument("--language", default=None, help="音声言語コード（例: ja）")
    whisper_group.add_argument(
        "--output-formats", nargs="+", default=["txt", "json"],
        choices=["txt", "json", "srt"],
        help="出力フォーマット"
    )

    # AIプリフライトオプション（アドオン）
    pf_group = parser.add_argument_group("AIプリフライト設定（addons/preflight.py が必要）")
    pf_group.add_argument(
        "--preflight", action="store_true",
        help="AIに動画を分析させ、最適なパラメータを自動設定する"
    )
    pf_group.add_argument(
        "--ai-backend", type=str, default="openai",
        choices=["openai", "claude", "grok", "gemini"],
        help="プリフライトに使用するAIバックエンド"
    )
    pf_group.add_argument(
        "--ai-model", type=str, default=None,
        help="プリフライトに使用するモデル名（省略時はバックエンドのデフォルト）"
    )
    pf_group.add_argument(
        "--ai-api-key", type=str, default=None,
        help="AIのAPIキー（省略時は環境変数から読む）"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
