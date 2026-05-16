"""
video2ai - 動画をAIが理解しやすい形式（変化検知画像＋タイムスタンプ付きテキスト）に変換するツール
"""

import argparse
import sys
import time
from pathlib import Path

from screenshot import extract_changed_frames
from transcribe import transcribe_video
from sync_map import build_sync_map
from v2ai_archive import V2AIArchive


def main():
    parser = argparse.ArgumentParser(
        description="動画をAI向けに最適化：変化検知スクリーンショットとタイムスタンプ付き文字起こしを生成します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 必須引数
    parser.add_argument("video_path", type=str, help="入力動画ファイルのパス")
    parser.add_argument("output_dir", type=str, help="出力ファイルを保存するディレクトリ")

    # 実行モード制御
    group_mode = parser.add_argument_group("実行モード")
    group_mode.add_argument("--skip-screenshot", action="store_true", help="スクリーンショット抽出をスキップする")
    group_mode.add_argument("--skip-transcribe", action="store_true", help="文字起こしをスキップする")

    # スクリーンショット設定
    group_ss = parser.add_argument_group("スクリーンショット設定")
    group_ss.add_argument(
        "--threshold", type=float, default=0.02,
        help="変化検知の閾値（0〜1）。小さいほど敏感に変化を検知"
    )
    group_ss.add_argument(
        "--min-interval", type=float, default=0.5,
        help="変化検知モード時の最小撮影間隔（秒）"
    )
    group_ss.add_argument(
        "--interval", type=float, default=0.0,
        help="一定間隔モード（秒）。0 の場合は変化検知モード。例: 2.0 で 2 秒ごとに 1 枚"
    )
    group_ss.add_argument(
        "--image-format", type=str, default="jpg", choices=["jpg", "png"],
        help="出力画像フォーマット。jpg（軽量）または png（高品質・可逆）"
    )
    group_ss.add_argument(
        "--jpeg-quality", type=int, default=85,
        help="JPEG 保存時の品質（0〜100）。高品質=95、バランス=85、軽量=70"
    )
    group_ss.add_argument(
        "--output-width", type=int, default=0,
        help="出力画像の幅（ピクセル）。0 の場合は元解像度のまま。例: 640"
    )
    group_ss.add_argument(
        "--change-method", type=str, default="diff",
        choices=["diff", "ssim", "optical-flow", "or-all"],
        help=(
            "変化検知手法。"
            "diff: BGR差分（高速・デフォルト）、"
            "ssim: 構造的類似度（精度重視・スライド動画に最適）、"
            "optical-flow: オプティカルフロー（動体量で判断・スポーツ動画に最適）、"
            "or-all: 3方式OR（diff/ssim/optical-flowを並列実行、どれか1つでも閾値超えで検出）"
        )
    )

    group_ss.add_argument(
        "--preview", type=float, default=0.0,
        help="N秒分だけ解析するプレビューモード（閾値チューニング用）。例: --preview 60"
    )
    group_ss.add_argument(
        "--start", type=float, default=0.0,
        help="指定秒数から開始（冒頭が静止画面の場合に途中からサンプリング）。例: --start 120"
    )

    # 文字起こし設定
    group_tr = parser.add_argument_group("文字起こし設定")
    group_tr.add_argument(
        "--whisper-model", type=str, default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper のモデルサイズ"
    )
    group_tr.add_argument(
        "--language", type=str, default=None,
        help="音声言語コード（例: ja, en）。指定しない場合は自動検出"
    )
    group_tr.add_argument(
        "--output-formats", type=str, nargs="+", default=["txt", "json"],
        choices=["txt", "json", "srt"],
        help="出力するテキストフォーマット"
    )

    # アーカイブ設定
    group_ar = parser.add_argument_group(".v2ai 自己完結型アーカイブ設定")
    group_ar.add_argument(
        "--archive", action="store_true",
        help="画像・文字起こし・同期マップを1つの .v2ai ファイルに統合する"
    )
    group_ar.add_argument(
        "--archive-only", action="store_true",
        help=".v2ai アーカイブのみ生成し、個別の画像ファイルを保存しない（--archive を含む）"
    )

    # 同期マップ設定
    group_sm = parser.add_argument_group("Audio/Visual 同期マップ設定")
    group_sm.add_argument(
        "--sync-map", action="store_true",
        help="スクリーンショットと文字起こしを紐付けた同期マップ（sync_map.json）を生成する"
    )
    group_sm.add_argument(
        "--sync-window", type=float, default=5.0,
        help="同期マップで各フレームに紐付ける前後ウィンドウ幅（秒）"
    )

    # プリフライトオプション（アドオン）
    group_pf = parser.add_argument_group("AIプリフライト設定（addons/preflight.py が必要）")
    group_pf.add_argument(
        "--preflight", action="store_true",
        help="AIに動画を分析させ、最適なパラメータを自動設定する（addons/preflight.py が必要）"
    )
    group_pf.add_argument(
        "--ai-backend", type=str, default="openai",
        choices=["openai", "claude", "grok", "gemini"],
        help="プリフライトに使用するAIバックエンド"
    )
    group_pf.add_argument(
        "--ai-model", type=str, default=None,
        help="プリフライトに使用するモデル名（省略時はバックエンドのデフォルト）"
    )
    group_pf.add_argument(
        "--ai-api-key", type=str, default=None,
        help="AIのAPIキー（省略時は環境変数から読む）"
    )

    args = parser.parse_args()
    run(args)


def run(args):
    """処理本体（youtube_main.py などから直接呼び出し可能）。"""
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Video2AI 処理開始 ===")
    print(f"入力: {video_path}")
    print(f"出力: {output_dir}")
    print("-" * 30)

    start_time = time.time()

    # 0. AIプリフライト（オプション）
    if getattr(args, "preflight", False):
        print("\n[Phase 0] AIプリフライト: 動画を分析してパラメータを自動設定中...")
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent))
            from addons.preflight import analyze_video_and_suggest_params
            suggested = analyze_video_and_suggest_params(
                video_path=video_path,
                backend=getattr(args, "ai_backend", "openai"),
                model=getattr(args, "ai_model", None),
                api_key=getattr(args, "ai_api_key", None),
            )
            # AIの提案でargsを上書き（ユーザーが明示的に指定した値は上書きしない）
            if suggested.get("mode") == "interval" and args.interval == 0.0:
                args.interval = suggested["interval"]
            if args.threshold == 0.02:
                args.threshold = suggested.get("threshold", 0.02)
            if args.jpeg_quality == 85:
                args.jpeg_quality = suggested.get("jpeg_quality", 85)
            if args.output_width == 0:
                args.output_width = suggested.get("output_width", 0)
            if args.whisper_model == "base":
                args.whisper_model = suggested.get("whisper_model", "base")
            if args.language is None:
                lang = suggested.get("language", "auto")
                args.language = None if lang == "auto" else lang
            print(f"[Phase 0] 設定確定: mode={'interval' if args.interval > 0 else 'change'}, "
                  f"interval={args.interval}, jpeg_quality={args.jpeg_quality}, "
                  f"output_width={args.output_width}, whisper_model={args.whisper_model}")
        except ImportError:
            print("[Phase 0] addons/preflight.py が見つかりません。デフォルト設定で続行します。", file=sys.stderr)
        except Exception as e:
            print(f"[Phase 0] プリフライトエラー: {e}。デフォルト設定で続行します。", file=sys.stderr)

    # 1. スクリーンショット抽出
    if not args.skip_screenshot:
        print("\n[Phase 1] 変化検知スクリーンショット抽出")
        frames_dir = output_dir / "frames"
        try:
            frames = extract_changed_frames(
                video_path=str(video_path),
                output_dir=str(frames_dir),
                threshold=args.threshold,
                min_interval=args.min_interval,
                image_format=args.image_format,
                jpeg_quality=args.jpeg_quality,
                output_width=args.output_width,
                interval_mode=args.interval,
                change_method=getattr(args, "change_method", "diff"),
                max_seconds=getattr(args, "preview", 0.0),
                start_seconds=getattr(args, "start", 0.0),
            )
        except Exception as e:
            print(f"スクリーンショット抽出中にエラーが発生しました: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\n[Phase 1] スクリーンショット抽出はスキップされました")

    # 2. 文字起こし
    if not args.skip_transcribe:
        print("\n[Phase 2] タイムスタンプ付き文字起こし")
        try:
            result = transcribe_video(
                video_path=str(video_path),
                output_dir=str(output_dir),
                model_name=args.whisper_model,
                language=args.language,
                output_formats=args.output_formats,
            )
        except Exception as e:
            print(f"文字起こし中にエラーが発生しました: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\n[Phase 2] 文字起こしはスキップされました")

    # 3. 同期マップ生成（スクリーンショットと文字起こしが両方揃っている場合）
    if getattr(args, "sync_map", False):
        if not args.skip_screenshot and not args.skip_transcribe:
            print("\n[Phase 3] Audio/Visual 同期マップ生成")
            try:
                build_sync_map(
                    video_path=str(video_path),
                    frames=frames,
                    segments=result.get("segments", []),
                    output_dir=str(output_dir),
                    window_sec=getattr(args, "sync_window", 5.0),
                    output_dir_base=str(output_dir),
                )
            except Exception as e:
                print(f"同期マップ生成中にエラーが発生しました: {e}", file=sys.stderr)
        else:
            print("\n[Phase 3] 同期マップはスキップ（スクリーンショットまたは文字起こしがスキップされています）")

    # 4. .v2ai アーカイブ生成
    archive_flag = getattr(args, "archive", False) or getattr(args, "archive_only", False)
    if archive_flag:
        if not args.skip_screenshot and not args.skip_transcribe:
            print("\n[Phase 4] .v2ai 自己完結型アーカイブ生成")
            try:
                archive_path = output_dir / f"{video_path.stem}.v2ai"
                with V2AIArchive(str(archive_path)) as archive:
                    archive.create(
                        video_path=str(video_path),
                        frames=frames,
                        segments=result.get("segments", []),
                        window_sec=getattr(args, "sync_window", 5.0),
                    )
                # --archive-only の場合は個別画像ファイルを削除
                if getattr(args, "archive_only", False):
                    import shutil
                    frames_dir = output_dir / "frames"
                    if frames_dir.exists():
                        shutil.rmtree(frames_dir)
                        print(f"[Phase 4] 個別画像ファイルを削除しました（アーカイブに統合済み）")
            except Exception as e:
                print(f".v2ai アーカイブ生成中にエラーが発生しました: {e}", file=sys.stderr)
        else:
            print("\n[Phase 4] アーカイブはスキップ（スクリーンショットまたは文字起こしがスキップされています）")

    elapsed = time.time() - start_time
    print("-" * 30)
    print(f"=== Video2AI 処理完了 ({elapsed:.1f}秒) ===")
    print(f"出力ディレクトリを確認してください: {output_dir}")


if __name__ == "__main__":
    main()
