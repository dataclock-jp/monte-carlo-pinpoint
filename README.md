# Video2AI: Translating Motion into Machine Understanding

**A Temporal Interface for AI Video Comprehension**

> **About this repository (English).**
> This repository hosts the reference implementation and reproduction
> artifacts for the arXiv preprint:
>
> > Tomio Hisanari. *Monte Carlo Pinpoint: Stochastic Coordinate
> > Localization via VLM Probing.* arXiv preprint, 2026.
>
> Tagged release for the preprint state: `v1.0-arxiv-v1`.
>
> Paper-specific entry points:
> - Core algorithm: [`pinpoint_core.py`](pinpoint_core.py)
> - Benchmark harness, datasets, and per-version result JSON files: [`benchmark/`](benchmark/)
> - Paper LaTeX sources: [`docs/paper/`](docs/paper/)
>
> The repository is also the upstream of the broader **Video2AI** project
> — a Python CLI tool for translating videos into AI-readable formats
> (timestamped screenshots + transcripts). See the
> [Research / Citation](#research--citation) section for the full citation,
> or read on for the Japanese product documentation.

---

動画をAIが最も理解しやすい形式（画像＋テキスト）へ論理的に翻訳するインターフェース

現在のAIモデル（LLM/VLM）は、動画を直接読み込むことに多くの制約を抱えています。
GIFでは色が256色に制限され、単なる文字起こしでは画面上の視覚的な変化（UIの動きやスライドの切り替わりなど）が欠落します。

**Video2AI** は、この問題を解決するために設計されたPythonコマンドラインツールです。
動画を「変化があった瞬間の高画質スクリーンショット（時間名ファイル）」と「タイムスタンプ付き文字起こし」に分解し、AIに「時間軸」という概念を与えます。

## 特徴

1. **効率的なトリガー（動きが生じたら撮る）**
   - OpenCVを用いてフレーム間の変化を計算し、**「変化が生じた瞬間」**だけを厳選して画像を保存します。
   - 変化検知手法として、高速な「BGR差分」、精度重視の「SSIM（構造的類似度）」、動体量で判断する「オプティカルフロー」の3種類をサポートしています。
   - 情報の重複を削ぎ落とし、AIのコンテキストウィンドウを最大限に活かすことができます。
2. **AIに「時間の流れを理解する脳」を与える**
   - 画像ファイル名を `00-01-23.050.png` のような時間情報にすることで、AIは文字起こしデータ内のタイムスタンプと画像を即座に紐付ける（シンクロさせる）ことができます。
3. **高品質な静止画と正確な文脈**
   - 動画の視覚情報は劣化のないPNG画像として、音声情報はOpenAI Whisperによる高精度なテキストとして抽出されます。
4. **自己完結型アーカイブ（.v2ai 形式）**
   - 画像、文字起こし、同期マップを1つのSQLiteベースのファイル（`.v2ai`）に統合。
   - 外部ファイルへの依存がなくなり、1ファイルで全ての情報を管理・共有することが可能になります。

## インストール

### 前提条件

- Python 3.9 以上
- FFmpeg（システムにインストールされていること）

Ubuntu / Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```

macOS (Homebrew):
```bash
brew install ffmpeg
```

Windows (winget):
```powershell
winget install ffmpeg
```
※または [FFmpeg公式サイト](https://ffmpeg.org/download.html) からダウンロードしてパスを通してください。

### pip でインストール（推奨）

```bash
pip install video2ai
```

オプション機能を含める場合：
```bash
pip install video2ai[ai]       # AIプリフライト（設定自動提案）
pip install video2ai[mcp]      # MCP サーバー
pip install video2ai[diarize]  # Speaker Diarization
pip install video2ai[agent]    # 自律エージェント
pip install video2ai[all]      # 全機能
```

インストール後は `video2ai` コマンドで直接実行できます：
```bash
video2ai input_video.mp4 output_directory
```

### ソースからインストール

```bash
git clone https://github.com/dataclock-jp/monte-carlo-pinpoint.git
cd monte-carlo-pinpoint
pip install -e .
```

### YouTubeアドオンのインストール（オプション）

YouTube URLを直接処理したい場合は、追加のパッケージをインストールします。

```bash
pip install video2ai[youtube]
# または: pip install -r requirements-youtube.txt
```

### MCPサーバーとしての利用（オプション）

Claude Desktop や Claude Code などの AI アシスタントから、Video2AI を自律的に呼び出せるようにする MCP (Model Context Protocol) サーバー機能です。

```bash
pip install -r requirements-mcp.txt
```

**Claude Desktop への設定例** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "video2ai": {
      "command": "python",
      "args": ["/絶対パス/video2ai/addons/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

AIに「この動画を分析して」と頼むだけで、自動的にスクリーンショット抽出と文字起こしが行われ、AIがその結果を読んで回答してくれます。

### Speaker Diarizationアドオンのインストール（オプション）

文字起こし結果に「誰が話したか」の話者ラベルを付与します。HuggingFaceアカウントと以下のモデルへの利用規約同意が必要です。

> **動作環境について:** Speaker Diarizationは `pyannote/speaker-diarization-community-1` モデルを使用します。このモデルは `torchcodec` に依存しており、**GPU（CUDA）環境またはCPU版torchcodecが正しくインストールされた環境**での動作を推奨します。クラウドのCPU専用環境（AWS Lambda等）では `libnppicc.so` の欠如によりエラーが発生する場合があります。その場合は `pyannote/speaker-diarization-3.1` モデルを `--model` オプションで指定してください。

1. [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

```bash
pip install -r requirements-diarize.txt
export HF_TOKEN="your_huggingface_token"
```

### AIプリフライトアドオンのインストール（オプション）

AIに動画を自動判断させてパラメータを決定する場合は、使用するバックエンドに応じてインストールします。

```bash
pip install -r requirements-ai.txt
```

| バックエンド | 必要なライブラリ | 環境変数 |
|---|---|---|
| `openai`（デフォルト） | `openai` | `OPENAI_API_KEY` |
| `claude` | `anthropic` | `ANTHROPIC_API_KEY` |
| `grok` | `openai` | `XAI_API_KEY` |
| `gemini` | `google-generativeai`, `pillow` | `GOOGLE_API_KEY` |

## 使い方

### ローカル動画の処理（コアエンジン）

```bash
python main.py input_video.mp4 output_directory
```

実行後、`output_directory` には以下が生成されます：
- `frames/` : 変化検知スクリーンショット（例: `00-01-23.050.png`）
- `input_video_transcript.txt` : タイムスタンプ付き文字起こし
- `input_video_transcript.json` : セグメント詳細データ

`--sync-map` を追加すると、さらに以下が生成されます：
- `input_video_sync_map.json` : Audio/Visual 同期マップ（各スクリーンショットと対応する発話を紐付けたJSON）

### AIプリフライト（アドオン）

`--preflight` オプションを付けると、処理前にAIが動画の代表フレームを分析し、最適なパラメータを自動設定します。

```bash
# ローカル動画にプリフライトを適用（OpenAI GPT-4o-miniで判断）
python main.py video.mp4 output/ --preflight

# Claudeで判断
python main.py video.mp4 output/ --preflight --ai-backend claude

# YouTube動画にもプリフライトを適用
python addons/youtube_main.py URL output/ --preflight --ai-backend gemini
```

AIが返す判断例（スポーツ動画の場合）:
```json
{
  "video_type": "sports",
  "reason": "サッカーの試合映像のため、常に動きがあり変化検知モードでは過剰に撮影される",
  "mode": "interval",
  "interval": 3.0,
  "jpeg_quality": 78,
  "output_width": 960
}
```

### 話者分離（Speaker Diarization）（アドオン）

Whisperで生成した文字起こしJSONに、話者ラベルを後付けで付与します。

```bash
# 基本的な使い方（HF_TOKENを環境変数に設定済みの場合）
python addons/diarize.py video.mp4 output/video_transcript.json

# 話者数を指定する場合
python addons/diarize.py video.mp4 output/video_transcript.json --num-speakers 2

# 出力ディレクトリを指定する場合
python addons/diarize.py video.mp4 output/video_transcript.json output/
```

出力ファイル：
- `video_diarized.txt` : 話者ラベル付きのテキスト（話者が変わるたびにヘッダーが入る）
- `video_diarized.json` : 話者ラベル付きのセグメントデータ

出力例（`video_diarized.txt`）：
```
=== SPEAKER_00 ===
[00:00:00] 皆さんこんにちは、ワンダー佐藤です。
[00:00:05] 今回はNotebookLMを使って...

=== SPEAKER_01 ===
[00:02:30] ありがとうございます。では次に...
```

| オプション | 説明 |
|---|---|
| `--num-speakers N` | 話者数を固定する |
| `--min-speakers N` | 最小話者数を指定 |
| `--max-speakers N` | 最大話者数を指定 |
| `--hf-token TOKEN` | HuggingFaceトークンを直接指定 |
| `--model MODEL` | pyannoteモデル名（デフォルト: `pyannote/speaker-diarization-community-1`） |

### YouTube動画の処理（アドオン）

YouTubeのURLを直接指定して処理できます。自動的に公式字幕を取得し、動画をダウンロードして画像抽出を行います。

```bash
python addons/youtube_main.py https://www.youtube.com/watch?v=VIDEO_ID output/
```

- 公式字幕が取得できる場合はWhisperによる文字起こしをスキップするため**非常に高速**です。
- 字幕が取得できない場合（または `--force-whisper` 指定時）は、自動的にWhisperにフォールバックします。
- 字幕だけ欲しい場合: `python addons/youtube_main.py URL output/ --transcript-only`

### オプション

```bash
python main.py <video_path> <output_dir> [オプション]
```

**実行モード**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--skip-screenshot` | False | スクリーンショット抽出をスキップ |
| `--skip-transcribe` | False | 文字起こしをスキップ |

**.v2ai 自己完結型アーカイブ設定**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--archive` | False | 画像・文字起こし・同期マップを1つの `.v2ai` ファイルに統合する |
| `--archive-only` | False | `.v2ai` アーカイブのみ生成し、個別の画像ファイルを保存しない |

**Speaker Diarization設定**

| オプション | 説明 |
|---|---|
| `--diarize` | 話者分離を実行する（`addons/diarize.py` を内部呼び出し） |
| `--num-speakers N` | 話者数を固定する |
| `--hf-token TOKEN` | HuggingFaceトークンを直接指定（デフォルト: `HF_TOKEN` 環境変数） |

**Audio/Visual 同期マップ設定**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--sync-map` | False | 同期マップ（sync_map.json）を生成する |
| `--sync-window` | 5.0 | 各フレームに紐付ける前後ウィンドウ幅（秒） |

**スクリーンショット設定**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--threshold` | 0.02 | 変化検知の閾値（0〜1）。小さいほど敏感 |
| `--min-interval` | 0.5 | 変化検知モード時の最小撮影間隔（秒） |
| `--interval` | 0.0 | 一定間隔モード（秒）。0 で変化検知モード |
| `--change-method` | diff | 変化検知手法（`diff`, `ssim`, `optical-flow`） |
| `--image-format` | jpg | 出力フォーマット：`jpg`（軽量）または `png`（高品質） |
| `--jpeg-quality` | 85 | JPEG 品質（0〜100）。高品質=95、バランス=85、軽量=70 |
| `--output-width` | 0 | 出力画像の幅（px）。0 で元解像度のまま |

**文字起こし設定**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--whisper-model` | base | Whisper モデルサイズ（tiny/base/small/medium/large） |
| `--language` | 自動検出 | 音声言語コード（例: `ja`, `en`） |
| `--output-formats` | txt json | 出力フォーマット（`txt` `json` `srt` の組み合わせ） |

## 使用例

**スライド・チュートリアル動画（変化検知モード・デフォルト）**
```bash
python main.py lecture.mp4 output/
```

**サーキット・スポーツ動画（一定間隔モード・軽量設定）**
```bash
python main.py race.mp4 output/ --interval 2.0 --image-format jpg --jpeg-quality 80 --output-width 640
```

**高品質アーカイブ（PNG・元解像度）**
```bash
python main.py video.mp4 output/ --image-format png --skip-transcribe
```

**Audio/Visual 同期マップを生成（AIへの一括渡しに最適）**
```bash
python main.py lecture.mp4 output/ --sync-map
```

**自己完結型アーカイブ（.v2ai）を生成（個別ファイル不要の場合）**
```bash
python main.py lecture.mp4 output/ --archive-only
```
生成された `lecture.v2ai` は1ファイルで全データ（画像・テキスト・同期マップ）を保持します。`v2ai_archive.py` を使って中身の確認や画像の取り出しが可能です。
```bash
python v2ai_archive.py info output/lecture.v2ai
python v2ai_archive.py export output/lecture.v2ai exported_frames/
```

生成される `sync_map.json` の構造例：
```json
{
  "frames": [
    {
      "timestamp": 912.9,
      "filename": "00-15-12.920.jpg",
      "transcript_before": [],
      "transcript_during": [
        { "timestamp": "[00:15:12]", "text": "ということで画面の方を見ていきましょう" }
      ],
      "transcript_window": [...]
    }
  ]
}
```

AIに渡す際は `transcript_during` を参照するだけで、**「この画面が映っていたとき何が話されていたか」が即座に分かります**。

## AIへのプロンプト例

生成されたファイル群をClaudeやGPT-4VなどのマルチモーダルAIにアップロードし、以下のように指示を出します：

> 添付した画像群（ファイル名が動画のタイムスタンプになっています）と、タイムスタンプ付き文字起こしテキストを読み込んでください。
> 
> これらは1つの動画から抽出されたデータです。テキストの内容と、対応する時間帯の画像を照らし合わせて、この動画で解説されている手順をステップバイステップでまとめてください。

## Research / Citation

This repository also contains the reference implementation and reproduction
artifacts for:

> Tomio Hisanari. *Monte Carlo Pinpoint: Stochastic Coordinate
> Localization via VLM Probing.* arXiv preprint, 2026.

The core algorithm lives in `pinpoint_core.py`. The benchmark harness and
datasets are under `benchmark/` (see `benchmark/run_benchmark.py` and
`benchmark/run_subscription.sh`). Per-version result JSON files
(`results.v6.*.json`, `results.v7_full_*.json`, `results.stage2_v9_full_*.json`,
`results.bundle_2_16_10case.json`, `results.arm_b_opus_*.json`, etc.) capture
the empirical evaluations reported in the paper. The paper LaTeX sources are
under `docs/paper/`.

This repository is released under the **MIT License** (see `LICENSE`).

## ライセンス

MIT License
