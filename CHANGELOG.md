# Changelog

All notable changes to this project will be documented in this file.

## [1.7.0] - 2026-03-30

### Added
- `pyproject.toml` for PyPI publishing (`pip install video2ai`)
- CLI entry points: `video2ai` and `video2ai-youtube` commands
- `LICENSE` file (MIT)
- Optional dependency groups: `[ai]`, `[mcp]`, `[diarize]`, `[agent]`, `[youtube]`, `[all]`

### Fixed
- README.md clone URL (`yourusername` -> `tbtomo`)

### Removed
- Test artifacts from repository (output_manga/, test_interval/, test_jpg_640/)

## [1.6.0] - 2026-03-29

### Added
- **PyAutoGUI Autonomous Agent** (`agent/` module + `agent_loop.py`)
  - 6 single-responsibility modules: screen_capture, change_detector, llm_vision, action_executor, session_logger
  - Closed-loop pipeline: Screen Capture -> Change Detection -> LLM Vision -> Action Execution
  - Reuses core `screenshot.py` algorithms (diff/ssim/optical-flow) for real-time change detection
  - Supports dry-run mode, Ctrl+C graceful shutdown, JSONL session logging
  - `on_event` callback for Web UI integration
- `test_agent.py` unit and integration tests

## [1.5.0] - 2026-03-29

### Added
- **Speaker Diarization** add-on (`addons/diarize.py`)
  - Uses pyannote.audio to annotate Whisper transcripts with speaker labels
  - Outputs `_diarized.txt` and `_diarized.json`
  - Supports GPU acceleration, configurable speaker count
  - Standalone CLI: `python addons/diarize.py video.mp4 transcript.json`

## [1.4.0] - 2026-03-29

### Added
- **SSIM and Optical Flow** change detection methods in `screenshot.py`
  - `--change-method ssim` for structural similarity (precise, ideal for slides)
  - `--change-method optical-flow` for motion-based detection (ideal for sports)
  - Original BGR diff remains default (`--change-method diff`)

## [1.3.0] - 2026-03-29

### Added
- **SQLite self-contained archive** (`v2ai_archive.py`)
  - `.v2ai` format: images as BLOBs + transcripts + sync map in one file
  - `--archive` and `--archive-only` flags in `main.py`
  - CLI for info, export, and AI prompt generation

## [1.2.0] - 2026-03-29

### Added
- **Audio/Visual sync map** (`sync_map.py`)
  - Links each screenshot to concurrent transcript segments
  - Outputs `_sync_map.json` with `transcript_before`, `transcript_during`, `transcript_window`
  - `sync_map_to_ai_prompt()` utility for LLM-ready text
  - `--sync-map` and `--sync-window` flags in `main.py`

## [1.1.0] - 2026-03-29

### Added
- **MCP server** (`addons/mcp_server.py`) for Claude Desktop / Claude Code integration
  - 5 tools: preflight_video, analyze_video, analyze_youtube, extract_screenshots, transcribe_video
  - Returns images as base64 in tool responses
- **AI preflight** (`addons/preflight.py`) for automatic parameter suggestion
  - Supports OpenAI, Claude, Grok, Gemini backends
  - `--preflight` flag in `main.py`

## [1.0.0] - 2026-03-29

### Added
- **Core engine**: `main.py`, `screenshot.py`, `transcribe.py`
  - BGR change detection with configurable threshold
  - OpenAI Whisper transcription with TXT/JSON/SRT output
  - Timestamp-named screenshots (e.g., `00-01-23.050.jpg`)
- **YouTube add-on** (`addons/youtube.py`, `addons/youtube_main.py`)
  - Official subtitle fetching via youtube-transcript-api
  - Fallback to yt-dlp + Whisper
- JPEG quality, output width, interval mode options
