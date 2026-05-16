"""
addons/diarize.py - Speaker Diarization Add-on for Video2AI

Uses pyannote.audio to identify "who spoke when" and annotates
the existing Whisper transcript with speaker labels.

Requirements:
    pip install -r requirements-diarize.txt

Environment variables:
    HF_TOKEN: HuggingFace access token (required)

Usage (standalone CLI):
    python addons/diarize.py <audio_or_video> <transcript_json> [output_dir]

Usage (from main.py):
    python main.py video.mp4 output/ --diarize

Design:
    - Core engine (transcribe.py) is NOT modified.
    - This add-on reads the existing transcript JSON produced by transcribe.py,
      runs pyannote diarization on the audio, then merges the two by
      assigning each Whisper segment to the speaker who was dominant
      during that time window.
    - Output: {stem}_diarized.txt and {stem}_diarized.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_wav(video_path: str | Path, wav_path: str | Path) -> None:
    """Extract 16kHz mono WAV from video/audio using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", "16000", "-ac", "1", "-vn",
        str(wav_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")


def _seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _dominant_speaker(
    diarization_turns: list[dict],
    seg_start: float,
    seg_end: float,
) -> str | None:
    """
    Find the speaker who spoke the most during [seg_start, seg_end].
    Returns speaker label or None if no overlap found.
    """
    overlap: dict[str, float] = {}
    for turn in diarization_turns:
        t_start = turn["start"]
        t_end = turn["end"]
        speaker = turn["speaker"]
        # Calculate overlap
        ov = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
        if ov > 0:
            overlap[speaker] = overlap.get(speaker, 0.0) + ov
    if not overlap:
        return None
    return max(overlap, key=overlap.__getitem__)


# ---------------------------------------------------------------------------
# Core diarization function
# ---------------------------------------------------------------------------

def run_diarization(
    video_path: str | Path,
    transcript_json_path: str | Path,
    output_dir: str | Path | None = None,
    hf_token: str | None = None,
    model_name: str = "pyannote/speaker-diarization-community-1",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """
    Run speaker diarization and annotate transcript with speaker labels.

    Args:
        video_path: Path to the video or audio file.
        transcript_json_path: Path to the transcript JSON produced by transcribe.py.
        output_dir: Directory to save output files. Defaults to same dir as transcript.
        hf_token: HuggingFace access token. Falls back to HF_TOKEN env var.
        model_name: pyannote pipeline model name.
        num_speakers: Fix the number of speakers (optional).
        min_speakers: Minimum number of speakers (optional).
        max_speakers: Maximum number of speakers (optional).

    Returns:
        dict with keys: 'segments' (annotated), 'output_txt', 'output_json'
    """
    # --- Resolve token ---
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token is required. Set HF_TOKEN environment variable "
            "or pass hf_token parameter."
        )

    video_path = Path(video_path)
    transcript_json_path = Path(transcript_json_path)

    if output_dir is None:
        output_dir = transcript_json_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem

    # --- Load transcript ---
    print(f"[diarize] トランスクリプトを読み込み中: {transcript_json_path}")
    with open(transcript_json_path, encoding="utf-8") as f:
        transcript_data = json.load(f)
    segments = transcript_data.get("segments", [])
    if not segments:
        raise ValueError("transcript JSON has no segments.")

    # --- Extract WAV ---
    print(f"[diarize] 音声を抽出中: {video_path}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        _extract_wav(video_path, wav_path)

        # --- Load pyannote pipeline ---
        print(f"[diarize] pyannoteパイプラインを読み込み中: {model_name}")
        from pyannote.audio import Pipeline  # type: ignore
        import torch

        pipeline = Pipeline.from_pretrained(model_name, token=token)

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        print(f"[diarize] デバイス: {device}")

        # --- Run diarization ---
        print("[diarize] 話者分離を実行中（数分かかる場合があります）...")
        kwargs: dict = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        diarization = pipeline(wav_path, **kwargs)

        # Convert diarization result to list of turns
        turns: list[dict] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })
        print(f"[diarize] 検出された話者ターン数: {len(turns)}")

        # Count unique speakers
        unique_speakers = sorted(set(t["speaker"] for t in turns))
        print(f"[diarize] 検出された話者数: {len(unique_speakers)} ({', '.join(unique_speakers)})")

    finally:
        Path(wav_path).unlink(missing_ok=True)

    # --- Annotate transcript segments ---
    annotated_segments = []
    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start + 1.0)
        text = seg.get("text", "").strip()
        speaker = _dominant_speaker(turns, seg_start, seg_end)
        label = speaker if speaker else "UNKNOWN"
        annotated_segments.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": label,
            "text": text,
            "timestamp": f"[{_seconds_to_hms(seg_start)}]",
        })

    # --- Write TXT output ---
    output_txt = output_dir / f"{stem}_diarized.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in annotated_segments:
            speaker = seg["speaker"]
            ts = seg["timestamp"]
            text = seg["text"]
            # Add speaker header when speaker changes
            if speaker != current_speaker:
                f.write(f"\n=== {speaker} ===\n")
                current_speaker = speaker
            f.write(f"{ts} {text}\n")
    print(f"[diarize] TXT出力: {output_txt}")

    # --- Write JSON output ---
    output_json = output_dir / f"{stem}_diarized.json"
    result_data = {
        "video": str(video_path),
        "model": model_name,
        "speakers": unique_speakers,
        "num_speakers": len(unique_speakers),
        "segments": annotated_segments,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"[diarize] JSON出力: {output_json}")

    return {
        "segments": annotated_segments,
        "speakers": unique_speakers,
        "output_txt": str(output_txt),
        "output_json": str(output_json),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Video2AI Speaker Diarization Add-on",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (HF_TOKEN must be set in environment)
  python addons/diarize.py video.mp4 output/video_transcript.json

  # Specify number of speakers
  python addons/diarize.py video.mp4 output/video_transcript.json --num-speakers 2

  # Specify output directory
  python addons/diarize.py video.mp4 output/video_transcript.json output/
        """,
    )
    parser.add_argument("video", help="Path to video or audio file")
    parser.add_argument("transcript_json", help="Path to transcript JSON from transcribe.py")
    parser.add_argument("output_dir", nargs="?", help="Output directory (default: same as transcript)")
    parser.add_argument("--hf-token", help="HuggingFace access token (default: HF_TOKEN env var)")
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1",
                        help="pyannote pipeline model name")
    parser.add_argument("--num-speakers", type=int, help="Fix number of speakers")
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers")

    args = parser.parse_args()

    result = run_diarization(
        video_path=args.video,
        transcript_json_path=args.transcript_json,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        model_name=args.model,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    print(f"\n完了！")
    print(f"  話者数: {result['num_speakers']} ({', '.join(result['speakers'])})")
    print(f"  TXT: {result['output_txt']}")
    print(f"  JSON: {result['output_json']}")


if __name__ == "__main__":
    main()
