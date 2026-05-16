"""
sync_map.py
Audio/Visual 同期マップ生成モジュール

スクリーンショット（時間名ファイル）と文字起こしセグメントを
タイムスタンプで照合し、1つの JSON に統合します。

出力形式（sync_map.json）:
{
  "video": "input.mp4",
  "generated_at": "2026-03-29T12:00:00",
  "total_duration": 1769.0,
  "frames": [
    {
      "timestamp": 0.0,
      "filename": "00-00-00.000.jpg",
      "filepath": "frames/00-00-00.000.jpg",
      "transcript_before": [],       # この画像が表示される直前の発話
      "transcript_during": [         # この画像が表示されている間の発話
        {
          "start": 0.0,
          "end": 3.5,
          "timestamp": "[00:00:00]",
          "text": "こんにちは"
        }
      ],
      "transcript_window": [...]     # 前後 N 秒のウィンドウ内の全発話
    },
    ...
  ]
}

設計思想:
  - コアエンジンの出力（frames リスト・segments リスト）を受け取るだけで動作する
  - 外部ライブラリへの依存は一切なし（標準ライブラリのみ）
  - AIに渡す際は frames[i]["transcript_during"] を参照するだけで
    「この画面が映っていたとき何が話されていたか」が即座に分かる
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def build_sync_map(
    video_path: str,
    frames: list[dict],
    segments: list[dict],
    output_dir: str,
    window_sec: float = 5.0,
    output_dir_base: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    スクリーンショットと文字起こしセグメントを照合して同期マップを生成する。

    Parameters
    ----------
    video_path : str
        元の動画ファイルパス（メタデータとして記録するのみ）。
    frames : list[dict]
        screenshot.extract_changed_frames() の戻り値。
        各要素: {"timestamp": float, "filename": str, "filepath": str}
    segments : list[dict]
        transcribe.transcribe_video() の戻り値["segments"]。
        各要素: {"start": float, "end": float, "text": str, ...}
    output_dir : str
        sync_map.json の保存先ディレクトリ。
    window_sec : float
        各フレームに紐付ける「前後ウィンドウ」の秒数（デフォルト: 5.0）。
        transcript_window には [timestamp - window_sec, timestamp + window_sec]
        の範囲に含まれるセグメントが入る。
    output_dir_base : str, optional
        filepath を相対パスに変換する際の基準ディレクトリ。
        None の場合は絶対パスのまま記録する。
    verbose : bool
        進捗メッセージを表示するか。

    Returns
    -------
    dict
        同期マップの辞書（sync_map.json の内容と同一）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 動画の総時間を推定（最後のフレームのタイムスタンプ、または最後のセグメントの終端）
    total_duration = 0.0
    if frames:
        total_duration = max(total_duration, frames[-1]["timestamp"])
    if segments:
        total_duration = max(total_duration, segments[-1].get("end", 0.0))

    # フレームリストをソート（念のため）
    sorted_frames = sorted(frames, key=lambda f: f["timestamp"])

    # セグメントリストをソート
    sorted_segments = sorted(segments, key=lambda s: s["start"])

    def _seconds_to_timestamp(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"[{h:02d}:{m:02d}:{s:02d}]"

    def _make_segment_entry(seg: dict) -> dict:
        return {
            "start": round(seg["start"], 3),
            "end": round(seg.get("end", seg["start"]), 3),
            "timestamp": _seconds_to_timestamp(seg["start"]),
            "text": seg.get("text", "").strip(),
        }

    def _get_filepath(frame: dict) -> str:
        """filepath を output_dir_base からの相対パスに変換する（可能な場合）。"""
        fp = Path(frame["filepath"])
        if output_dir_base:
            try:
                return str(fp.relative_to(output_dir_base))
            except ValueError:
                pass
        return frame["filepath"]

    # 各フレームに対してセグメントを照合する
    frame_entries = []
    for i, frame in enumerate(sorted_frames):
        ts = frame["timestamp"]

        # 次のフレームのタイムスタンプ（このフレームが「映っている」終端）
        next_ts = sorted_frames[i + 1]["timestamp"] if i + 1 < len(sorted_frames) else total_duration

        # transcript_during: このフレームが表示されている間（ts ≦ seg.start < next_ts）
        during = [
            _make_segment_entry(seg)
            for seg in sorted_segments
            if ts <= seg["start"] < next_ts
        ]

        # transcript_before: 直前のフレームから現在フレームまでの間に終わった発話
        prev_ts = sorted_frames[i - 1]["timestamp"] if i > 0 else 0.0
        before = [
            _make_segment_entry(seg)
            for seg in sorted_segments
            if prev_ts <= seg["start"] < ts
        ]

        # transcript_window: タイムスタンプ前後 window_sec 秒以内の全発話
        window_start = max(0.0, ts - window_sec)
        window_end = ts + window_sec
        window = [
            _make_segment_entry(seg)
            for seg in sorted_segments
            if window_start <= seg["start"] <= window_end
        ]

        frame_entries.append({
            "timestamp": round(ts, 3),
            "filename": frame["filename"],
            "filepath": _get_filepath(frame),
            "transcript_before": before,
            "transcript_during": during,
            "transcript_window": window,
        })

    # 同期マップの構築
    sync_map = {
        "video": str(video_path),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_duration": round(total_duration, 3),
        "total_frames": len(frame_entries),
        "total_segments": len(sorted_segments),
        "window_sec": window_sec,
        "frames": frame_entries,
    }

    # JSON として保存
    video_stem = Path(video_path).stem
    out_path = output_dir / f"{video_stem}_sync_map.json"
    out_path.write_text(
        json.dumps(sync_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print(f"[sync_map] 完了: {len(frame_entries)} フレーム × {len(sorted_segments)} セグメント → {out_path}")

    return sync_map


def sync_map_to_ai_prompt(sync_map: dict, max_frames: int = 0) -> str:
    """
    同期マップを「AIに渡すためのテキスト表現」に変換する。

    各フレームについて「このスクリーンショットが映っていたとき、
    何が話されていたか」を時系列順に記述したテキストを返す。

    Parameters
    ----------
    sync_map : dict
        build_sync_map() の戻り値。
    max_frames : int
        出力するフレーム数の上限（0 = 全フレーム）。

    Returns
    -------
    str
        AIに渡すためのテキスト表現。
    """
    lines = []
    lines.append(f"# 動画同期マップ")
    lines.append(f"- 動画: {sync_map['video']}")
    lines.append(f"- 総時間: {sync_map['total_duration']:.1f}秒")
    lines.append(f"- スクリーンショット数: {sync_map['total_frames']}枚")
    lines.append(f"- 文字起こしセグメント数: {sync_map['total_segments']}件")
    lines.append("")

    frames = sync_map["frames"]
    if max_frames > 0:
        frames = frames[:max_frames]

    for frame in frames:
        ts = frame["timestamp"]
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ts_str = f"{h:02d}:{m:02d}:{s:02d}"

        lines.append(f"## [{ts_str}] {frame['filename']}")

        during = frame.get("transcript_during", [])
        before = frame.get("transcript_before", [])

        if before:
            lines.append("**（直前の発話）**")
            for seg in before:
                lines.append(f"  {seg['timestamp']} {seg['text']}")

        if during:
            lines.append("**（この画面が表示されている間の発話）**")
            for seg in during:
                lines.append(f"  {seg['timestamp']} {seg['text']}")
        else:
            lines.append("  *（この画面の間に発話なし）*")

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("使い方: python sync_map.py <sync_map.json> [max_frames]")
        sys.exit(1)
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    max_f = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(sync_map_to_ai_prompt(data, max_frames=max_f))
