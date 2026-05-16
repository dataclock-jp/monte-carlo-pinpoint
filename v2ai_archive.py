"""
v2ai_archive.py
自己完結型アーカイブ生成・読み出しモジュール

画像ファイル・文字起こし・同期マップを1つの SQLite ファイル（.v2ai）に統合します。
外部ファイルへの依存がゼロになるため、.v2ai ファイルを渡すだけで
すべての情報が再現できます。

ファイル拡張子: .v2ai（実体は SQLite3 データベース）

スキーマ:
  video_meta   : 動画のメタデータ（キーバリュー形式）
  frames       : スクリーンショット（画像バイナリを BLOB として格納）
  segments     : 文字起こしセグメント
  sync_entries : 各フレームと対応するセグメントの紐付け（sync_map の実体）

使い方:
  # アーカイブを作成
  archive = V2AIArchive("output/video.v2ai")
  archive.create(video_path, frames, segments, window_sec=5.0)

  # アーカイブから画像を取り出す
  image_bytes = archive.get_frame_image(timestamp=912.9)

  # アーカイブから同期マップを取り出す（JSON互換の辞書）
  sync_map = archive.get_sync_map()

  # AIに渡すためのテキスト表現を生成
  prompt = archive.to_ai_prompt(max_frames=20)
"""

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# .v2ai ファイルのマジックナンバー（user_version として SQLite に埋め込む）
V2AI_VERSION = 1


class V2AIArchive:
    """
    Video2AI 自己完結型アーカイブ。

    SQLite3 を使って画像・文字起こし・同期マップを 1 ファイルに統合します。
    """

    def __init__(self, archive_path: str):
        self.archive_path = Path(archive_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.archive_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # スキーマ作成
    # ------------------------------------------------------------------

    def _create_schema(self, conn: sqlite3.Connection):
        conn.executescript("""
            PRAGMA user_version = 1;

            CREATE TABLE IF NOT EXISTS video_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS frames (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    REAL    NOT NULL,
                filename     TEXT    NOT NULL,
                image_data   BLOB    NOT NULL,
                image_format TEXT    NOT NULL DEFAULT 'jpg'
            );
            CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames(timestamp);

            CREATE TABLE IF NOT EXISTS segments (
                id    INTEGER PRIMARY KEY AUTOINCREMENT,
                start REAL NOT NULL,
                end   REAL NOT NULL,
                text  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_segments_start ON segments(start);

            CREATE TABLE IF NOT EXISTS sync_entries (
                frame_id    INTEGER NOT NULL REFERENCES frames(id),
                segment_id  INTEGER NOT NULL REFERENCES segments(id),
                relation    TEXT    NOT NULL DEFAULT 'during',
                PRIMARY KEY (frame_id, segment_id, relation)
            );
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # アーカイブ作成
    # ------------------------------------------------------------------

    def create(
        self,
        video_path: str,
        frames: list[dict],
        segments: list[dict],
        window_sec: float = 5.0,
        verbose: bool = True,
    ) -> None:
        """
        アーカイブを新規作成する。

        Parameters
        ----------
        video_path : str
            元の動画ファイルパス（メタデータとして記録するのみ）。
        frames : list[dict]
            screenshot.extract_changed_frames() の戻り値。
            各要素: {"timestamp": float, "filename": str, "filepath": str}
        segments : list[dict]
            transcribe.transcribe_video() の戻り値["segments"]。
            各要素: {"start": float, "end": float, "text": str}
        window_sec : float
            sync_window: 各フレームに紐付ける前後ウィンドウ幅（秒）。
        verbose : bool
            進捗メッセージを表示するか。
        """
        conn = self._connect()
        self._create_schema(conn)

        # --- メタデータ ---
        total_duration = 0.0
        if frames:
            total_duration = max(total_duration, frames[-1]["timestamp"])
        if segments:
            total_duration = max(total_duration, segments[-1].get("end", 0.0))

        meta = {
            "video_path": str(video_path),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_duration": str(round(total_duration, 3)),
            "total_frames": str(len(frames)),
            "total_segments": str(len(segments)),
            "window_sec": str(window_sec),
            "v2ai_version": str(V2AI_VERSION),
        }
        conn.executemany(
            "INSERT OR REPLACE INTO video_meta (key, value) VALUES (?, ?)",
            meta.items(),
        )
        conn.commit()

        # --- フレーム（画像 BLOB）---
        sorted_frames = sorted(frames, key=lambda f: f["timestamp"])
        frame_id_map: dict[float, int] = {}

        for frame in sorted_frames:
            fp = Path(frame["filepath"])
            if not fp.exists():
                if verbose:
                    print(f"[v2ai] 警告: 画像ファイルが見つかりません: {fp}")
                continue
            image_data = fp.read_bytes()
            image_format = fp.suffix.lstrip(".").lower()
            cur = conn.execute(
                "INSERT INTO frames (timestamp, filename, image_data, image_format) VALUES (?, ?, ?, ?)",
                (round(frame["timestamp"], 3), frame["filename"], image_data, image_format),
            )
            frame_id_map[round(frame["timestamp"], 3)] = cur.lastrowid
        conn.commit()

        if verbose:
            print(f"[v2ai] フレーム格納: {len(frame_id_map)} 枚")

        # --- セグメント ---
        sorted_segments = sorted(segments, key=lambda s: s["start"])
        segment_ids: list[int] = []

        for seg in sorted_segments:
            cur = conn.execute(
                "INSERT INTO segments (start, end, text) VALUES (?, ?, ?)",
                (round(seg["start"], 3), round(seg.get("end", seg["start"]), 3), seg.get("text", "").strip()),
            )
            segment_ids.append(cur.lastrowid)
        conn.commit()

        if verbose:
            print(f"[v2ai] セグメント格納: {len(segment_ids)} 件")

        # --- 同期エントリ（sync_entries）---
        sync_rows: list[tuple] = []
        frame_timestamps = sorted(frame_id_map.keys())

        for i, ts in enumerate(frame_timestamps):
            fid = frame_id_map[ts]
            next_ts = frame_timestamps[i + 1] if i + 1 < len(frame_timestamps) else total_duration
            prev_ts = frame_timestamps[i - 1] if i > 0 else 0.0
            window_start = max(0.0, ts - window_sec)
            window_end = ts + window_sec

            for j, seg in enumerate(sorted_segments):
                sid = segment_ids[j]
                seg_start = round(seg["start"], 3)

                if ts <= seg_start < next_ts:
                    sync_rows.append((fid, sid, "during"))
                elif prev_ts <= seg_start < ts:
                    sync_rows.append((fid, sid, "before"))
                if window_start <= seg_start <= window_end:
                    sync_rows.append((fid, sid, "window"))

        # 重複を除去して挿入
        unique_rows = list({(r[0], r[1], r[2]): r for r in sync_rows}.values())
        conn.executemany(
            "INSERT OR IGNORE INTO sync_entries (frame_id, segment_id, relation) VALUES (?, ?, ?)",
            unique_rows,
        )
        conn.commit()

        if verbose:
            print(f"[v2ai] 同期エントリ: {len(unique_rows)} 件")
            print(f"[v2ai] 完了: {self.archive_path} ({self.archive_path.stat().st_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # 読み出し API
    # ------------------------------------------------------------------

    def get_frame_image(self, timestamp: float, tolerance: float = 0.1) -> Optional[bytes]:
        """
        指定タイムスタンプに最も近いフレームの画像バイナリを返す。

        Parameters
        ----------
        timestamp : float
            取得したいフレームのタイムスタンプ（秒）。
        tolerance : float
            許容誤差（秒）。この範囲内に一致するフレームがなければ None を返す。

        Returns
        -------
        bytes or None
            画像バイナリ（JPEG または PNG）。
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT image_data FROM frames WHERE ABS(timestamp - ?) <= ? ORDER BY ABS(timestamp - ?) LIMIT 1",
            (timestamp, tolerance, timestamp),
        ).fetchone()
        return bytes(row["image_data"]) if row else None

    def get_frame_image_base64(self, timestamp: float, tolerance: float = 0.1) -> Optional[str]:
        """画像を base64 エンコードした文字列で返す（MCP・API 連携用）。"""
        data = self.get_frame_image(timestamp, tolerance)
        return base64.b64encode(data).decode("utf-8") if data else None

    def get_all_frames(self) -> list[dict]:
        """全フレームのメタデータ（画像バイナリなし）を返す。"""
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, timestamp, filename, image_format FROM frames ORDER BY timestamp"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_segments_for_frame(self, frame_id: int, relation: str = "during") -> list[dict]:
        """
        指定フレームに紐付いたセグメントを返す。

        Parameters
        ----------
        frame_id : int
            frames テーブルの id。
        relation : str
            'during'（この画面の間）, 'before'（直前）, 'window'（前後ウィンドウ）
        """
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT s.id, s.start, s.end, s.text
            FROM segments s
            JOIN sync_entries e ON s.id = e.segment_id
            WHERE e.frame_id = ? AND e.relation = ?
            ORDER BY s.start
            """,
            (frame_id, relation),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_sync_map(self) -> dict:
        """
        sync_map.py の build_sync_map() と同等の辞書を返す（JSON互換）。
        画像データは含まず、ファイル名とタイムスタンプのみ。
        """
        conn = self._connect()
        meta_rows = conn.execute("SELECT key, value FROM video_meta").fetchall()
        meta = {r["key"]: r["value"] for r in meta_rows}

        frames_data = self.get_all_frames()
        result_frames = []

        for frame in frames_data:
            fid = frame["id"]

            def _seg_entry(seg: dict) -> dict:
                h = int(seg["start"] // 3600)
                m = int((seg["start"] % 3600) // 60)
                s = int(seg["start"] % 60)
                return {
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "timestamp": f"[{h:02d}:{m:02d}:{s:02d}]",
                    "text": seg["text"],
                }

            result_frames.append({
                "timestamp": frame["timestamp"],
                "filename": frame["filename"],
                "image_format": frame["image_format"],
                "transcript_before": [_seg_entry(s) for s in self.get_segments_for_frame(fid, "before")],
                "transcript_during": [_seg_entry(s) for s in self.get_segments_for_frame(fid, "during")],
                "transcript_window": [_seg_entry(s) for s in self.get_segments_for_frame(fid, "window")],
            })

        return {
            "video": meta.get("video_path", ""),
            "generated_at": meta.get("generated_at", ""),
            "total_duration": float(meta.get("total_duration", 0)),
            "total_frames": int(meta.get("total_frames", 0)),
            "total_segments": int(meta.get("total_segments", 0)),
            "window_sec": float(meta.get("window_sec", 5.0)),
            "frames": result_frames,
        }

    def to_ai_prompt(self, max_frames: int = 0) -> str:
        """
        アーカイブの内容を AI に渡すためのテキスト表現に変換する。
        sync_map.sync_map_to_ai_prompt() と同等の出力。
        """
        sync_map = self.get_sync_map()
        from sync_map import sync_map_to_ai_prompt
        return sync_map_to_ai_prompt(sync_map, max_frames=max_frames)

    def export_frames(self, output_dir: str, verbose: bool = True) -> list[Path]:
        """
        アーカイブから全フレーム画像をファイルとして書き出す。
        （アーカイブから個別ファイルへの逆変換）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        rows = conn.execute(
            "SELECT timestamp, filename, image_data, image_format FROM frames ORDER BY timestamp"
        ).fetchall()

        exported = []
        for row in rows:
            out_path = output_dir / row["filename"]
            out_path.write_bytes(bytes(row["image_data"]))
            exported.append(out_path)

        if verbose:
            print(f"[v2ai] {len(exported)} 枚の画像を {output_dir} に書き出しました")
        return exported

    def get_meta(self) -> dict:
        """メタデータを辞書で返す。"""
        conn = self._connect()
        rows = conn.execute("SELECT key, value FROM video_meta").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def info(self) -> str:
        """アーカイブの概要を文字列で返す。"""
        meta = self.get_meta()
        size_kb = self.archive_path.stat().st_size / 1024
        lines = [
            f"Video2AI Archive: {self.archive_path.name}",
            f"  動画      : {meta.get('video_path', 'N/A')}",
            f"  生成日時  : {meta.get('generated_at', 'N/A')}",
            f"  総時間    : {meta.get('total_duration', 'N/A')} 秒",
            f"  フレーム  : {meta.get('total_frames', 'N/A')} 枚",
            f"  セグメント: {meta.get('total_segments', 'N/A')} 件",
            f"  ファイル  : {size_kb:.1f} KB",
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------
# コマンドラインユーティリティ
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使い方:")
        print("  python v2ai_archive.py info <archive.v2ai>")
        print("  python v2ai_archive.py export <archive.v2ai> <output_dir>")
        print("  python v2ai_archive.py prompt <archive.v2ai> [max_frames]")
        sys.exit(1)

    command = sys.argv[1]
    archive_path = sys.argv[2]

    with V2AIArchive(archive_path) as archive:
        if command == "info":
            print(archive.info())

        elif command == "export":
            out_dir = sys.argv[3] if len(sys.argv) > 3 else "exported_frames"
            archive.export_frames(out_dir)

        elif command == "prompt":
            max_f = int(sys.argv[3]) if len(sys.argv) > 3 else 0
            print(archive.to_ai_prompt(max_frames=max_f))

        else:
            print(f"不明なコマンド: {command}")
            sys.exit(1)
