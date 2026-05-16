"""
long_term_memory.py
長期記憶モジュール — SQLite + FTS5 + sqlite-vec バックエンド

AIが操作中に発見した知見を永続的に保存し、次のセッションで活用する。
人間の長期記憶（エピソード記憶＋意味記憶）に相当する。

app_knowledge（静的・手動作成）との違い:
  - AIが自分で書き込む（動的）
  - 操作経験から得た具体的な知見
  - 成功/失敗の文脈を含む
  - 時間とともに成長する

記憶の種類（category フィールド）:
  - operational: 操作方法の知見
  - pitfall: 失敗から学んだ注意点
  - pattern: 成功パターン
  - app_specific: アプリ固有の発見

記憶のレベル（entry_type フィールド、schema v3 以降）:
  - fact: 短文の操作知識 (title+content の既存形式)
  - segment: 動画/画面/因果関係を含む Semantic Segment (option b')
    intent + action_log + ocr_dump + ui_graph_json + timestamp 等

永続化: SQLite (video2ai_memory.db)
  - memories: 本体テーブル (fact と segment が同居)
  - memories_fts: FTS5 全文検索 (title, content, context, tags,
    intent, action_log, ocr_dump の 7 カラム)
  - vec_memories: sqlite-vec による semantic search 用 virtual table
    (schema v3 以降、sqlite-vec が import 可能な場合のみ作成)
旧バックエンド: JSON ファイル（初回起動時に自動マイグレーション）
"""
import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Optional, List
from pathlib import Path


# ============================================================
# マイグレーション（PRAGMA user_version 方式 — Neo側と共通パターン）
# ============================================================
_MIGRATIONS = [
    # version 1: initial schema
    """
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        category TEXT NOT NULL,
        app TEXT NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        context TEXT DEFAULT '',
        confidence REAL DEFAULT 0.5,
        created_at REAL DEFAULT 0,
        updated_at REAL DEFAULT 0,
        access_count INTEGER DEFAULT 0,
        tags TEXT DEFAULT '[]'
    );
    CREATE INDEX IF NOT EXISTS idx_memories_app ON memories(app);
    CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
    """,
    # version 2: FTS5 full-text search
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(title, content, context, tags, content=memories, content_rowid=rowid);

    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, title, content, context, tags)
        VALUES (new.rowid, new.title, new.content, new.context, new.tags);
    END;
    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, title, content, context, tags)
        VALUES ('delete', old.rowid, old.title, old.content, old.context, old.tags);
    END;
    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, title, content, context, tags)
        VALUES ('delete', old.rowid, old.title, old.content, old.context, old.tags);
        INSERT INTO memories_fts(rowid, title, content, context, tags)
        VALUES (new.rowid, new.title, new.content, new.context, new.tags);
    END;
    """,
    # version 3: option (b') Memory Segment extension
    # Adds 10 NULLABLE columns to `memories` for Autobiographical Causality
    # (intent + action_log + ocr_dump + ui_graph_json + temporal/spatial
    # anchors + embedding_blob) and rebuilds `memories_fts` to include
    # intent/action_log/ocr_dump per supervisor decision 019.
    # ui_graph_json deliberately NOT in FTS5 — use JSON1 access instead.
    # The vec_memories virtual table (sqlite-vec) is created at runtime
    # in _connect() after loading the sqlite-vec extension, since
    # extensions can only load on an opened connection.
    """
    -- 1. Add 10 NULLABLE segment columns
    ALTER TABLE memories ADD COLUMN entry_type TEXT DEFAULT 'fact';
    ALTER TABLE memories ADD COLUMN intent TEXT;
    ALTER TABLE memories ADD COLUMN action_log TEXT;
    ALTER TABLE memories ADD COLUMN ocr_dump TEXT;
    ALTER TABLE memories ADD COLUMN ui_graph_json TEXT;
    ALTER TABLE memories ADD COLUMN source_path TEXT;
    ALTER TABLE memories ADD COLUMN timestamp_start REAL;
    ALTER TABLE memories ADD COLUMN timestamp_end REAL;
    ALTER TABLE memories ADD COLUMN monitor_id INTEGER;
    ALTER TABLE memories ADD COLUMN embedding_blob BLOB;

    -- 2. Drop existing FTS5 triggers + virtual table
    DROP TRIGGER IF EXISTS memories_ai;
    DROP TRIGGER IF EXISTS memories_ad;
    DROP TRIGGER IF EXISTS memories_au;
    DROP TABLE IF EXISTS memories_fts;

    -- 3. Recreate FTS5 virtual table with 7 columns
    --    (title, content, context, tags, intent, action_log, ocr_dump)
    CREATE VIRTUAL TABLE memories_fts USING fts5(
        title, content, context, tags,
        intent, action_log, ocr_dump,
        content=memories, content_rowid=rowid
    );

    -- 4. Recreate synchronisation triggers with the new columns
    CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, title, content, context, tags,
                                  intent, action_log, ocr_dump)
        VALUES (new.rowid, new.title, new.content, new.context, new.tags,
                new.intent, new.action_log, new.ocr_dump);
    END;
    CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, title, content,
                                  context, tags, intent, action_log, ocr_dump)
        VALUES ('delete', old.rowid, old.title, old.content,
                old.context, old.tags, old.intent, old.action_log, old.ocr_dump);
    END;
    CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, title, content,
                                  context, tags, intent, action_log, ocr_dump)
        VALUES ('delete', old.rowid, old.title, old.content,
                old.context, old.tags, old.intent, old.action_log, old.ocr_dump);
        INSERT INTO memories_fts(rowid, title, content, context, tags,
                                  intent, action_log, ocr_dump)
        VALUES (new.rowid, new.title, new.content, new.context, new.tags,
                new.intent, new.action_log, new.ocr_dump);
    END;

    -- 5. Rebuild FTS5 index for all existing rows (new columns are NULL
    --    on pre-v3 rows; they become FTS-indexed as NULL which is fine)
    INSERT INTO memories_fts(memories_fts) VALUES('rebuild');

    -- 6. Index on entry_type for efficient fact/segment filtering
    CREATE INDEX IF NOT EXISTS idx_memories_entry_type
        ON memories(entry_type);
    """,
]

# Embedding dimension used by vec_memories. Must match the Embedding
# model in use (multilingual-e5-small = 384). Changing this requires a
# new migration + re-embedding.
EMBEDDING_DIM = 384

# Embedding model identifier. Decision 018: multilingual-e5-small for
# Japanese + English support. Latency measured at ~12.8ms on CPU in
# pre-flight verification.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Singleton embedding model, lazy-loaded on first use so that importing
# long_term_memory.py does not pay the ~2-second sentence-transformers
# startup cost when only the fact interface is used.
_embedding_model = None


def _get_embedding_model():
    """Lazy-load the singleton SentenceTransformer instance.

    Returns None if sentence-transformers is not installed; callers
    should treat this as a graceful degradation signal (embeddings are
    skipped, vec_memories is not populated, but the memory segment row
    is still written to the main table with embedding_blob=NULL).
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _encode_text_for_segment(text: str) -> Optional[bytes]:
    """Encode a segment's searchable text into a float32 blob for vec_memories.

    multilingual-e5 expects a task prefix: 'passage: ' for stored content,
    'query: ' for query text. This function is for stored content.

    Returns None if the embedding model is unavailable — downstream code
    must treat that as 'no vector search for this row' and fall back to
    FTS5-only matching.
    """
    model = _get_embedding_model()
    if model is None:
        return None
    vec = model.encode(f"passage: {text}", normalize_embeddings=True)
    import struct
    return struct.pack(f"{EMBEDDING_DIM}f", *vec.tolist())


def _encode_text_for_query(text: str) -> Optional[bytes]:
    """Encode a search query into a float32 blob for vec_memories lookup.

    Uses the 'query: ' prefix required by multilingual-e5. Returns None
    if the embedding model is unavailable.
    """
    model = _get_embedding_model()
    if model is None:
        return None
    vec = model.encode(f"query: {text}", normalize_embeddings=True)
    import struct
    return struct.pack(f"{EMBEDDING_DIM}f", *vec.tolist())


def _migrate(conn: sqlite3.Connection) -> None:
    """PRAGMA user_version ベースのマイグレーション。

    Each migration step runs inside a single SAVEPOINT so that if any
    statement fails, the whole step (and its PRAGMA user_version bump)
    rolls back together, matching Neo's migration pattern.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for i, sql in enumerate(_MIGRATIONS[current:], start=current):
        savepoint = f"migrate_v{i + 1}"
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            conn.executescript(sql)
            conn.execute(f"PRAGMA user_version = {i + 1}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
    conn.commit()


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """sqlite-vec 拡張を読み込み、vec_memories virtual table を作成する。

    sqlite_vec パッケージが import できない場合、または Python SQLite
    ビルドが enable_load_extension を有効にしていない場合は False を
    返す。vec 機能なしでも長期記憶モジュールは動作する (FTS5 のみで
    動く) ため、失敗しても例外にはしない。
    """
    try:
        import sqlite_vec  # type: ignore
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
    except sqlite3.NotSupportedError:
        return False
    try:
        sqlite_vec.load(conn)
    finally:
        try:
            conn.enable_load_extension(False)
        except sqlite3.NotSupportedError:
            pass
    # Create the virtual table if it does not already exist. sqlite-vec
    # supports CREATE VIRTUAL TABLE IF NOT EXISTS natively.
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0("
        f"  rowid INTEGER PRIMARY KEY,"
        f"  embedding FLOAT[{EMBEDDING_DIM}]"
        f")"
    )
    conn.commit()
    return True


# ============================================================
# データモデル（既存互換 + schema v3 拡張）
# ============================================================
@dataclass
class MemoryEntry:
    """長期記憶の1エントリ。

    schema v2 以前は title/content/context/category/tags/confidence の
    短文事実のみを保持していたが、schema v3 (2026-04 option b') で
    Semantic Segment のフィールド (entry_type, intent, action_log,
    ocr_dump, ui_graph_json, source_path, timestamp_start/end,
    monitor_id) が追加された。これらは全て NULLABLE / Optional で、
    entry_type == 'fact' のエントリでは None のまま運用される。
    """
    id: str
    category: str
    app: str
    title: str
    content: str
    context: str
    confidence: float
    created_at: float
    updated_at: float
    access_count: int = 0
    tags: list = None
    # --- schema v3 additions (Memory Segment / Autobiographical Causality)
    entry_type: str = "fact"
    intent: Optional[str] = None
    action_log: Optional[str] = None
    ocr_dump: Optional[str] = None
    ui_graph_json: Optional[str] = None
    source_path: Optional[str] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    monitor_id: Optional[int] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# ============================================================
# メインクラス
# ============================================================
class LongTermMemory:
    """
    AIの長期記憶を管理するモジュール。
    バックエンド: SQLite + FTS5。

    操作中に発見した知見を永続的に保存し、
    次のセッションで読み込んで活用する。
    """

    MAX_ENTRIES = 500

    def __init__(self, storage_dir: str = ""):
        if storage_dir:
            base = Path(storage_dir)
        else:
            base = Path.home() / "video2ai_agent_sessions"
        base.mkdir(parents=True, exist_ok=True)

        self._db_path = base / "video2ai_memory.db"
        self._legacy_json = base / "long_term_memory.json"
        self._conn = self._connect()
        _migrate(self._conn)
        self._vec_enabled = _try_load_sqlite_vec(self._conn)
        self._migrate_from_json()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def vec_enabled(self) -> bool:
        """sqlite-vec による semantic search が利用可能か。

        False の場合でも FTS5 ベースの全文検索は動作する。
        """
        return self._vec_enabled

    def _migrate_from_json(self) -> None:
        """旧 JSON ファイルからの一回限りのマイグレーション。"""
        if not self._legacy_json.exists():
            return
        # DB に既にデータがあればスキップ
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count > 0:
            return

        try:
            with open(self._legacy_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data.values():
                tags = d.get("tags", [])
                self._conn.execute(
                    "INSERT OR IGNORE INTO memories "
                    "(id, category, app, title, content, context, confidence, "
                    "created_at, updated_at, access_count, tags) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (d["id"], d["category"], d["app"], d["title"],
                     d["content"], d.get("context", ""),
                     d.get("confidence", 0.5),
                     d.get("created_at", 0), d.get("updated_at", 0),
                     d.get("access_count", 0),
                     json.dumps(tags, ensure_ascii=False)),
                )
            self._conn.commit()
            # 旧ファイルをリネーム（削除ではなくバックアップ）
            backup = self._legacy_json.with_suffix(".json.bak")
            self._legacy_json.rename(backup)
        except Exception:
            pass

    def learn(self, category: str, app: str, title: str, content: str,
              context: str = "", confidence: float = 0.7,
              tags: list = None) -> MemoryEntry:
        """新しい知見を記憶する。同じタイトル+アプリの既存エントリがあれば更新。"""
        tags = tags or []
        now = time.time()

        existing = self._find_by_title(app, title)
        if existing:
            new_confidence = min(1.0, existing.confidence + 0.1)
            merged_tags = list(set(
                (existing.tags if existing.tags else []) + tags
            ))
            self._conn.execute(
                "UPDATE memories SET content=?, context=?, confidence=?, "
                "updated_at=?, tags=? WHERE id=?",
                (content, context, new_confidence, now,
                 json.dumps(merged_tags, ensure_ascii=False), existing.id),
            )
            self._conn.commit()
            existing.content = content
            existing.context = context
            existing.confidence = new_confidence
            existing.updated_at = now
            existing.tags = merged_tags
            return existing

        entry_id = f"ltm_{int(now * 1000)}"
        self._conn.execute(
            "INSERT INTO memories "
            "(id, category, app, title, content, context, confidence, "
            "created_at, updated_at, access_count, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (entry_id, category, app, title, content, context,
             confidence, now, now,
             json.dumps(tags, ensure_ascii=False)),
        )
        self._conn.commit()
        self._evict_if_needed()

        return MemoryEntry(
            id=entry_id, category=category, app=app, title=title,
            content=content, context=context, confidence=confidence,
            created_at=now, updated_at=now, access_count=0, tags=tags,
        )

    def learn_segment(
        self,
        intent: str,
        action_log: str = "",
        ocr_dump: str = "",
        ui_graph_json: str = "",
        source_path: str = "",
        timestamp_start: Optional[float] = None,
        timestamp_end: Optional[float] = None,
        monitor_id: Optional[int] = None,
        app: str = "*",
        category: str = "pattern",
        confidence: float = 0.7,
        tags: Optional[list] = None,
        title: str = "",
    ) -> MemoryEntry:
        """Semantic Segment (Autobiographical Causality) を記憶する。

        option (b') のシングルテーブル設計: 既存 memories テーブルに
        entry_type='segment' として INSERT する。Embedding は intent +
        action_log + ocr_dump を結合したテキストから multilingual-e5-small
        で生成し、vec_memories virtual table に同 rowid で挿入する。

        Args:
            intent: 操作の意図 (LLM 要約、必須)
            action_log: 実行した action (例: "Click(coord=(100,200))")
            ocr_dump: action 後の画面 OCR テキスト (結果の状態)
            ui_graph_json: 構造化 UI グラフ (JSON 文字列、FTS5 対象外)
            source_path: 元動画/画像のパス (あれば)
            timestamp_start, timestamp_end: 時間範囲 (UNIX 秒 or 動画内秒)
            monitor_id: マルチモニタ環境での screen 識別
            app: アプリ名 (default: "*" 汎用)
            category: 記憶の種類 (既存と同じ語彙)
            confidence: 信頼度
            tags: タグリスト
            title: 明示的なタイトル (空なら intent を流用)

        Returns:
            作成された MemoryEntry (entry_type='segment')
        """
        tags = tags or []
        now = time.time()
        resolved_title = title.strip() or intent[:100]

        # Content フィールドは既存 fact 記憶と互換させるため、人間可読な
        # サマリーを入れる (FTS5 にも載る)
        content = intent
        context = ""
        if action_log:
            context = f"action: {action_log}"

        # 埋め込みテキスト: intent + action_log + ocr_dump (tokens 節約のため
        # 各フィールドを適度に切り詰める)
        embed_text_parts = [intent.strip()]
        if action_log.strip():
            embed_text_parts.append(action_log.strip()[:500])
        if ocr_dump.strip():
            embed_text_parts.append(ocr_dump.strip()[:1000])
        embed_text = " | ".join(embed_text_parts)
        embedding_blob = _encode_text_for_segment(embed_text)

        entry_id = f"seg_{int(now * 1000)}"

        self._conn.execute(
            "INSERT INTO memories "
            "(id, category, app, title, content, context, confidence, "
            " created_at, updated_at, access_count, tags, "
            " entry_type, intent, action_log, ocr_dump, ui_graph_json, "
            " source_path, timestamp_start, timestamp_end, monitor_id, "
            " embedding_blob) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, "
            "        'segment', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry_id, category, app, resolved_title, content, context,
                confidence, now, now,
                json.dumps(tags, ensure_ascii=False),
                intent, action_log or None, ocr_dump or None,
                ui_graph_json or None, source_path or None,
                timestamp_start, timestamp_end, monitor_id,
                embedding_blob,
            ),
        )
        # rowid needed for vec_memories INSERT
        new_rowid = self._conn.execute(
            "SELECT rowid FROM memories WHERE id = ?", (entry_id,)
        ).fetchone()[0]

        if embedding_blob is not None and self._vec_enabled:
            self._conn.execute(
                "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                (new_rowid, embedding_blob),
            )

        self._conn.commit()
        self._evict_if_needed()

        return MemoryEntry(
            id=entry_id, category=category, app=app, title=resolved_title,
            content=content, context=context, confidence=confidence,
            created_at=now, updated_at=now, access_count=0, tags=tags,
            entry_type="segment", intent=intent,
            action_log=action_log or None, ocr_dump=ocr_dump or None,
            ui_graph_json=ui_graph_json or None,
            source_path=source_path or None,
            timestamp_start=timestamp_start, timestamp_end=timestamp_end,
            monitor_id=monitor_id,
        )

    def recall_hybrid(
        self,
        query: str,
        entry_type: str = "all",
        fts_weight: float = 0.5,
        vector_weight: float = 0.5,
        app: str = "",
        category: str = "",
        limit: int = 5,
        time_range_start: Optional[float] = None,
        time_range_end: Optional[float] = None,
    ) -> List[tuple]:
        """Hybrid FTS5 + Vector recall with Reciprocal Rank Fusion (RRF).

        gemini-agent の第3部設計 + 監督 decision 025 の仕様に従う。
        - Stage 1: FTS5 top-N (N=limit*4) で候補を取得
        - Stage 2: vec_memories top-N で候補を取得 (vec_enabled の場合)
        - RRF 統合: rank_fts と rank_vec を Reciprocal Rank Fusion で
          1 / (k + rank) 形式で合算 (k=60、業界標準)
        - entry_type フィルタ ('fact' / 'segment' / 'all')
        - time_range フィルタ (v0.5.1, Option A セマンティクス):
          時間範囲 kwargs がどちらか指定された場合、timestamp_start
          / timestamp_end が NULL の row は **除外**される。指定が
          なければ全 row が通過する。

        Returns a list of tuples: (MemoryEntry, fts_score, vector_score,
        rrf_score). Pure FTS5 mode (no vec) sets vector_score=None. Pure
        vector mode (no FTS hit) sets fts_score=None.

        Args:
            query: 検索クエリ (人間言語、日本語/英語/混在可)
            entry_type: 'fact' | 'segment' | 'all'
            fts_weight, vector_weight: スコア合成時の重み (0.0-1.0、
                通常は等分の 0.5 / 0.5、正確な threshold は e5 の
                baseline cos sim が 0.78-0.81 と高いため絶対値では
                効かない → 常に rank-based RRF を使う)
            app: アプリ名フィルタ (空で全て、'*' は wildcard 扱い)
            category: category フィルタ (空で全て)
            limit: 最終返却件数
            time_range_start, time_range_end: 時間範囲フィルタ (UNIX
                秒 or 動画内秒、None で無効)。指定時は Option A
                (NULL 除外) セマンティクスで、timestamp_start/end が
                NULL の memory row は結果から除外される。監督
                decision 034 (interface topic) の確定仕様。
        """
        if not query.strip():
            return []
        rrf_k = 60
        fetch_n = limit * 4

        # FTS5 候補取得
        words = query.strip().split()
        fts_query = " OR ".join(f'"{w}"' for w in words if w)
        # SQLite の SELECT m.* は rowid を含まないので m.rowid を明示
        fts_sql = """
            SELECT m.rowid AS mem_rowid, m.*, f.rank AS fts_rank
            FROM memories m
            JOIN memories_fts f ON m.rowid = f.rowid
            WHERE memories_fts MATCH ?
        """
        fts_params: list = [fts_query]
        if entry_type != "all":
            fts_sql += " AND m.entry_type = ?"
            fts_params.append(entry_type)
        if app:
            fts_sql += " AND (m.app = ? OR m.app = '*')"
            fts_params.append(app)
        if category:
            fts_sql += " AND m.category = ?"
            fts_params.append(category)
        # Option A: when time_range kwargs are passed, rows with NULL
        # timestamp_start/end are excluded (NULL-aware comparison).
        if time_range_start is not None:
            fts_sql += " AND m.timestamp_start >= ?"
            fts_params.append(time_range_start)
        if time_range_end is not None:
            fts_sql += " AND m.timestamp_end <= ?"
            fts_params.append(time_range_end)
        fts_sql += " ORDER BY f.rank LIMIT ?"
        fts_params.append(fetch_n)
        fts_rows = self._conn.execute(fts_sql, fts_params).fetchall()

        fts_results: dict[int, dict] = {}
        for rank_idx, row in enumerate(fts_rows):
            fts_results[row["mem_rowid"]] = {
                "row": row,
                "fts_rank": rank_idx + 1,
                "fts_score": row["fts_rank"],
            }

        # Vector 候補取得 (vec_enabled && embedding model available)
        vec_results: dict[int, dict] = {}
        if self._vec_enabled:
            q_blob = _encode_text_for_query(query)
            if q_blob is not None:
                vec_sql = """
                    SELECT rowid, distance
                    FROM vec_memories
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance
                """
                vec_rows = self._conn.execute(
                    vec_sql, (q_blob, fetch_n)
                ).fetchall()
                # vec_memories is independent of `memories` filters, so
                # we need to re-fetch the actual row for entry_type/app/
                # category filtering below.
                vec_rowids = [r["rowid"] for r in vec_rows]
                if vec_rowids:
                    placeholders = ",".join("?" * len(vec_rowids))
                    filter_sql = f"""
                        SELECT m.rowid AS _rowid, m.*
                        FROM memories m
                        WHERE m.rowid IN ({placeholders})
                    """
                    filter_params: list = list(vec_rowids)
                    if entry_type != "all":
                        filter_sql += " AND m.entry_type = ?"
                        filter_params.append(entry_type)
                    if app:
                        filter_sql += " AND (m.app = ? OR m.app = '*')"
                        filter_params.append(app)
                    if category:
                        filter_sql += " AND m.category = ?"
                        filter_params.append(category)
                    # Option A: mirror the FTS branch's time_range filter
                    # so vector-only hits obey the same semantics.
                    if time_range_start is not None:
                        filter_sql += " AND m.timestamp_start >= ?"
                        filter_params.append(time_range_start)
                    if time_range_end is not None:
                        filter_sql += " AND m.timestamp_end <= ?"
                        filter_params.append(time_range_end)
                    filtered = {
                        r["_rowid"]: r
                        for r in self._conn.execute(
                            filter_sql, filter_params
                        ).fetchall()
                    }
                    rank_idx = 0
                    for vr in vec_rows:
                        rid = vr["rowid"]
                        if rid in filtered:
                            rank_idx += 1
                            vec_results[rid] = {
                                "row": filtered[rid],
                                "vec_rank": rank_idx,
                                "vec_score": float(vr["distance"]),
                            }

        # RRF 統合
        all_rowids = set(fts_results.keys()) | set(vec_results.keys())
        scored = []
        for rid in all_rowids:
            rrf = 0.0
            fts_info = fts_results.get(rid)
            vec_info = vec_results.get(rid)
            if fts_info is not None:
                rrf += fts_weight * (1.0 / (rrf_k + fts_info["fts_rank"]))
            if vec_info is not None:
                rrf += vector_weight * (1.0 / (rrf_k + vec_info["vec_rank"]))
            row = (fts_info or vec_info)["row"]
            entry = self._row_to_entry(row)
            fts_score = fts_info["fts_score"] if fts_info else None
            vec_score = vec_info["vec_score"] if vec_info else None
            scored.append((entry, fts_score, vec_score, rrf))

        scored.sort(key=lambda t: t[3], reverse=True)
        top = scored[:limit]
        self._bump_access([t[0] for t in top])
        return top

    def recall(self, app: str = "", category: str = "",
               query: str = "", limit: int = 10,
               entry_type: str = "") -> List[MemoryEntry]:
        """記憶を検索する。FTS5 でキーワード検索、app/category でフィルタ。

        Args:
            entry_type: 'fact' / 'segment' / '' (all) でフィルタ。
                decision 025 の entry_type フィルタ機能。
        """
        if query.strip():
            return self._recall_fts(app, category, query, limit, entry_type)
        return self._recall_filter(app, category, limit, entry_type)

    def _recall_fts(self, app: str, category: str,
                    query: str, limit: int,
                    entry_type: str = "") -> List[MemoryEntry]:
        """FTS5 による全文検索。entry_type フィルタ対応 (schema v3)."""
        words = query.strip().split()
        fts_query = " OR ".join(f'"{w}"' for w in words if w)
        if not fts_query:
            return self._recall_filter(app, category, limit, entry_type)

        sql = """
            SELECT m.* FROM memories m
            JOIN memories_fts f ON m.rowid = f.rowid
            WHERE memories_fts MATCH ?
        """
        params: list = [fts_query]

        if app:
            sql += " AND (m.app = ? OR m.app = '*')"
            params.append(app)
        if category:
            sql += " AND m.category = ?"
            params.append(category)
        if entry_type:
            sql += " AND m.entry_type = ?"
            params.append(entry_type)

        sql += " ORDER BY rank, m.access_count * m.confidence DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        self._bump_access(entries)
        return entries

    def _recall_filter(self, app: str, category: str,
                       limit: int,
                       entry_type: str = "") -> List[MemoryEntry]:
        """app / category / entry_type フィルタのみ（キーワードなし）。"""
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list = []

        if app:
            sql += " AND (app = ? OR app = '*')"
            params.append(app)
        if category:
            sql += " AND category = ?"
            params.append(category)
        if entry_type:
            sql += " AND entry_type = ?"
            params.append(entry_type)

        sql += " ORDER BY access_count * confidence DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        self._bump_access(entries)
        return entries

    def recall_as_prompt(self, app: str = "", limit: int = 5) -> str:
        """LLM プロンプトに注入する形式で記憶を返す。"""
        entries = self.recall(app=app, limit=limit)
        if not entries:
            return ""

        lines = [f"\n■ 長期記憶（過去の操作経験、{len(entries)}件）"]
        for e in entries:
            tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(f"  [{e.category}] {e.title}: {e.content}{tag_str}")
        return "\n".join(lines)

    def forget(self, entry_id: str) -> bool:
        """指定 ID の記憶を削除する。"""
        cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def stats(self) -> dict:
        """統計情報を返す。"""
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        cats = self._conn.execute(
            "SELECT category, COUNT(*) FROM memories GROUP BY category"
        ).fetchall()
        apps = self._conn.execute(
            "SELECT app, COUNT(*) FROM memories GROUP BY app"
        ).fetchall()
        return {
            "total_entries": total,
            "categories": {r[0]: r[1] for r in cats},
            "apps": {r[0]: r[1] for r in apps},
            "storage_path": str(self._db_path),
            "backend": "sqlite+fts5",
        }

    def _find_by_title(self, app: str, title: str) -> Optional[MemoryEntry]:
        """同じアプリ+タイトルの既存エントリを検索する。"""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE app = ? AND title = ? COLLATE NOCASE",
            (app, title),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def _evict_if_needed(self) -> None:
        """容量超過時に最も古く参照されていないエントリを削除する。"""
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count <= self.MAX_ENTRIES:
            return
        excess = count - self.MAX_ENTRIES
        self._conn.execute(
            "DELETE FROM memories WHERE id IN ("
            "  SELECT id FROM memories "
            "  ORDER BY access_count ASC, confidence ASC, updated_at ASC "
            "  LIMIT ?"
            ")",
            (excess,),
        )
        self._conn.commit()

    def _bump_access(self, entries: List[MemoryEntry]) -> None:
        """参照回数をインクリメントする。"""
        if not entries:
            return
        ids = [e.id for e in entries]
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"UPDATE memories SET access_count = access_count + 1, "
            f"updated_at = ? WHERE id IN ({placeholders})",
            [time.time()] + ids,
        )
        self._conn.commit()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        """sqlite3.Row → MemoryEntry 変換。schema v3 カラムは optional."""
        tags_raw = row["tags"]
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
            except (json.JSONDecodeError, TypeError):
                tags = []
        else:
            tags = tags_raw or []

        # schema v3 columns — present after migration, absent before.
        # sqlite3.Row supports 'col in row.keys()' for membership testing.
        keys = row.keys()

        def opt(col: str):
            return row[col] if col in keys else None

        entry_type = opt("entry_type") or "fact"

        return MemoryEntry(
            id=row["id"],
            category=row["category"],
            app=row["app"],
            title=row["title"],
            content=row["content"],
            context=row["context"] or "",
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
            tags=tags,
            entry_type=entry_type,
            intent=opt("intent"),
            action_log=opt("action_log"),
            ocr_dump=opt("ocr_dump"),
            ui_graph_json=opt("ui_graph_json"),
            source_path=opt("source_path"),
            timestamp_start=opt("timestamp_start"),
            timestamp_end=opt("timestamp_end"),
            monitor_id=opt("monitor_id"),
        )

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def __repr__(self) -> str:
        return f"LongTermMemory({len(self)} entries, backend=sqlite+fts5)"
