"""
dialectic.py
弁証法的推論フレームワーク

ヘーゲルの弁証法（テーゼ → アンチテーゼ → ジンテーゼ）をAIの思考プロセスに
適用する汎用推論エンジン。

いきなりゴールを目指すのではなく、意図的に反論・矛盾・別の視点を設定し、
それを止揚（アウフヘーベン）することで、より高い質の結論に到達する。

用途:
  - 操作計画の事前検証（「この操作で本当に正しいか？」）
  - 設計判断（「このアーキテクチャの弱点は？」）
  - バグ調査（「この仮説の矛盾点は？」）
  - 任意の思考タスク（「別の視点から見るとどうか？」）

思考の流れ:
  1. テーゼ（thesis）: 初期の主張・計画・仮説
  2. アンチテーゼ（antithesis）: テーゼへの反論・矛盾・別の視点
  3. ジンテーゼ（synthesis）: テーゼとアンチテーゼを止揚した高次の結論
  4. イテレーション: ジンテーゼを新たなテーゼとして繰り返し

参考: Hegel's dialectic, Aufheben (sublation)
"""
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import time


@dataclass
class DialecticalTriad:
    """弁証法の三項（テーゼ・アンチテーゼ・ジンテーゼ）の1ラウンド。"""
    round: int
    thesis: str
    antithesis: str = ""
    synthesis: str = ""
    thesis_timestamp: float = 0.0
    antithesis_timestamp: float = 0.0
    synthesis_timestamp: float = 0.0

    @property
    def is_complete(self) -> bool:
        return bool(self.thesis and self.antithesis and self.synthesis)

    @property
    def phase(self) -> str:
        if not self.antithesis:
            return "thesis"
        if not self.synthesis:
            return "antithesis"
        return "synthesis"


@dataclass
class DialecticalSession:
    """弁証法的推論のセッション。"""
    goal: str
    triads: List[DialecticalTriad] = field(default_factory=list)
    conclusion: str = ""
    created_at: float = 0.0

    @property
    def current_triad(self) -> Optional[DialecticalTriad]:
        return self.triads[-1] if self.triads else None

    @property
    def current_round(self) -> int:
        return len(self.triads)

    @property
    def phase(self) -> str:
        if self.conclusion:
            return "concluded"
        t = self.current_triad
        if not t:
            return "not_started"
        return t.phase


class DialecticalReasoner:
    """
    弁証法的推論を管理するエンジン。

    複数の名前付きセッションを並行管理できる。
    各セッションは複数ラウンドのテーゼ→アンチテーゼ→ジンテーゼを経て
    最終結論に到達する。

    永続化 (2026-04 追加): db_path を渡すと SQLite の
    dialectic_sessions テーブルに自動保存される。video2ai_memory.db
    と同じファイルを使用することで、記憶サブシステムの一部として
    他の認知アーティファクト (memories, vec_memories) と共存する。

    db_path を渡さなかった場合はインメモリのみ (従来動作)。
    DialecticalReasoner は cross-process 共有しない前提 (MCP サーバー
    プロセス内のシングルトン) なので、WAL モードと合わせて単一プロセス
    からの読み書きで十分。
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._sessions: dict[str, DialecticalSession] = {}
        self._db_path: Optional[Path] = db_path
        self._conn: Optional[sqlite3.Connection] = None
        if db_path is not None:
            self._init_db()
            self._load_all()

    def _init_db(self) -> None:
        """dialectic_sessions テーブルを CREATE IF NOT EXISTS で用意する。

        他の SQLite コンポーネント (long_term_memory の memories、
        vec_memories) と同じデータベースファイルを共有する想定。
        PRAGMA user_version の migration チェーンには含めず、自己完結
        で IF NOT EXISTS する — dialectic_sessions はバージョン間の
        スキーマ変更を持たないシンプルな永続ストレージであり、
        将来 v4 migration が必要になるまで独立扱いにする。
        """
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dialectic_sessions (
                name TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                phase TEXT NOT NULL,
                rounds_json TEXT NOT NULL,
                conclusion TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def _load_all(self) -> None:
        """DB から全セッションを読み込んでメモリに展開する。"""
        if self._conn is None:
            return
        rows = self._conn.execute(
            "SELECT name, goal, phase, rounds_json, conclusion, "
            "created_at, updated_at FROM dialectic_sessions"
        ).fetchall()
        for row in rows:
            try:
                rounds_data = json.loads(row["rounds_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            session = DialecticalSession(
                goal=row["goal"],
                created_at=row["created_at"],
                conclusion=row["conclusion"] or "",
            )
            for r in rounds_data:
                triad = DialecticalTriad(
                    round=r.get("round", 0),
                    thesis=r.get("thesis", ""),
                    antithesis=r.get("antithesis", ""),
                    synthesis=r.get("synthesis", ""),
                    thesis_timestamp=r.get("thesis_timestamp", 0.0),
                    antithesis_timestamp=r.get("antithesis_timestamp", 0.0),
                    synthesis_timestamp=r.get("synthesis_timestamp", 0.0),
                )
                session.triads.append(triad)
            self._sessions[row["name"]] = session

    def _save_session(self, name: str) -> None:
        """指定セッションを DB に upsert する。"""
        if self._conn is None:
            return
        session = self._sessions.get(name)
        if session is None:
            # Deletion: remove the row if present.
            self._conn.execute(
                "DELETE FROM dialectic_sessions WHERE name = ?", (name,)
            )
            self._conn.commit()
            return
        rounds_data = [
            {
                "round": t.round,
                "thesis": t.thesis,
                "antithesis": t.antithesis,
                "synthesis": t.synthesis,
                "thesis_timestamp": t.thesis_timestamp,
                "antithesis_timestamp": t.antithesis_timestamp,
                "synthesis_timestamp": t.synthesis_timestamp,
            }
            for t in session.triads
        ]
        now = time.time()
        self._conn.execute(
            "INSERT INTO dialectic_sessions "
            "(name, goal, phase, rounds_json, conclusion, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "  goal=excluded.goal, phase=excluded.phase, "
            "  rounds_json=excluded.rounds_json, "
            "  conclusion=excluded.conclusion, "
            "  updated_at=excluded.updated_at",
            (
                name,
                session.goal,
                session.phase,
                json.dumps(rounds_data, ensure_ascii=False),
                session.conclusion,
                session.created_at,
                now,
            ),
        )
        self._conn.commit()

    def start(self, name: str, goal: str, thesis: str) -> DialecticalSession:
        """
        新しい弁証法セッションを開始する。

        Parameters
        ----------
        name : str
            セッション名（後から参照するための識別子）
        goal : str
            最終的に達成したいゴール
        thesis : str
            初期のテーゼ（主張・計画・仮説）
        """
        session = DialecticalSession(
            goal=goal,
            created_at=time.time(),
        )
        triad = DialecticalTriad(
            round=1,
            thesis=thesis,
            thesis_timestamp=time.time(),
        )
        session.triads.append(triad)
        self._sessions[name] = session
        self._save_session(name)
        return session

    def challenge(self, name: str, antithesis: str) -> Optional[DialecticalTriad]:
        """
        テーゼに対するアンチテーゼを設定する。

        Parameters
        ----------
        name : str
            セッション名
        antithesis : str
            テーゼへの反論・矛盾・別の視点
        """
        session = self._sessions.get(name)
        if not session or not session.current_triad:
            return None
        triad = session.current_triad
        if triad.antithesis:
            return None  # 既にアンチテーゼ設定済み
        triad.antithesis = antithesis
        triad.antithesis_timestamp = time.time()
        self._save_session(name)
        return triad

    def synthesize(self, name: str, synthesis: str) -> Optional[DialecticalTriad]:
        """
        テーゼとアンチテーゼを止揚してジンテーゼを設定する。

        Parameters
        ----------
        name : str
            セッション名
        synthesis : str
            止揚された高次の結論
        """
        session = self._sessions.get(name)
        if not session or not session.current_triad:
            return None
        triad = session.current_triad
        if not triad.antithesis or triad.synthesis:
            return None  # アンチテーゼ未設定 or 既にジンテーゼ設定済み
        triad.synthesis = synthesis
        triad.synthesis_timestamp = time.time()
        self._save_session(name)
        return triad

    def iterate(self, name: str, new_antithesis: str = "") -> Optional[DialecticalTriad]:
        """
        現在のジンテーゼを新たなテーゼとして次のラウンドを開始する。

        Parameters
        ----------
        name : str
            セッション名
        new_antithesis : str, optional
            新ラウンドのアンチテーゼ（省略時はアンチテーゼなしで開始）
        """
        session = self._sessions.get(name)
        if not session or not session.current_triad:
            return None
        prev = session.current_triad
        if not prev.is_complete:
            return None  # 前のラウンドが未完了

        new_triad = DialecticalTriad(
            round=prev.round + 1,
            thesis=prev.synthesis,  # ジンテーゼが新しいテーゼになる
            thesis_timestamp=time.time(),
        )
        if new_antithesis:
            new_triad.antithesis = new_antithesis
            new_triad.antithesis_timestamp = time.time()

        session.triads.append(new_triad)
        self._save_session(name)
        return new_triad

    def conclude(self, name: str, conclusion: str = "") -> Optional[str]:
        """
        セッションを結論づける。

        Parameters
        ----------
        name : str
            セッション名
        conclusion : str, optional
            最終結論（省略時は最新のジンテーゼを使用）
        """
        session = self._sessions.get(name)
        if not session:
            return None
        if conclusion:
            session.conclusion = conclusion
        elif session.current_triad and session.current_triad.synthesis:
            session.conclusion = session.current_triad.synthesis
        elif session.current_triad:
            session.conclusion = session.current_triad.thesis
        self._save_session(name)
        return session.conclusion

    def view(self, name: str) -> Optional[str]:
        """セッションの全体像をテキストで返す。"""
        session = self._sessions.get(name)
        if not session:
            return None

        lines = [
            f"弁証法セッション: {name}",
            f"ゴール: {session.goal}",
            f"フェーズ: {session.phase}",
            "",
        ]

        for triad in session.triads:
            lines.append(f"--- ラウンド {triad.round} ---")
            lines.append(f"テーゼ: {triad.thesis}")
            if triad.antithesis:
                lines.append(f"アンチテーゼ: {triad.antithesis}")
            if triad.synthesis:
                lines.append(f"ジンテーゼ: {triad.synthesis}")
            lines.append("")

        if session.conclusion:
            lines.append(f"=== 結論 ===")
            lines.append(session.conclusion)

        return "\n".join(lines)

    def list_sessions(self) -> List[dict]:
        """全セッションの概要を返す。"""
        return [
            {
                "name": name,
                "goal": s.goal[:50],
                "rounds": len(s.triads),
                "phase": s.phase,
            }
            for name, s in self._sessions.items()
        ]

    def delete(self, name: str) -> bool:
        """セッションを削除する。"""
        if name in self._sessions:
            del self._sessions[name]
            # Propagate deletion to DB (no-op if not persistent).
            self._save_session(name)
            return True
        return False

    def clear_all(self) -> int:
        """全セッションを削除する。"""
        count = len(self._sessions)
        self._sessions.clear()
        if self._conn is not None:
            self._conn.execute("DELETE FROM dialectic_sessions")
            self._conn.commit()
        return count

    def __len__(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        return f"DialecticalReasoner({len(self._sessions)} sessions)"
