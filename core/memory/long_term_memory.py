"""
Persistent Long-Term Memory Dual-Store.
Combines:
- SQLite FTS5 database (storage/memory.db) with memory_facts and episodic_memory tables.
- WAL journal mode and thread-safe locking (threading.RLock).
- Fact triple storage and fast lexical FTS5 retrieval with LIKE fallback.
- Episodic memory recording and querying past execution plans.
- Bi-directional Markdown vault sync with memory/ (reading user facts and atomic updates).
"""

from __future__ import annotations
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, List, Union, Dict

from core.memory.models import FactTriple, EpisodicRecord

DEFAULT_GLOBAL_MEMORY_PATH = Path("~/.rag_memory/global_memory.db").expanduser().resolve()


class LongTermMemoryManager:
    """
    Persistent Dual-Store Memory Engine.
    Coordinates SQLite FTS5 relational/lexical storage with a human-readable Markdown vault.
    Supports WAL mode, version history tracking, soft/hard deletion, TTL expiration,
    and namespace scoping across projects.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        vault_dir: Optional[Union[str, Path]] = None,
    ):
        if db_path is None or str(db_path).strip() in ("", "None", "null"):
            env_path = os.environ.get("GLOBAL_MEMORY_DB_PATH") or os.environ.get("RAG_MEMORY_DB_PATH")
            if env_path:
                self.db_path = str(Path(env_path).expanduser().resolve())
            else:
                self.db_path = str(DEFAULT_GLOBAL_MEMORY_PATH)
        elif str(db_path) == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = str(Path(db_path).expanduser().resolve())

        self.vault_dir = Path(vault_dir) if vault_dir else None
        if self.vault_dir:
            self.vault_dir.mkdir(parents=True, exist_ok=True)

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.create_function("lower", 1, lambda s: s.lower() if isinstance(s, str) else s)
        except Exception:
            pass
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema, FTS5 virtual table, performance indexes, and WAL mode."""
        with self._lock:
            cur = self._conn.cursor()
            if self.db_path != ":memory:":
                try:
                    cur.execute("PRAGMA journal_mode=WAL;")
                    cur.execute("PRAGMA synchronous=NORMAL;")
                except Exception:
                    pass
            cur.execute("PRAGMA busy_timeout=5000;")

            # 1. Facts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source_citation TEXT,
                    tags TEXT,
                    version INTEGER DEFAULT 1,
                    ttl_days INTEGER DEFAULT NULL,
                    expires_at TIMESTAMP DEFAULT NULL,
                    is_deleted INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Non-destructive schema migration for pre-existing databases
            cur.execute("PRAGMA table_info(memory_facts);")
            existing_cols = {row[1] for row in cur.fetchall()}
            if "version" not in existing_cols:
                cur.execute("ALTER TABLE memory_facts ADD COLUMN version INTEGER DEFAULT 1;")
            if "ttl_days" not in existing_cols:
                cur.execute("ALTER TABLE memory_facts ADD COLUMN ttl_days INTEGER DEFAULT NULL;")
            if "expires_at" not in existing_cols:
                cur.execute("ALTER TABLE memory_facts ADD COLUMN expires_at TIMESTAMP DEFAULT NULL;")
            if "is_deleted" not in existing_cols:
                cur.execute("ALTER TABLE memory_facts ADD COLUMN is_deleted INTEGER DEFAULT 0;")
            if "updated_at" not in existing_cols:
                cur.execute("ALTER TABLE memory_facts ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")

            # 2. History table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_facts_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_id INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    namespace TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source_citation TEXT,
                    tags TEXT,
                    ttl_days INTEGER DEFAULT NULL,
                    expires_at TIMESTAMP DEFAULT NULL,
                    created_at TIMESTAMP NOT NULL,
                    superseded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action TEXT NOT NULL DEFAULT 'replaced'
                );
            """)

            # 3. Performance Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_facts_ns ON memory_facts(namespace);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_facts_ns_sub ON memory_facts(namespace, subject);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_facts_expires ON memory_facts(expires_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_facts_created ON memory_facts(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_facts_deleted ON memory_facts(is_deleted);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_facts_conf ON memory_facts(confidence);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_history_fact_id ON memory_facts_history(fact_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_history_ns ON memory_facts_history(namespace);")

            # 4. Full-Text Search FTS5 virtual table
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_facts_fts USING fts5(
                    subject, predicate, object_value, content='memory_facts', content_rowid='id', tokenize='unicode61'
                );
            """)

            # 5. Episodic memory table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    episode_id TEXT PRIMARY KEY,
                    task_goal TEXT NOT NULL,
                    plan_executed TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    tools_used TEXT,
                    lessons_learned TEXT,
                    error_signature TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_episodic_outcome ON episodic_memory(outcome);")
            self._conn.commit()

    def _row_to_fact(self, row: sqlite3.Row) -> FactTriple:
        """Convert a SQLite row from memory_facts into a FactTriple model."""
        tags_list: List[str] = []
        if row["tags"]:
            try:
                tags_list = json.loads(row["tags"])
            except Exception:
                tags_list = []

        keys = row.keys() if hasattr(row, "keys") else []
        version = row["version"] if "version" in keys and row["version"] is not None else 1
        ttl_days = row["ttl_days"] if "ttl_days" in keys else None
        expires_at = row["expires_at"] if "expires_at" in keys else None
        is_deleted = bool(row["is_deleted"]) if "is_deleted" in keys and row["is_deleted"] is not None else False
        created_at = row["created_at"] if "created_at" in keys and row["created_at"] is not None else time.time()

        return FactTriple(
            id=row["id"],
            namespace=row["namespace"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_value=row["object_value"],
            confidence=float(row["confidence"]) if row["confidence"] is not None else 1.0,
            source_citation=row["source_citation"],
            tags=tags_list,
            version=int(version),
            ttl_days=int(ttl_days) if ttl_days is not None else None,
            expires_at=str(expires_at) if expires_at is not None else None,
            is_deleted=is_deleted,
            created_at=created_at,
        )

    # ==========================================================================
    # FACT TRIPLE STORE & FTS5 RETRIEVAL
    # ==========================================================================

    def store_fact(self, fact: FactTriple, sync_vault: bool = True) -> int:
        """
        Store a discrete fact triple into SQLite and FTS5 index.
        Syncs atomically to the Markdown vault if configured.
        """
        tags_val = getattr(fact, "tags", None)
        tags_json = json.dumps(tags_val, ensure_ascii=False) if tags_val else ""

        # Compute expires_at if ttl_days is specified
        expires_at_str = None
        if fact.ttl_days is not None and fact.ttl_days > 0:
            if fact.expires_at:
                if isinstance(fact.expires_at, datetime):
                    expires_at_str = fact.expires_at.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    expires_at_str = str(fact.expires_at)
            else:
                now = datetime.now(timezone.utc)
                expires_at_str = (now + timedelta(days=fact.ttl_days)).strftime("%Y-%m-%d %H:%M:%S")
        elif fact.expires_at:
            if isinstance(fact.expires_at, datetime):
                expires_at_str = fact.expires_at.strftime("%Y-%m-%d %H:%M:%S")
            else:
                expires_at_str = str(fact.expires_at)

        fact.expires_at = expires_at_str

        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                INSERT INTO memory_facts (
                    namespace, subject, predicate, object_value, confidence,
                    source_citation, tags, version, ttl_days, expires_at, is_deleted
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.namespace,
                fact.subject,
                fact.predicate,
                fact.object_value,
                fact.confidence,
                fact.source_citation,
                tags_json,
                fact.version or 1,
                fact.ttl_days,
                expires_at_str,
                1 if fact.is_deleted else 0,
            ))
            fact_id = cur.lastrowid
            fact.id = fact_id

            # Insert into FTS5 only if active (not soft-deleted)
            if not fact.is_deleted:
                cur.execute("""
                    INSERT INTO memory_facts_fts(rowid, subject, predicate, object_value)
                    VALUES (?, ?, ?, ?)
                """, (fact_id, fact.subject, fact.predicate, fact.object_value))
            self._conn.commit()

        if sync_vault and self.vault_dir:
            self._sync_vault(fact)

        return fact_id

    def replace_fact(
        self,
        old_fact_id_or_key: Union[int, str, dict],
        new_fact: FactTriple,
        sync_vault: bool = True,
    ) -> int:
        """
        Replace an active fact with a new version, preserving full history.
        - Lookup by integer ID, numeric string, or dict {'namespace': ..., 'subject': ..., 'predicate': ...}.
        - Moves current row snapshot to memory_facts_history with incremented version and superseded_at.
        - Updates active row in memory_facts and synchronizes FTS5 index.
        - Returns the active fact ID.
        """
        with self._lock:
            cur = self._conn.cursor()
            old_row = None

            if isinstance(old_fact_id_or_key, int) or (isinstance(old_fact_id_or_key, str) and str(old_fact_id_or_key).isdigit()):
                cur.execute("SELECT * FROM memory_facts WHERE id = ?", (int(old_fact_id_or_key),))
                old_row = cur.fetchone()
            elif isinstance(old_fact_id_or_key, dict):
                cur.execute("""
                    SELECT * FROM memory_facts
                    WHERE namespace = ? AND subject = ? AND predicate = ? AND is_deleted = 0
                    ORDER BY id DESC LIMIT 1
                """, (
                    old_fact_id_or_key.get("namespace"),
                    old_fact_id_or_key.get("subject"),
                    old_fact_id_or_key.get("predicate"),
                ))
                old_row = cur.fetchone()
            else:
                raise ValueError(f"Invalid fact lookup identifier: {old_fact_id_or_key}")

            if not old_row:
                raise KeyError(f"Fact matching {old_fact_id_or_key} not found.")

            if old_row["is_deleted"] == 1:
                raise KeyError(f"Active fact matching {old_fact_id_or_key} is deleted.")

            fact_id = old_row["id"]
            old_version = old_row["version"] if old_row["version"] is not None else 1
            new_version = old_version + 1

            # 1. Archive previous state into memory_facts_history
            cur.execute("""
                INSERT INTO memory_facts_history (
                    fact_id, version, namespace, subject, predicate, object_value,
                    confidence, source_citation, tags, ttl_days, expires_at,
                    created_at, superseded_at, action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'replaced')
            """, (
                fact_id,
                old_version,
                old_row["namespace"],
                old_row["subject"],
                old_row["predicate"],
                old_row["object_value"],
                old_row["confidence"],
                old_row["source_citation"],
                old_row["tags"],
                old_row["ttl_days"],
                old_row["expires_at"],
                old_row["created_at"],
            ))

            # 2. Compute expiration for new_fact
            expires_at_str = None
            if new_fact.ttl_days is not None and new_fact.ttl_days > 0:
                if new_fact.expires_at:
                    if isinstance(new_fact.expires_at, datetime):
                        expires_at_str = new_fact.expires_at.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        expires_at_str = str(new_fact.expires_at)
                else:
                    now = datetime.now(timezone.utc)
                    expires_at_str = (now + timedelta(days=new_fact.ttl_days)).strftime("%Y-%m-%d %H:%M:%S")
            elif new_fact.expires_at:
                if isinstance(new_fact.expires_at, datetime):
                    expires_at_str = new_fact.expires_at.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    expires_at_str = str(new_fact.expires_at)

            tags_val = getattr(new_fact, "tags", None)
            tags_json = json.dumps(tags_val, ensure_ascii=False) if tags_val else ""

            # 3. Update active row in memory_facts
            cur.execute("""
                UPDATE memory_facts
                SET namespace = ?, subject = ?, predicate = ?, object_value = ?,
                    confidence = ?, source_citation = ?, tags = ?, version = ?,
                    ttl_days = ?, expires_at = ?, is_deleted = 0, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                new_fact.namespace,
                new_fact.subject,
                new_fact.predicate,
                new_fact.object_value,
                new_fact.confidence,
                new_fact.source_citation,
                tags_json,
                new_version,
                new_fact.ttl_days,
                expires_at_str,
                fact_id,
            ))

            # 4. Synchronize FTS5 index: delete old tokens, insert new tokens
            cur.execute("""
                INSERT INTO memory_facts_fts(memory_facts_fts, rowid, subject, predicate, object_value)
                VALUES('delete', ?, ?, ?, ?)
            """, (fact_id, old_row["subject"], old_row["predicate"], old_row["object_value"]))

            cur.execute("""
                INSERT INTO memory_facts_fts(rowid, subject, predicate, object_value)
                VALUES(?, ?, ?, ?)
            """, (fact_id, new_fact.subject, new_fact.predicate, new_fact.object_value))

            self._conn.commit()

            new_fact.id = fact_id
            new_fact.version = new_version
            new_fact.expires_at = expires_at_str

        if sync_vault and self.vault_dir:
            self._sync_vault(new_fact)

        return fact_id

    def delete_fact(self, fact_id: int, soft: bool = True) -> bool:
        """
        Delete fact from SQLite and FTS5 index.
        - soft=True: Marks is_deleted=1, records history entry with action='deleted', prunes from FTS5.
        - soft=False: Hard deletes row completely from memory_facts and FTS5.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM memory_facts WHERE id = ?", (fact_id,))
            row = cur.fetchone()
            if not row:
                return False

            if soft:
                if row["is_deleted"] == 1:
                    return False  # Already soft-deleted

                # Record deletion in history
                cur.execute("""
                    INSERT INTO memory_facts_history (
                        fact_id, version, namespace, subject, predicate, object_value,
                        confidence, source_citation, tags, ttl_days, expires_at,
                        created_at, superseded_at, action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'deleted')
                """, (
                    fact_id,
                    row["version"] if row["version"] is not None else 1,
                    row["namespace"],
                    row["subject"],
                    row["predicate"],
                    row["object_value"],
                    row["confidence"],
                    row["source_citation"],
                    row["tags"],
                    row["ttl_days"],
                    row["expires_at"],
                    row["created_at"],
                ))

                # Mark as deleted in relational table
                cur.execute("UPDATE memory_facts SET is_deleted = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (fact_id,))

                # Prune from FTS5 index
                cur.execute("""
                    INSERT INTO memory_facts_fts(memory_facts_fts, rowid, subject, predicate, object_value)
                    VALUES('delete', ?, ?, ?, ?)
                """, (fact_id, row["subject"], row["predicate"], row["object_value"]))

                self._conn.commit()
                return True
            else:
                # Hard delete
                if row["is_deleted"] == 0:
                    cur.execute("""
                        INSERT INTO memory_facts_fts(memory_facts_fts, rowid, subject, predicate, object_value)
                        VALUES('delete', ?, ?, ?, ?)
                    """, (fact_id, row["subject"], row["predicate"], row["object_value"]))
                cur.execute("DELETE FROM memory_facts WHERE id = ?", (fact_id,))
                self._conn.commit()
                return True

    def get_fact(self, fact_id: int, include_deleted: bool = False) -> Optional[FactTriple]:
        """Fetch a specific fact by ID."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM memory_facts WHERE id = ?", (fact_id,))
            row = cur.fetchone()
            if not row:
                return None
            if not include_deleted and row["is_deleted"] == 1:
                return None
            return self._row_to_fact(row)

    def get_fact_history(self, fact_id: int) -> List[Dict[str, Any]]:
        """Retrieve full audit history trail of a fact across replacements and deletions."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                SELECT history_id, fact_id, version, namespace, subject, predicate, object_value,
                       confidence, source_citation, tags, ttl_days, expires_at, created_at,
                       superseded_at, action
                FROM memory_facts_history
                WHERE fact_id = ?
                ORDER BY version ASC, history_id ASC
            """, (fact_id,))
            rows = cur.fetchall()
            history: List[Dict[str, Any]] = []
            for r in rows:
                tags_list = []
                if r["tags"]:
                    try:
                        tags_list = json.loads(r["tags"])
                    except Exception:
                        tags_list = []
                history.append({
                    "history_id": r["history_id"],
                    "fact_id": r["fact_id"],
                    "version": r["version"],
                    "namespace": r["namespace"],
                    "subject": r["subject"],
                    "predicate": r["predicate"],
                    "object_value": r["object_value"],
                    "confidence": float(r["confidence"]) if r["confidence"] is not None else 1.0,
                    "source_citation": r["source_citation"],
                    "tags": tags_list,
                    "ttl_days": r["ttl_days"],
                    "expires_at": r["expires_at"],
                    "created_at": r["created_at"],
                    "superseded_at": r["superseded_at"],
                    "action": r["action"],
                })
            return history

    def search_facts(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 20,
        include_expired: bool = False,
        include_deleted: bool = False,
    ) -> List[FactTriple]:
        """
        Search fact triples using FTS5 lexical match with LIKE fallback.
        Supports optional namespace scoping, soft-deletion exclusion, and TTL expiration filtering.
        """
        raw_q = query.strip()
        if not raw_q:
            return []

        # Extract alphanumeric tokens, filtering out lonely FTS boolean operators
        tokens = [t for t in re.findall(r'[\w]+', raw_q) if len(t) > 1 and t.upper() not in ("OR", "AND", "NOT")]

        with self._lock:
            cur = self._conn.cursor()
            rows: List[Any] = []
            fts_succeeded = False

            # 1. Attempt FTS5 query
            if tokens:
                try:
                    match_clause = " OR ".join(f'"{t}"*' for t in tokens)
                    clauses = ["memory_facts_fts MATCH ?"]
                    params: List[Any] = [match_clause]

                    if not include_deleted:
                        clauses.append("f.is_deleted = 0")
                    if not include_expired:
                        clauses.append("(f.expires_at IS NULL OR datetime(f.expires_at) > CURRENT_TIMESTAMP)")
                    if namespace:
                        clauses.append("f.namespace = ?")
                        params.append(namespace)

                    where_sql = " AND ".join(clauses)
                    cur.execute(f"""
                        SELECT f.*
                        FROM memory_facts_fts
                        JOIN memory_facts f ON memory_facts_fts.rowid = f.id
                        WHERE {where_sql}
                        ORDER BY rank
                        LIMIT ?
                    """, (*params, limit))
                    rows = cur.fetchall()
                    fts_succeeded = True
                except sqlite3.OperationalError:
                    fts_succeeded = False

            # 2. Robust fallback to LIKE if FTS fails or yields no matches
            if not fts_succeeded or len(rows) == 0:
                search_terms = tokens if tokens else [raw_q]
                like_clauses = []
                like_params: List[Any] = []
                for t in search_terms[:5]:
                    p = f"%{t}%"
                    like_clauses.append("(LOWER(subject) LIKE LOWER(?) OR LOWER(predicate) LIKE LOWER(?) OR LOWER(object_value) LIKE LOWER(?))")
                    like_params.extend([p, p, p])

                where_clauses = []
                params = []
                if like_clauses:
                    where_clauses.append(f"({' OR '.join(like_clauses)})")
                    params.extend(like_params)

                if not include_deleted:
                    where_clauses.append("is_deleted = 0")
                if not include_expired:
                    where_clauses.append("(expires_at IS NULL OR datetime(expires_at) > CURRENT_TIMESTAMP)")
                if namespace:
                    where_clauses.append("namespace = ?")
                    params.append(namespace)

                if where_clauses:
                    where_sql = " AND ".join(where_clauses)
                    cur.execute(f"""
                        SELECT *
                        FROM memory_facts
                        WHERE {where_sql}
                        ORDER BY id DESC
                        LIMIT ?
                    """, (*params, limit))
                    rows = cur.fetchall()

            return [self._row_to_fact(r) for r in rows]

    def recall_facts(
        self,
        query: Optional[str] = None,
        namespace: Optional[str] = None,
        min_confidence: float = 0.0,
        include_expired: bool = False,
        include_deleted: bool = False,
        sort_by: str = "relevance",  # "relevance" | "confidence" | "recency"
        limit: int = 20,
    ) -> List[FactTriple]:
        """
        Unified recall method supporting full queries or relational filtering,
        namespace isolation, TTL expiration, confidence thresholds, and flexible ordering.
        """
        raw_q = query.strip() if query else ""
        if raw_q:
            tokens = [t for t in re.findall(r'[\w]+', raw_q) if len(t) > 1 and t.upper() not in ("OR", "AND", "NOT")]
            with self._lock:
                cur = self._conn.cursor()
                rows: List[Any] = []
                fts_succeeded = False

                if tokens:
                    try:
                        match_clause = " OR ".join(f'"{t}"*' for t in tokens)
                        clauses = ["memory_facts_fts MATCH ?"]
                        params: List[Any] = [match_clause]

                        if not include_deleted:
                            clauses.append("f.is_deleted = 0")
                        if not include_expired:
                            clauses.append("(f.expires_at IS NULL OR datetime(f.expires_at) > CURRENT_TIMESTAMP)")
                        if min_confidence > 0.0:
                            clauses.append("f.confidence >= ?")
                            params.append(min_confidence)
                        if namespace:
                            clauses.append("f.namespace = ?")
                            params.append(namespace)

                        order_sql = "rank ASC, f.confidence DESC"
                        if sort_by == "confidence":
                            order_sql = "f.confidence DESC, rank ASC"
                        elif sort_by == "recency":
                            order_sql = "f.id DESC"

                        where_sql = " AND ".join(clauses)
                        cur.execute(f"""
                            SELECT f.*
                            FROM memory_facts_fts
                            JOIN memory_facts f ON memory_facts_fts.rowid = f.id
                            WHERE {where_sql}
                            ORDER BY {order_sql}
                            LIMIT ?
                        """, (*params, limit))
                        rows = cur.fetchall()
                        fts_succeeded = True
                    except sqlite3.OperationalError:
                        fts_succeeded = False

                if not fts_succeeded or len(rows) == 0:
                    search_terms = tokens if tokens else [raw_q]
                    like_clauses = []
                    like_params: List[Any] = []
                    for t in search_terms[:5]:
                        p = f"%{t}%"
                        like_clauses.append("(LOWER(subject) LIKE LOWER(?) OR LOWER(predicate) LIKE LOWER(?) OR LOWER(object_value) LIKE LOWER(?))")
                        like_params.extend([p, p, p])

                    where_clauses = []
                    params = []
                    if like_clauses:
                        where_clauses.append(f"({' OR '.join(like_clauses)})")
                        params.extend(like_params)

                    if not include_deleted:
                        where_clauses.append("is_deleted = 0")
                    if not include_expired:
                        where_clauses.append("(expires_at IS NULL OR datetime(expires_at) > CURRENT_TIMESTAMP)")
                    if min_confidence > 0.0:
                        where_clauses.append("confidence >= ?")
                        params.append(min_confidence)
                    if namespace:
                        where_clauses.append("namespace = ?")
                        params.append(namespace)

                    order_sql = "id DESC"
                    if sort_by == "confidence":
                        order_sql = "confidence DESC, id DESC"

                    where_sql = " AND ".join(where_clauses)
                    cur.execute(f"""
                        SELECT *
                        FROM memory_facts
                        WHERE {where_sql}
                        ORDER BY {order_sql}
                        LIMIT ?
                    """, (*params, limit))
                    rows = cur.fetchall()

                return [self._row_to_fact(r) for r in rows]

        # No query string provided: pure relational filtering
        with self._lock:
            cur = self._conn.cursor()
            where_clauses = []
            params = []

            if not include_deleted:
                where_clauses.append("is_deleted = 0")
            if not include_expired:
                where_clauses.append("(expires_at IS NULL OR datetime(expires_at) > CURRENT_TIMESTAMP)")
            if min_confidence > 0.0:
                where_clauses.append("confidence >= ?")
                params.append(min_confidence)
            if namespace:
                where_clauses.append("namespace = ?")
                params.append(namespace)

            order_sql = "id DESC"
            if sort_by == "confidence":
                order_sql = "confidence DESC, id DESC"
            elif sort_by == "recency":
                order_sql = "id DESC"

            where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            cur.execute(f"""
                SELECT *
                FROM memory_facts
                {where_sql}
                ORDER BY {order_sql}
                LIMIT ?
            """, (*params, limit))
            rows = cur.fetchall()
            return [self._row_to_fact(r) for r in rows]

    def list_facts(
        self,
        namespace: Optional[str] = None,
        limit: int = 100,
        include_deleted: bool = False,
        include_expired: bool = False,
    ) -> List[FactTriple]:
        """List stored facts in chronological order, filtering deleted and expired by default."""
        with self._lock:
            cur = self._conn.cursor()
            where_clauses = []
            params: List[Any] = []

            if not include_deleted:
                where_clauses.append("is_deleted = 0")
            if not include_expired:
                where_clauses.append("(expires_at IS NULL OR datetime(expires_at) > CURRENT_TIMESTAMP)")
            if namespace:
                where_clauses.append("namespace = ?")
                params.append(namespace)

            where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            cur.execute(f"""
                SELECT *
                FROM memory_facts
                {where_sql}
                ORDER BY id ASC
                LIMIT ?
            """, (*params, limit))
            rows = cur.fetchall()
            return [self._row_to_fact(r) for r in rows]

    # ==========================================================================
    # EPISODIC MEMORY
    # ==========================================================================

    def record_episode(self, episode: EpisodicRecord) -> None:
        """Persist or update an agent task execution episode."""
        plan_json = json.dumps(episode.plan_executed, ensure_ascii=False)
        tools_json = json.dumps(episode.tools_used, ensure_ascii=False)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO episodic_memory (
                    episode_id, task_goal, plan_executed, outcome, tools_used, lessons_learned, error_signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.episode_id,
                episode.task_goal,
                plan_json,
                episode.outcome,
                tools_json,
                episode.lessons_learned,
                episode.error_signature,
            ))
            self._conn.commit()

    def get_episode(self, episode_id: str) -> Optional[EpisodicRecord]:
        """Retrieve an episode by ID."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                SELECT episode_id, task_goal, plan_executed, outcome, tools_used, lessons_learned, error_signature, created_at
                FROM episodic_memory WHERE episode_id = ?
            """, (episode_id,))
            row = cur.fetchone()
            if not row:
                return None
            try:
                plan = json.loads(row["plan_executed"])
            except Exception:
                plan = []
            try:
                tools = json.loads(row["tools_used"]) if row["tools_used"] else []
            except Exception:
                tools = []
            return EpisodicRecord(
                episode_id=row["episode_id"],
                task_goal=row["task_goal"],
                plan_executed=plan,
                outcome=row["outcome"],
                tools_used=tools,
                lessons_learned=row["lessons_learned"],
                error_signature=row["error_signature"],
                timestamp=row["created_at"],
            )

    def search_episodes(
        self, query: str, outcome: Optional[str] = None, limit: int = 10
    ) -> List[EpisodicRecord]:
        """Search past execution episodes by task goal, lessons learned, error signature, tools used, or plan executed."""
        pattern = f"%{query.strip()}%"
        with self._lock:
            cur = self._conn.cursor()
            if outcome:
                cur.execute("""
                    SELECT episode_id, task_goal, plan_executed, outcome, tools_used, lessons_learned, error_signature, created_at
                    FROM episodic_memory
                    WHERE outcome = ? AND (
                        task_goal LIKE ? OR
                        lessons_learned LIKE ? OR
                        error_signature LIKE ? OR
                        tools_used LIKE ? OR
                        plan_executed LIKE ?
                    )
                    ORDER BY created_at DESC LIMIT ?
                """, (outcome, pattern, pattern, pattern, pattern, pattern, limit))
            else:
                cur.execute("""
                    SELECT episode_id, task_goal, plan_executed, outcome, tools_used, lessons_learned, error_signature, created_at
                    FROM episodic_memory
                    WHERE (
                        task_goal LIKE ? OR
                        lessons_learned LIKE ? OR
                        error_signature LIKE ? OR
                        tools_used LIKE ? OR
                        plan_executed LIKE ?
                    )
                    ORDER BY created_at DESC LIMIT ?
                """, (pattern, pattern, pattern, pattern, pattern, limit))
            rows = cur.fetchall()
            episodes = []
            for r in rows:
                try:
                    plan = json.loads(r["plan_executed"])
                except Exception:
                    plan = []
                try:
                    tools = json.loads(r["tools_used"]) if r["tools_used"] else []
                except Exception:
                    tools = []
                episodes.append(
                    EpisodicRecord(
                        episode_id=r["episode_id"],
                        task_goal=r["task_goal"],
                        plan_executed=plan,
                        outcome=r["outcome"],
                        tools_used=tools,
                        lessons_learned=r["lessons_learned"],
                        error_signature=r["error_signature"],
                        timestamp=r["created_at"],
                    )
                )
            return episodes

    # ==========================================================================
    # BI-DIRECTIONAL MARKDOWN VAULT SYNC
    # ==========================================================================

    def _sync_vault(self, fact: FactTriple) -> None:
        """
        Atomically append or update a fact in the Markdown vault.
        Matches format: - **Subject** predicate `object_value` (conf: 1.0)
        """
        if not self.vault_dir:
            return

        with self._lock:
            target_file = self.vault_dir / f"{fact.namespace}.md"
            line_to_write = f"- **{fact.subject}** {fact.predicate} `{fact.object_value}` (conf: {fact.confidence})\n"

            target_file.parent.mkdir(parents=True, exist_ok=True)
            # Check if already written
            existing_text = ""
            if target_file.exists():
                try:
                    existing_text = target_file.read_text(encoding="utf-8")
                except Exception:
                    existing_text = ""

            # Avoid exact duplicate lines
            if line_to_write.strip() in existing_text:
                return

            # Atomic append
            temp_file = target_file.with_suffix(f".tmp_{uuid.uuid4().hex}")
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    if existing_text:
                        f.write(existing_text)
                        if not existing_text.endswith("\n"):
                            f.write("\n")
                    f.write(line_to_write)
                os.replace(str(temp_file), str(target_file))
            except Exception:
                # Fallback direct append if replace fails
                try:
                    with open(target_file, "a", encoding="utf-8") as f:
                        f.write(line_to_write)
                except Exception:
                    pass
            finally:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass

    def append_decision(self, project: str, decision: str, rationale: str = "") -> Path:
        """
        Append an architectural decision atomically to the markdown vault and SQLite facts.
        """
        if not self.vault_dir:
            raise ValueError("Vault directory is not configured.")

        with self._lock:
            project_dir = self.vault_dir / project
            project_dir.mkdir(parents=True, exist_ok=True)
            decisions_file = project_dir / "decisions.md"

            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            entry = f"\n### Decision: {decision}\n- **Date**: {timestamp_str}\n- **Rationale**: {rationale}\n"

            existing = ""
            if decisions_file.exists():
                try:
                    existing = decisions_file.read_text(encoding="utf-8")
                except Exception:
                    pass

            temp_file = decisions_file.with_suffix(f".tmp_{uuid.uuid4().hex}")
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    if existing:
                        f.write(existing)
                    else:
                        f.write(f"# Architectural Decisions: {project}\n")
                    f.write(entry)
                os.replace(str(temp_file), str(decisions_file))
            except Exception:
                try:
                    with open(decisions_file, "a", encoding="utf-8") as f:
                        if not existing:
                            f.write(f"# Architectural Decisions: {project}\n")
                        f.write(entry)
                except Exception:
                    pass
            finally:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass

            # Also store into SQLite memory_facts
            self.store_fact(
                FactTriple(
                    namespace=project,
                    subject=decision,
                    predicate="decision",
                    object_value=rationale or "accepted",
                    confidence=1.0,
                    source_citation=str(decisions_file),
                ),
                sync_vault=False,
            )
            return decisions_file

    def sync_from_vault(self, vault_dir: Optional[Union[str, Path]] = None) -> int:
        """
        Ingest facts from markdown files in vault_dir into SQLite FTS5 store.
        Parses bullet points like `- **Subject** predicate `object`` or `Key: Value`.
        """
        source_dir = Path(vault_dir) if vault_dir else self.vault_dir
        if not source_dir or not source_dir.exists():
            return 0

        synced_count = 0
        md_pattern = re.compile(r"-\s+\*\*(.+?)\*\*\s+([^\`]+?)\s+\`(.+?)\`(?:\s+\(conf:\s*([0-9\.]+)\))?")
        colon_pattern = re.compile(r"^[-*]?\s*([^:\n]{2,30}):\s*(.+)$")

        for md_file in source_dir.glob("**/*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            namespace = md_file.stem.lower()
            if namespace in ("_root", "index"):
                namespace = "user"

            for line in content.splitlines():
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue

                try:
                    # Pattern 1: - **Subject** predicate `value`
                    m1 = md_pattern.match(line_str)
                    if m1:
                        sub, pred, val = m1.group(1).strip(), m1.group(2).strip(), m1.group(3).strip()
                        conf = 1.0
                        if m1.group(4):
                            try:
                                conf = max(0.0, min(1.0, float(m1.group(4))))
                            except (ValueError, TypeError):
                                conf = 1.0

                        # Check if fact already exists before inserting
                        with self._lock:
                            cur = self._conn.cursor()
                            cur.execute("""
                                SELECT id FROM memory_facts
                                WHERE namespace = ? AND subject = ? AND predicate = ? AND object_value = ?
                                LIMIT 1
                            """, (namespace, sub, pred, val))
                            if cur.fetchone() is not None:
                                continue

                        self.store_fact(
                            FactTriple(
                                namespace=namespace,
                                subject=sub,
                                predicate=pred,
                                object_value=val,
                                confidence=conf,
                                source_citation=str(md_file),
                            ),
                            sync_vault=False,
                        )
                        synced_count += 1
                        continue

                    # Pattern 2: Key: Value (e.g. "Формат работы: СТРОГО УДАЛЁНКА")
                    m2 = colon_pattern.match(line_str)
                    if m2:
                        sub = m2.group(1).strip()
                        val = m2.group(2).strip()
                        if val and len(sub) > 1 and not sub.startswith("http"):
                            # Check if fact already exists before inserting
                            with self._lock:
                                cur = self._conn.cursor()
                                cur.execute("""
                                    SELECT id FROM memory_facts
                                    WHERE namespace = ? AND subject = ? AND predicate = ? AND object_value = ?
                                    LIMIT 1
                                """, (namespace, sub, "значение", val))
                                if cur.fetchone() is not None:
                                    continue

                            self.store_fact(
                                FactTriple(
                                    namespace=namespace,
                                    subject=sub,
                                    predicate="значение",
                                    object_value=val,
                                    confidence=1.0,
                                    source_citation=str(md_file),
                                ),
                                sync_vault=False,
                            )
                            synced_count += 1
                except Exception:
                    continue

        return synced_count

    def close(self) -> None:
        """Close SQLite database connection cleanly."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> LongTermMemoryManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
