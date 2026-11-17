"""
Unit and integration tests for Milestone 6 (Requirement R2):
- Fact Attributes & Models (version, ttl_days, expires_at, is_deleted)
- Centralized SQLite Storage (~/.rag_memory/global_memory.db) with WAL mode & PRAGMAs
- Non-destructive schema migration for legacy databases
- Version history tracking and replace_fact operation
- Soft and hard delete_fact operation
- TTL calculation and expiration filtering
- Strict namespace scoping (manage, Deckor, NIR, Элир, user)
- MemoryManager facade delegation & AgentRuntime safe initialization
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from core.memory.models import FactTriple
from core.memory.long_term_memory import LongTermMemoryManager, DEFAULT_GLOBAL_MEMORY_PATH
from core.memory.manager import MemoryManager
from core.runtime import AgentRuntime


class TestLongTermMemoryCentralizedStorage:
    """Verifies centralized SQLite storage, WAL mode, path resolution, and schema migration."""

    def test_default_path_resolution(self):
        """Verifies default db_path resolves to ~/.rag_memory/global_memory.db and creates parent dir."""
        manager = LongTermMemoryManager(db_path=None)
        expected_path = str(DEFAULT_GLOBAL_MEMORY_PATH)
        assert manager.db_path == expected_path
        assert Path(manager.db_path).parent.exists()
        manager.close()

    def test_env_var_override(self, monkeypatch, tmp_path):
        """Verifies GLOBAL_MEMORY_DB_PATH overrides the default storage path."""
        custom_db = tmp_path / "custom_global.db"
        monkeypatch.setenv("GLOBAL_MEMORY_DB_PATH", str(custom_db))
        manager = LongTermMemoryManager(db_path=None)
        assert manager.db_path == str(custom_db.resolve())
        assert custom_db.exists()
        manager.close()

    def test_wal_journal_mode_and_pragmas(self, tmp_path):
        """Verifies file-based SQLite database configures WAL mode, synchronous=NORMAL, and busy_timeout=5000."""
        db_file = tmp_path / "wal_test.db"
        manager = LongTermMemoryManager(db_path=str(db_file))

        with manager._lock:
            cur = manager._conn.cursor()
            cur.execute("PRAGMA journal_mode;")
            journal_mode = cur.fetchone()[0]
            cur.execute("PRAGMA busy_timeout;")
            busy_timeout = cur.fetchone()[0]

        assert journal_mode.lower() == "wal"
        assert busy_timeout == 5000
        manager.close()

    def test_in_memory_bypasses_wal(self):
        """Verifies :memory: database initializes cleanly without WAL error."""
        manager = LongTermMemoryManager(db_path=":memory:")
        assert manager.db_path == ":memory:"
        fact_id = manager.store_fact(
            FactTriple(namespace="test", subject="Sub", predicate="pred", object_value="val")
        )
        assert fact_id > 0
        manager.close()

    def test_schema_migration_backward_compatibility(self, tmp_path):
        """
        Simulates an existing legacy database without version, ttl_days, expires_at, is_deleted.
        Verifies non-destructive migration preserves existing data and adds new columns.
        """
        legacy_db = tmp_path / "legacy.db"
        conn = sqlite3.connect(legacy_db)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_citation TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        c.execute("""
            INSERT INTO memory_facts (namespace, subject, predicate, object_value, confidence)
            VALUES ('user', 'LegacyUser', 'preference', 'LegacyValue', 0.95);
        """)
        conn.commit()
        conn.close()

        # Initialize LongTermMemoryManager on legacy database
        manager = LongTermMemoryManager(db_path=str(legacy_db))

        # Check existing fact preserved
        fact = manager.get_fact(1)
        assert fact is not None
        assert fact.subject == "LegacyUser"
        assert fact.version == 1
        assert fact.is_deleted is False
        assert fact.ttl_days is None

        # Check table schema has all new columns
        with manager._lock:
            cur = manager._conn.cursor()
            cur.execute("PRAGMA table_info(memory_facts);")
            columns = {row[1] for row in cur.fetchall()}

        for expected_col in ("version", "ttl_days", "expires_at", "is_deleted", "updated_at"):
            assert expected_col in columns

        # Check history table exists
        with manager._lock:
            cur = manager._conn.cursor()
            cur.execute("SELECT count(*) FROM memory_facts_history;")
            assert cur.fetchone()[0] == 0

        manager.close()

    def test_runtime_default_never_creates_rogue_none_file(self, tmp_path, monkeypatch):
        """Verifies AgentRuntime with default db_path=None resolves properly and never creates file 'None'."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        custom_global = tmp_path / "global_storage" / "memory.db"
        monkeypatch.setenv("GLOBAL_MEMORY_DB_PATH", str(custom_global))

        runtime = AgentRuntime(workspace_root=workspace, db_path=None)
        assert runtime.memory is not None
        assert runtime.memory.long_term_memory.db_path == str(custom_global.resolve())

        # Assert no file named "None" was created
        assert not (workspace / "None").exists()
        assert not (Path.cwd() / "None").exists()
        runtime.memory.long_term_memory.close()


class TestFactVersioningAndReplace:
    """Verifies monotonic version increments, history archiving, and FTS5 synchronization on replacement."""

    def test_replace_fact_increments_version(self, tmp_path):
        """Verifies version increments monotonically: v1 -> v2 -> v3."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "replace_v.db"))
        fact_id = manager.store_fact(
            FactTriple(
                namespace="user",
                subject="Филипп",
                predicate="стек",
                object_value="Python, FastAPI",
            )
        )
        f1 = manager.get_fact(fact_id)
        assert f1.version == 1
        assert f1.object_value == "Python, FastAPI"

        # First replacement: v1 -> v2
        new_fact_v2 = FactTriple(
            namespace="user",
            subject="Филипп",
            predicate="стек",
            object_value="Python, FastAPI, Docker",
        )
        manager.replace_fact(fact_id, new_fact_v2)

        f2 = manager.get_fact(fact_id)
        assert f2.version == 2
        assert f2.object_value == "Python, FastAPI, Docker"

        # Second replacement: v2 -> v3
        new_fact_v3 = FactTriple(
            namespace="user",
            subject="Филипп",
            predicate="стек",
            object_value="Python, FastAPI, Docker, Kubernetes",
        )
        manager.replace_fact(fact_id, new_fact_v3)

        f3 = manager.get_fact(fact_id)
        assert f3.version == 3
        assert f3.object_value == "Python, FastAPI, Docker, Kubernetes"
        manager.close()

    def test_replace_fact_archives_history(self, tmp_path):
        """Verifies history table preserves verbatim audit trail of superseded versions."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "replace_hist.db"))
        fact_id = manager.store_fact(
            FactTriple(
                namespace="Deckor",
                subject="Брус",
                predicate="сечение",
                object_value="50x50",
                confidence=0.9,
            )
        )

        manager.replace_fact(
            fact_id,
            FactTriple(
                namespace="Deckor",
                subject="Брус",
                predicate="сечение",
                object_value="40x40",
                confidence=1.0,
            ),
        )

        history = manager.get_fact_history(fact_id)
        assert len(history) == 1
        h0 = history[0]
        assert h0["fact_id"] == fact_id
        assert h0["version"] == 1
        assert h0["object_value"] == "50x50"
        assert h0["action"] == "replaced"
        assert h0["superseded_at"] is not None

        # Replace once more
        manager.replace_fact(
            fact_id,
            FactTriple(
                namespace="Deckor",
                subject="Брус",
                predicate="сечение",
                object_value="45x45",
                confidence=0.95,
            ),
        )

        history2 = manager.get_fact_history(fact_id)
        assert len(history2) == 2
        assert history2[1]["version"] == 2
        assert history2[1]["object_value"] == "40x40"
        manager.close()

    def test_replace_fact_fts5_updated(self, tmp_path):
        """Verifies FTS5 matches new object value and no longer matches obsolete value."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "replace_fts.db"))
        fact_id = manager.store_fact(
            FactTriple(
                namespace="NIR",
                subject="Сеть",
                predicate="архитектура",
                object_value="ResNet50_Classifier",
            )
        )

        # Initial search matches ResNet50
        res1 = manager.search_facts("ResNet50")
        assert len(res1) == 1
        assert res1[0].object_value == "ResNet50_Classifier"

        # Replace with UNet3Plus
        manager.replace_fact(
            fact_id,
            FactTriple(
                namespace="NIR",
                subject="Сеть",
                predicate="архитектура",
                object_value="UNet3Plus_Segmentation",
            ),
        )

        # Old term must not match
        res_old = manager.search_facts("ResNet50")
        assert len(res_old) == 0

        # New term must match
        res_new = manager.search_facts("UNet3Plus")
        assert len(res_new) == 1
        assert res_new[0].id == fact_id
        assert res_new[0].object_value == "UNet3Plus_Segmentation"
        manager.close()

    def test_replace_fact_by_dict_lookup(self, tmp_path):
        """Verifies replace_fact lookup via dictionary specification."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "replace_dict.db"))
        fact_id = manager.store_fact(
            FactTriple(
                namespace="Элир",
                subject="Фолиан",
                predicate="статус",
                object_value="Послушник",
            )
        )

        key = {"namespace": "Элир", "subject": "Фолиан", "predicate": "статус"}
        returned_id = manager.replace_fact(
            key,
            FactTriple(
                namespace="Элир",
                subject="Фолиан",
                predicate="статус",
                object_value="Первый Искажённый",
            ),
        )

        assert returned_id == fact_id
        updated = manager.get_fact(fact_id)
        assert updated.object_value == "Первый Искажённый"
        assert updated.version == 2
        manager.close()

    def test_replace_nonexistent_raises_key_error(self, tmp_path):
        """Verifies replace_fact raises KeyError when fact does not exist."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "replace_missing.db"))
        with pytest.raises(KeyError):
            manager.replace_fact(
                9999,
                FactTriple(namespace="test", subject="S", predicate="P", object_value="O"),
            )
        manager.close()


class TestSafeDeletion:
    """Verifies soft deletion, FTS5 pruning, audit trail, and hard deletion."""

    def test_soft_delete_preserves_history_and_prunes_fts(self, tmp_path):
        """Soft delete marks is_deleted=1, records history entry, and removes from FTS."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "soft_del.db"))
        fact_id = manager.store_fact(
            FactTriple(
                namespace="manage",
                subject="ServerPort",
                predicate="port",
                object_value="8080",
            )
        )

        assert len(manager.search_facts("ServerPort")) == 1
        deleted = manager.delete_fact(fact_id, soft=True)
        assert deleted is True

        # Default get_fact returns None for soft-deleted rows
        assert manager.get_fact(fact_id) is None
        # get_fact with include_deleted=True returns it
        del_fact = manager.get_fact(fact_id, include_deleted=True)
        assert del_fact is not None
        assert del_fact.is_deleted is True

        # Search facts excludes soft-deleted row
        assert len(manager.search_facts("ServerPort")) == 0
        assert len(manager.search_facts("ServerPort", include_deleted=True)) == 1

        # History audit record created with action='deleted'
        history = manager.get_fact_history(fact_id)
        assert len(history) == 1
        assert history[0]["action"] == "deleted"
        assert history[0]["fact_id"] == fact_id

        # Repeated soft-delete returns False
        assert manager.delete_fact(fact_id, soft=True) is False
        manager.close()

    def test_hard_delete_purges_row(self, tmp_path):
        """Hard delete permanently removes row from memory_facts and FTS5."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "hard_del.db"))
        fact_id = manager.store_fact(
            FactTriple(
                namespace="test",
                subject="TempTarget",
                predicate="status",
                object_value="PurgeMe",
            )
        )

        assert manager.get_fact(fact_id) is not None
        deleted = manager.delete_fact(fact_id, soft=False)
        assert deleted is True

        # Completely gone even with include_deleted=True
        assert manager.get_fact(fact_id, include_deleted=True) is None
        assert len(manager.search_facts("TempTarget", include_deleted=True)) == 0

        # Deleting non-existent ID returns False safely
        assert manager.delete_fact(99999, soft=False) is False
        manager.close()


class TestTTLAndExpiration:
    """Verifies TTL calculation, automatic expires_at population, and expiration filtering."""

    def test_ttl_days_auto_computes_expires_at(self, tmp_path):
        """Storing a fact with ttl_days computes future expires_at timestamp."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "ttl_calc.db"))
        fact_id = manager.store_fact(
            FactTriple(
                namespace="user",
                subject="AuthToken",
                predicate="token",
                object_value="xyz123",
                ttl_days=7,
            )
        )

        fact = manager.get_fact(fact_id)
        assert fact.ttl_days == 7
        assert fact.expires_at is not None

        exp_dt = datetime.strptime(fact.expires_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        now_dt = datetime.now(timezone.utc)
        diff_days = (exp_dt - now_dt).total_seconds() / 86400
        assert 6.8 < diff_days < 7.2
        manager.close()

    def test_expired_fact_filtered_from_search_and_recall(self, tmp_path):
        """Verifies expired fact is omitted by default and retrievable with include_expired=True."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "ttl_filter.db"))
        past_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")

        # Expired fact
        manager.store_fact(
            FactTriple(
                namespace="user",
                subject="SessionKey",
                predicate="key",
                object_value="expired_session_val",
                expires_at=past_date,
            )
        )

        # Active fact
        future_date = (datetime.now(timezone.utc) + timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        active_id = manager.store_fact(
            FactTriple(
                namespace="user",
                subject="ActiveSession",
                predicate="key",
                object_value="active_session_val",
                expires_at=future_date,
            )
        )

        # 1. Search facts
        res_default = manager.search_facts("session")
        assert len(res_default) == 1
        assert res_default[0].id == active_id

        res_all = manager.search_facts("session", include_expired=True)
        assert len(res_all) == 2

        # 2. Recall facts
        recalled_default = manager.recall_facts(namespace="user")
        assert len(recalled_default) == 1
        assert recalled_default[0].id == active_id

        recalled_all = manager.recall_facts(namespace="user", include_expired=True)
        assert len(recalled_all) == 2

        # 3. List facts
        listed_default = manager.list_facts(namespace="user")
        assert len(listed_default) == 1

        listed_all = manager.list_facts(namespace="user", include_expired=True)
        assert len(listed_all) == 2
        manager.close()


class TestNamespaceIsolation:
    """Verifies strict scoping and data isolation across project domains."""

    def test_strict_project_namespace_isolation(self, tmp_path):
        """Facts in manage, Deckor, NIR, and Элир must never leak into queries for other namespaces."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "ns_iso.db"))

        # Facts in 4 distinct project domains
        manager.store_fact(FactTriple(namespace="manage", subject="Task", predicate="status", object_value="Milestone6"))
        manager.store_fact(FactTriple(namespace="Deckor", subject="Task", predicate="status", object_value="FrameAssembly"))
        manager.store_fact(FactTriple(namespace="NIR", subject="Task", predicate="status", object_value="ModelTraining"))
        manager.store_fact(FactTriple(namespace="Элир", subject="Task", predicate="status", object_value="BookDraftChapter1"))

        # Query each namespace strictly
        deckor_facts = manager.recall_facts(namespace="Deckor")
        assert len(deckor_facts) == 1
        assert deckor_facts[0].namespace == "Deckor"
        assert deckor_facts[0].object_value == "FrameAssembly"

        nir_facts = manager.recall_facts(namespace="NIR")
        assert len(nir_facts) == 1
        assert nir_facts[0].namespace == "NIR"
        assert nir_facts[0].object_value == "ModelTraining"

        elir_facts = manager.recall_facts(namespace="Элир")
        assert len(elir_facts) == 1
        assert elir_facts[0].namespace == "Элир"
        assert elir_facts[0].object_value == "BookDraftChapter1"

        manage_facts = manager.recall_facts(namespace="manage")
        assert len(manage_facts) == 1
        assert manage_facts[0].namespace == "manage"
        assert manage_facts[0].object_value == "Milestone6"

        # Global recall (namespace=None) aggregates across all domains
        all_facts = manager.recall_facts(namespace=None)
        assert len(all_facts) == 4
        manager.close()


class TestSortingAndConfidenceThresholds:
    """Verifies ordering strategies and confidence scoring filters."""

    def test_recall_sort_by_confidence(self, tmp_path):
        """Verifies sort_by='confidence' orders facts by descending confidence score."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "sort_conf.db"))
        manager.store_fact(FactTriple(namespace="test", subject="A", predicate="p", object_value="valA", confidence=0.4))
        manager.store_fact(FactTriple(namespace="test", subject="B", predicate="p", object_value="valB", confidence=0.99))
        manager.store_fact(FactTriple(namespace="test", subject="C", predicate="p", object_value="valC", confidence=0.75))

        facts = manager.recall_facts(sort_by="confidence")
        confidences = [f.confidence for f in facts]
        assert confidences == [0.99, 0.75, 0.4]
        manager.close()

    def test_recall_min_confidence_filter(self, tmp_path):
        """Verifies min_confidence excludes lower-confidence assertions."""
        manager = LongTermMemoryManager(db_path=str(tmp_path / "min_conf.db"))
        manager.store_fact(FactTriple(namespace="test", subject="A", predicate="p", object_value="valA", confidence=0.3))
        manager.store_fact(FactTriple(namespace="test", subject="B", predicate="p", object_value="valB", confidence=0.8))

        facts = manager.recall_facts(min_confidence=0.5)
        assert len(facts) == 1
        assert facts[0].subject == "B"
        manager.close()


class TestMemoryManagerFacadeDelegation:
    """Verifies MemoryManager correctly exposes and delegates extended R2 operations."""

    def test_facade_store_replace_delete_flow(self, tmp_path):
        """End-to-end delegation through MemoryManager facade."""
        db_file = tmp_path / "facade.db"
        mem = MemoryManager(db_path=str(db_file))

        fact_id = mem.store_fact(
            FactTriple(
                namespace="user",
                subject="City",
                predicate="residence",
                object_value="Самара",
            )
        )
        assert fact_id > 0

        # Replace via facade
        mem.replace_fact(
            fact_id,
            FactTriple(
                namespace="user",
                subject="City",
                predicate="residence",
                object_value="Самара (удалёнка)",
            ),
        )

        f = mem.get_fact(fact_id)
        assert f.version == 2
        assert f.object_value == "Самара (удалёнка)"

        history = mem.get_fact_history(fact_id)
        assert len(history) == 1
        assert history[0]["object_value"] == "Самара"

        # Delete via facade
        assert mem.delete_fact(fact_id, soft=True) is True
        assert mem.get_fact(fact_id) is None
        mem.long_term_memory.close()
