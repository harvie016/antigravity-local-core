"""
tests/test_skills_and_local_rag.py
Comprehensive unit and integration test suite for Milestone 7 (Requirement R3):
- Skills progressive disclosure index (SkillsIndex, parse_frontmatter, path jail security).
- Consolidated project skills catalog (bezos, karpathy, torvalds, command-recommendations, vacancy-monitor, career-outreach).
- Local CPU semantic embeddings (LocalFastEmbedProvider, 384-dim TF-IDF fallback, SQLiteVectorStore compatibility).
- 100% offline execution with zero external network dependencies.
"""

import math
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from core.skills import (
    SkillsIndex,
    parse_frontmatter,
    SkillNotFoundError,
    SkillSecurityError,
)
from core.rag.embeddings import (
    LocalFastEmbedProvider,
    get_embedding_provider,
)
from core.rag.models import DocumentChunk
from core.rag.vector_store import SQLiteVectorStore


# ==============================================================================
# 1. FRONTMATTER PARSING & SKILLS INDEX TESTS
# ==============================================================================

class TestFrontmatterParsing:
    """Tests for pure-Python regex YAML frontmatter parser."""

    def test_valid_yaml_frontmatter_extraction(self):
        content = (
            "---\n"
            "name: test-skill\n"
            "description: \"A test skill for progressive disclosure.\"\n"
            "category: testing\n"
            "---\n"
            "# Test Skill Title\n"
            "Full instructions body here."
        )
        meta, body = parse_frontmatter(content)
        assert meta["name"] == "test-skill"
        assert meta["description"] == "A test skill for progressive disclosure."
        assert meta["category"] == "testing"
        assert "# Test Skill Title" in body
        assert "Full instructions body here." in body

    def test_frontmatter_with_colons_and_single_quotes(self):
        content = (
            "---\n"
            "name: bezos-api-first\n"
            "description: 'Манифест Джеффа Безоса: API-First, модульная изоляция и сервисы.'\n"
            "---\n"
            "Content body"
        )
        meta, body = parse_frontmatter(content)
        assert meta["name"] == "bezos-api-first"
        assert meta["description"] == "Манифест Джеффа Безоса: API-First, модульная изоляция и сервисы."
        assert body.strip() == "Content body"

    def test_frontmatter_missing_delimiters(self):
        raw_text = "# Just standard markdown\nNo frontmatter delimiters present."
        meta, body = parse_frontmatter(raw_text)
        assert meta == {}
        assert body == raw_text

    def test_frontmatter_empty_or_whitespace(self):
        meta, body = parse_frontmatter("")
        assert meta == {}
        assert body == ""

        meta2, body2 = parse_frontmatter("   \n\n  ")
        assert meta2 == {}

    def test_frontmatter_crlf_and_lf_tolerance(self):
        crlf_content = "---\r\nname: crlf-skill\r\ndescription: Windows line endings\r\n---\r\nBody with CRLF"
        meta, body = parse_frontmatter(crlf_content)
        assert meta["name"] == "crlf-skill"
        assert meta["description"] == "Windows line endings"
        assert "Body with CRLF" in body


class TestSkillsIndex:
    """Tests for filesystem scanning, progressive disclosure, and path jail."""

    @pytest.fixture
    def skills_index(self):
        return SkillsIndex()

    def test_skills_discovery_and_catalog(self, skills_index):
        skills_list = skills_index.list_skills()
        assert isinstance(skills_list, list)
        assert len(skills_list) >= 6

        names = {s["name"] for s in skills_list}
        expected_core_skills = {
            "bezos-api-first",
            "karpathy-discipline",
            "torvalds-pragmatism",
            "command-recommendations",
            "vacancy-monitor",
            "career-outreach",
        }
        assert expected_core_skills.issubset(names), f"Missing skills: {expected_core_skills - names}"

        # Progressive disclosure validation: every item must contain only name and description
        for item in skills_list:
            assert "name" in item and "description" in item
            assert len(item["description"]) > 5
            # Ensure full markdown body is not leaked into lightweight catalog
            assert len(item["description"]) < 500

    def test_skill_load_full_content(self, skills_index):
        # Load existing skills and verify markdown content
        for skill_name in ["bezos-api-first", "command-recommendations", "vacancy-monitor", "career-outreach"]:
            content = skills_index.skill_load(skill_name)
            assert isinstance(content, str)
            assert len(content) > 50
            assert f"name: {skill_name}" in content

    def test_skill_load_alias_load_skill(self, skills_index):
        res1 = skills_index.skill_load("torvalds-pragmatism")
        res2 = skills_index.load_skill("torvalds-pragmatism")
        assert res1 == res2
        assert "Linus Torvalds" in res1

    def test_skills_index_mapping_interface(self, skills_index):
        assert len(skills_index) >= 6
        assert "career-outreach" in skills_index
        assert "invalid-skill-404" not in skills_index
        # Test __getitem__
        body = skills_index["career-outreach"]
        assert "career-outreach" in body

    def test_nonexistent_skill_raises_skill_not_found(self, skills_index):
        with pytest.raises(SkillNotFoundError) as exc_info:
            skills_index.skill_load("nonexistent_skill_12345")
        
        # Verify dual inheritance for caller flexibility
        assert isinstance(exc_info.value, FileNotFoundError)
        assert isinstance(exc_info.value, KeyError)

    @pytest.mark.parametrize("malicious_name", [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "../secret.txt",
        "..\\secret.txt",
        "/etc/shadow",
        "C:\\Windows\\win.ini",
        "skill with spaces",
        "skill;rm -rf /",
        "skill\x00nullbyte",
        "",
        "   ",
        "nested/path/skill",
    ])
    def test_path_jail_security_rejection(self, skills_index, malicious_name):
        with pytest.raises(SkillSecurityError) as exc_info:
            skills_index.skill_load(malicious_name)
        
        # Verify dual inheritance: both ValueError and PermissionError
        assert isinstance(exc_info.value, ValueError)
        assert isinstance(exc_info.value, PermissionError)

    def test_hermetic_isolated_skills_index(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_skills_"))
        try:
            # Create a mock skills repository
            skill1_dir = tmp_dir / "custom-agent"
            skill1_dir.mkdir(parents=True)
            (skill1_dir / "SKILL.md").write_text(
                "---\nname: custom-agent\ndescription: Isolated test skill\n---\n# Custom Agent Body",
                encoding="utf-8"
            )

            skill2_dir = tmp_dir / "deep-analyst"
            skill2_dir.mkdir(parents=True)
            (skill2_dir / "SKILL.md").write_text(
                "---\nname: deep-analyst\ndescription: Deep analytical process\n---\n# Deep Analyst Body",
                encoding="utf-8"
            )

            isolated_index = SkillsIndex(skills_root=tmp_dir)
            skills = isolated_index.list_skills()
            assert len(skills) == 2
            assert skills[0]["name"] == "custom-agent"
            assert skills[1]["name"] == "deep-analyst"

            # Check loading
            content = isolated_index.skill_load("custom-agent")
            assert "# Custom Agent Body" in content

            # Check path jail within isolated root
            with pytest.raises(SkillSecurityError):
                isolated_index.skill_load("../other_folder")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ==============================================================================
# 2. LOCAL FASTEMBED & SEMANTIC RAG EMBEDDINGS TESTS
# ==============================================================================

class TestLocalFastEmbedProvider:
    """Tests for LocalFastEmbedProvider, fallback resilience, and vector normalization."""

    def test_default_model_and_dimension(self):
        provider = LocalFastEmbedProvider()
        assert provider.dimension == 384
        assert provider.model_name == "BAAI/bge-small-en-v1.5"

    def test_graceful_fallback_when_fastembed_missing(self):
        # On this environment, fastembed is not installed.
        provider = LocalFastEmbedProvider(dimension=384)
        
        # Before inference, not loaded
        assert not provider.is_loaded
        
        # Embed single text
        vec = provider.embed_text("Системный аналитик проектирование интеграций REST API")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)
        
        # Verify fallback is safely activated
        assert provider.is_loaded
        assert provider.is_fallback_active

        # Verify L2 normalization: norm should be approximately 1.0
        norm = np.linalg.norm(np.array(vec, dtype=np.float32))
        assert math.isclose(norm, 1.0, rel_tol=1e-4)

    def test_batch_embeddings_dimension_and_consistency(self):
        provider = LocalFastEmbedProvider(dimension=384)
        texts = [
            "Автоматизация бизнес-процессов на n8n и Make",
            "Регламент согласования графика магистратуры",
            "Postman Swagger JSON Schema SQL JOIN",
        ]
        vectors = provider.embed_batch(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 384
            norm = np.linalg.norm(np.array(v, dtype=np.float32))
            assert math.isclose(norm, 1.0, rel_tol=1e-4)

    def test_empty_and_boundary_inputs(self):
        provider = LocalFastEmbedProvider(dimension=384)
        assert provider.embed_batch([]) == []

        empty_vec = provider.embed_text("")
        assert len(empty_vec) == 384
        norm = np.linalg.norm(np.array(empty_vec, dtype=np.float32))
        assert math.isclose(norm, 1.0, rel_tol=1e-4)

    def test_factory_registration(self):
        for name in ["local_fastembed", "fastembed", "onnx", "local_onnx"]:
            prov = get_embedding_provider(name)
            assert isinstance(prov, LocalFastEmbedProvider)
            assert prov.dimension == 384

        # Custom dimension pass-through
        prov_custom = get_embedding_provider("local_fastembed", dimension=384)
        assert prov_custom.dimension == 384

    def test_mocked_fastembed_success_path(self):
        """Simulate environment where fastembed is installed and returns vectors."""
        mock_model = MagicMock()
        # Return 2 mock numpy vectors of dimension 384
        v1 = np.ones(384, dtype=np.float32)
        v2 = np.ones(384, dtype=np.float32) * 2.0
        mock_model.embed.return_value = [v1, v2]

        provider = LocalFastEmbedProvider(dimension=384)
        provider._model = mock_model
        provider._is_loaded = True
        provider._fallback_active = False

        results = provider.embed_batch(["text 1", "text 2"])
        assert len(results) == 2
        assert len(results[0]) == 384
        assert not provider.is_fallback_active
        mock_model.embed.assert_called_once_with(["text 1", "text 2"])

        # Check normalization
        norm = np.linalg.norm(np.array(results[0], dtype=np.float32))
        assert math.isclose(norm, 1.0, rel_tol=1e-4)

    def test_mocked_fastembed_runtime_error_falls_back(self):
        """Simulate unexpected inference exception in fastembed triggering graceful fallback."""
        mock_model = MagicMock()
        mock_model.embed.side_effect = RuntimeError("ONNX Runtime internal CUDA/CPU error")

        provider = LocalFastEmbedProvider(dimension=384)
        provider._model = mock_model
        provider._is_loaded = True
        provider._fallback_active = False

        # Should not raise; must seamlessly fall back to LocalTFIDF with 384 dimensions
        results = provider.embed_batch(["test text"])
        assert len(results) == 1
        assert len(results[0]) == 384
        assert provider.is_fallback_active


# ==============================================================================
# 3. VECTOR STORE INTEGRATION & NETWORK ISOLATION TESTS
# ==============================================================================

class TestVectorStoreIntegrationAndOfflineGuarantees:
    """Verifies SQLiteVectorStore dimension compatibility and zero network calls."""

    def test_sqlite_vector_store_compatibility_with_fastembed_provider(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_sqlite_vec_"))
        try:
            db_path = tmp_dir / "test_vec.db"
            store = SQLiteVectorStore(db_path=db_path)
            provider = LocalFastEmbedProvider(dimension=384)

            # Generate embeddings
            c1_text = "Проектирование спецификаций REST API и схемы базы данных"
            c2_text = "Парсинг вакансий HeadHunter и отклики в Telegram"
            v1 = provider.embed_text(c1_text)
            v2 = provider.embed_text(c2_text)

            chunk1 = DocumentChunk(
                id="chunk_api_1",
                doc_id="doc_analyst",
                content=c1_text,
                chunk_index=0,
                start_char=0,
                end_char=len(c1_text),
                metadata={"source": "analyst.md"}
            )
            chunk2 = DocumentChunk(
                id="chunk_hh_1",
                doc_id="doc_vacancies",
                content=c2_text,
                chunk_index=0,
                start_char=0,
                end_char=len(c2_text),
                metadata={"source": "vacancies.md"}
            )

            # Add chunks to SQLite vector store (stores 384-dim BLOBs)
            store.add_chunks([chunk1, chunk2], [v1, v2])
            assert store.count() == 2

            # Query vector store with same provider
            query_vec = provider.embed_text("REST API интеграции")
            assert len(query_vec) == 384

            # Search must succeed without dimension mismatch exception
            results = store.search(query_vec, top_k=2)
            assert len(results) == 2
            assert results[0].chunk.id == "chunk_api_1"
            assert results[0].score >= results[1].score

            store.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_zero_external_network_calls(self):
        """Assert that LocalFastEmbedProvider and SkillsIndex execute with zero network calls."""
        with patch("urllib.request.urlopen") as mock_url, patch("http.client.HTTPConnection.connect") as mock_http:
            mock_url.side_effect = AssertionError("Network access attempted during offline embedding!")
            mock_http.side_effect = AssertionError("Network connection attempted during offline embedding!")

            # 1. Skills Indexing & Loading
            idx = SkillsIndex()
            skills = idx.list_skills()
            assert len(skills) >= 6
            content = idx.skill_load("career-outreach")
            assert len(content) > 0

            # 2. Local Embedding Generation
            embedder = LocalFastEmbedProvider()
            vec = embedder.embed_text("100% offline zero-network test")
            assert len(vec) == 384

            # Verify no network mock was called
            mock_url.assert_not_called()
            mock_http.assert_not_called()
