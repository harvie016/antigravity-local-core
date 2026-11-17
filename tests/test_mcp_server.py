"""
tests/test_mcp_server.py
Unit, integration, security, stdio JSON-RPC, and IDE config verification tests
for core/mcp_server.py and API-key-decoupled AgentRuntime.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from core.mcp_server import mcp, reset_runtime, set_runtime
from core.memory.manager import MemoryManager
from core.memory.models import FactTriple
from core.rag.models import RAGConfig
from core.rag.pipeline import RAGPipeline
from core.runtime import AgentRuntime
from core.skills import SkillsIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _call_mcp_tool(name: str, arguments: dict) -> str:
    """Invoke an MCP tool in-memory and extract combined text content."""
    result = asyncio.run(mcp.call_tool(name, arguments))
    # mcp 2.x returns CallToolResult(content=[TextContent(...)]); mcp 1.x may return list/tuple
    blocks = getattr(result, "content", result)
    if isinstance(blocks, tuple):
        blocks = blocks[0]
    if isinstance(blocks, list):
        return "\n".join(getattr(b, "text", str(b)) for b in blocks)
    return str(blocks)


@pytest.fixture
def isolated_mcp_runtime(tmp_path):
    """Provide an isolated hermetic AgentRuntime backed by tmp_path SQLite databases."""
    mem_db = tmp_path / "global_memory.db"
    rag_db = tmp_path / "rag_vectors.db"
    vault_dir = tmp_path / "vault"

    memory_manager = MemoryManager(db_path=str(mem_db), vault_dir=vault_dir)
    rag_pipeline = RAGPipeline(
        config=RAGConfig(
            vector_store_type="sqlite",
            sqlite_db_path=str(rag_db),
            embedding_backend="local_fastembed",
        )
    )
    skills_index = SkillsIndex(skills_root=PROJECT_ROOT / "skills")

    runtime = AgentRuntime(
        workspace_root=PROJECT_ROOT,
        memory_manager=memory_manager,
        rag_pipeline=rag_pipeline,
        skills_index=skills_index,
        enable_offline_loop=False,
    )
    set_runtime(runtime)
    try:
        yield runtime
    finally:
        reset_runtime()
        rag_pipeline.close()
        memory_manager.long_term_memory.close()


class TestMCPServerRegistrationAndTools:
    """Verify all required tools are registered and function hermetically offline."""

    def test_mcp_server_registers_all_required_tools(self):
        tools = asyncio.run(mcp.list_tools())
        tool_names = {t.name for t in tools}
        required = {
            "rag_index",
            "rag_search",
            "memory_recall",
            "memory_upsert",
            "memory_replace",
            "skill_load",
        }
        assert required.issubset(tool_names), f"Missing MCP tools: {required - tool_names}"

    def test_mcp_rag_index_and_search_roundtrip(self, isolated_mcp_runtime, tmp_path):
        doc = tmp_path / "architecture_notes.md"
        doc.write_text(
            "# Deckor Geometry Engine\n\n"
            "Deckor calculates timber truss loads and exports 3D scenes to standalone HTML.\n"
            "Проект использует Python и модульную архитектуру каркасов.\n",
            encoding="utf-8",
        )

        index_out = _call_mcp_tool("rag_index", {"paths": [str(doc)]})
        assert "Indexed 1 file(s)" in index_out
        assert "chunks=" in index_out

        search_out = _call_mcp_tool(
            "rag_search",
            {"query": "Deckor timber truss 3D HTML", "top_k": 3, "score_threshold": 0.01},
        )
        assert "architecture_notes.md" in search_out
        assert "Deckor" in search_out

        empty_out = _call_mcp_tool("rag_index", {"paths": []})
        assert "Error:" in empty_out

    def test_mcp_memory_upsert_recall_replace_and_namespaces(self, isolated_mcp_runtime):
        # Store facts across distinct project namespaces
        up_manage = _call_mcp_tool(
            "memory_upsert",
            {
                "subject": "manage",
                "predicate": "llm_mode",
                "object_value": "IDE MCP delegation without paid API",
                "namespace": "manage",
                "confidence": 0.95,
            },
        )
        assert "Fact stored with ID" in up_manage
        match_id = re.search(r"ID\s+(\d+)", up_manage)
        assert match_id is not None
        manage_fact_id = int(match_id.group(1))

        up_deckor = _call_mcp_tool(
            "memory_upsert",
            {
                "subject": "Deckor",
                "predicate": "export_format",
                "object_value": "Interactive HTML 3D viewer",
                "namespace": "Deckor",
                "confidence": 1.0,
            },
        )
        assert "namespace 'Deckor'" in up_deckor

        # Namespace isolation check
        deckor_recall = _call_mcp_tool(
            "memory_recall", {"query": "HTML", "namespace": "Deckor", "limit": 10}
        )
        assert "Interactive HTML 3D viewer" in deckor_recall
        assert "IDE MCP delegation" not in deckor_recall

        # Replace fact with version bump + history archival
        replace_out = _call_mcp_tool(
            "memory_replace",
            {
                "old_fact_id": manage_fact_id,
                "subject": "manage",
                "predicate": "llm_mode",
                "object_value": "Antigravity + Claude Code MCP stdio server",
                "namespace": "manage",
                "confidence": 1.0,
                "ttl_days": 30,
            },
        )
        assert "updated to v2" in replace_out
        assert "archived in history" in replace_out

        recalled_updated = _call_mcp_tool(
            "memory_recall", {"query": "Antigravity", "namespace": "manage"}
        )
        assert "v2" in recalled_updated
        assert "Antigravity + Claude Code MCP stdio server" in recalled_updated

        history = isolated_mcp_runtime.memory.get_fact_history(manage_fact_id)
        assert len(history) >= 1
        assert history[0]["object_value"] == "IDE MCP delegation without paid API"

        # Replacing non-existent ID returns clean error string
        bad_replace = _call_mcp_tool(
            "memory_replace",
            {
                "old_fact_id": 999999,
                "subject": "x",
                "predicate": "y",
                "object_value": "z",
            },
        )
        assert "Error replacing fact #999999" in bad_replace

    def test_mcp_memory_recall_filters_expired_ttl(self, isolated_mcp_runtime):
        ltm = isolated_mcp_runtime.memory.long_term_memory
        expired_fact = FactTriple(
            namespace="NIR",
            subject="NIR",
            predicate="temp_note",
            object_value="expired radionuclide draft",
            expires_at="2020-01-01 00:00:00",
        )
        ltm.store_fact(expired_fact, sync_vault=False)

        active_out = _call_mcp_tool(
            "memory_recall", {"query": "radionuclide", "namespace": "NIR", "include_expired": False}
        )
        assert "No memory facts found" in active_out

        all_out = _call_mcp_tool(
            "memory_recall", {"query": "radionuclide", "namespace": "NIR", "include_expired": True}
        )
        assert "expired radionuclide draft" in all_out

    def test_mcp_skill_load_progressive_disclosure_and_path_jail(self, isolated_mcp_runtime):
        catalog = _call_mcp_tool("skills_list", {})
        assert "bezos-api-first" in catalog
        assert "karpathy-discipline" in catalog

        catalog_via_empty = _call_mcp_tool("skill_load", {"name": ""})
        assert "bezos-api-first" in catalog_via_empty

        full_skill = _call_mcp_tool("skill_load", {"name": "bezos-api-first"})
        expected_text = (PROJECT_ROOT / "skills" / "bezos-api-first" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert full_skill == expected_text

        traversal_out = _call_mcp_tool("skill_load", {"name": "../../etc/passwd"})
        assert "Security error" in traversal_out

        missing_out = _call_mcp_tool("skill_load", {"name": "nonexistent_skill_xyz"})
        assert "not found" in missing_out
        assert "bezos-api-first" in missing_out


class TestAgentRuntimeAPIDecoupling:
    """Verify AgentRuntime requires zero API keys and keeps loop.py as optional runner."""

    def test_agent_runtime_decoupled_from_api_keys(self, monkeypatch, tmp_path):
        for key in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GIGACHAT_AUTH_KEY",
            "GIGACHAT_CREDENTIALS",
        ):
            monkeypatch.delenv(key, raising=False)

        runtime = AgentRuntime(
            workspace_root=tmp_path,
            db_path=str(tmp_path / "mem.db"),
        )
        status = runtime.status()
        assert status["offline_mode"] is True
        assert status["api_keys_required"] is False
        assert status["llm_mode"] == "ide_delegated"
        assert status["offline_runner_initialized"] is False
        assert runtime._loop is None

        # Accessing optional offline runner initializes it lazily without GigaChat
        loop = runtime.loop
        assert loop is not None
        assert loop.llm_client is None
        assert runtime.status()["offline_runner_initialized"] is True


class TestIDEConfigsAndStdioHandshake:
    """Verify real IDE MCP configs and live stdio JSON-RPC handshake."""

    def test_ide_configs_valid_and_registered(self):
        antigravity_cfg = Path.home() / ".gemini" / "config" / "mcp_config.json"
        claude_project_cfg = PROJECT_ROOT / ".mcp.json"
        claude_user_cfg = Path.home() / ".claude.json"

        for cfg_path in (antigravity_cfg, claude_project_cfg):
            assert cfg_path.is_file(), f"Missing IDE MCP config: {cfg_path}"
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert "mcpServers" in data and "manage-core" in data["mcpServers"]
            srv = data["mcpServers"]["manage-core"]
            assert Path(srv["command"]).is_file()
            assert srv["args"] == ["-m", "core.mcp_server"]
            assert Path(srv["env"]["PYTHONPATH"]).resolve() == PROJECT_ROOT.resolve()

        if claude_user_cfg.is_file():
            user_data = json.loads(claude_user_cfg.read_text(encoding="utf-8"))
            assert "manage-core" in user_data.get("mcpServers", {})

    def test_mcp_server_real_stdio_initialize_and_list_tools(self, tmp_path):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["PYTHONUTF8"] = "1"
        env["GLOBAL_MEMORY_DB"] = str(tmp_path / "stdio_mem.db")
        env["RAG_DB_PATH"] = str(tmp_path / "stdio_rag.db")
        env["RAG_EMBEDDING_BACKEND"] = "local_tfidf"

        proc = subprocess.Popen(
            [sys.executable, "-m", "core.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            encoding="utf-8",
        )
        try:
            init_req = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "pytest-client", "version": "1.0"},
                    },
                }
            )
            initialized_notif = json.dumps(
                {"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
            list_req = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})

            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(init_req + "\n")
            proc.stdin.flush()

            by_id = {}
            deadline_lines = 0
            while 1 not in by_id and deadline_lines < 50:
                line = proc.stdout.readline()
                if not line:
                    break
                deadline_lines += 1
                stripped = line.strip()
                if stripped.startswith("{"):
                    msg = json.loads(stripped)
                    if "id" in msg:
                        by_id[msg["id"]] = msg

            assert 1 in by_id, f"Missing initialize response. by_id={by_id!r}"
            assert "result" in by_id[1]
            assert by_id[1]["result"]["serverInfo"]["name"] == "manage-core"

            proc.stdin.write(initialized_notif + "\n" + list_req + "\n")
            proc.stdin.flush()

            while 2 not in by_id and deadline_lines < 100:
                line = proc.stdout.readline()
                if not line:
                    break
                deadline_lines += 1
                stripped = line.strip()
                if stripped.startswith("{"):
                    msg = json.loads(stripped)
                    if "id" in msg:
                        by_id[msg["id"]] = msg

            proc.stdin.close()
            proc.wait(timeout=10)

            assert 2 in by_id, f"Missing tools/list response. by_id={by_id!r}"
            listed = {t["name"] for t in by_id[2]["result"]["tools"]}
            assert {
                "rag_index",
                "rag_search",
                "memory_recall",
                "memory_upsert",
                "memory_replace",
                "skill_load",
            }.issubset(listed)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
