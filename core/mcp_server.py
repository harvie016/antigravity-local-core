"""
core/mcp_server.py
Model Context Protocol (MCP) server for the local Core subsystem (stdio transport).

Exposes local RAG, Self-Managing Long-Term Memory (SQLite WAL + namespaces + TTL),
and Progressive Disclosure Skills to host IDEs (Antigravity, Claude Code, Codex)
so all LLM reasoning runs on the IDE's native subscription tokens with zero paid APIs.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

# Safe FastMCP / MCPServer import supporting both mcp 1.x and mcp 2.x
try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server.mcpserver import MCPServer as FastMCP

from core.memory.long_term_memory import DEFAULT_GLOBAL_MEMORY_PATH
from core.memory.manager import MemoryManager
from core.memory.models import FactTriple
from core.rag.models import RAGConfig
from core.rag.pipeline import RAGPipeline
from core.runtime import AgentRuntime
from core.skills import SkillNotFoundError, SkillSecurityError, SkillsIndex

# Stdio transport reserves stdout exclusively for JSON-RPC frames; log only to stderr.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("core.mcp_server")

DEFAULT_RAG_DB_PATH = Path("~/.rag_memory/rag_vectors.db").expanduser().resolve()

mcp = FastMCP(
    name="manage-core",
    instructions=(
        "Local Core MCP Server providing hybrid RAG search, persistent multi-project "
        "long-term memory (namespaces: manage, Deckor, NIR, Элир), and progressive "
        "disclosure skill loading via skill_load."
    ),
)

_runtime_lock = threading.RLock()
_runtime: Optional[AgentRuntime] = None


def _build_default_runtime() -> AgentRuntime:
    """Construct default persistent offline AgentRuntime for IDE MCP sessions."""
    workspace_root = Path(
        os.environ.get("RAG_WORKSPACE_ROOT") or Path(__file__).resolve().parent.parent
    ).resolve()

    memory_db = os.environ.get("GLOBAL_MEMORY_DB") or str(DEFAULT_GLOBAL_MEMORY_PATH)
    rag_db = os.environ.get("RAG_DB_PATH") or str(DEFAULT_RAG_DB_PATH)
    embedding_backend = os.environ.get("RAG_EMBEDDING_BACKEND", "local_tfidf")

    Path(memory_db).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(rag_db).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    memory_manager = MemoryManager(db_path=memory_db)
    rag_pipeline = RAGPipeline(
        config=RAGConfig(
            vector_store_type="sqlite",
            sqlite_db_path=rag_db,
            embedding_backend=embedding_backend,
        )
    )
    skills_index = SkillsIndex(skills_root=workspace_root / "skills")

    return AgentRuntime(
        workspace_root=workspace_root,
        memory_manager=memory_manager,
        rag_pipeline=rag_pipeline,
        skills_index=skills_index,
        enable_offline_loop=False,
    )


def get_runtime() -> AgentRuntime:
    """Thread-safe lazy getter for the shared AgentRuntime singleton."""
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = _build_default_runtime()
        return _runtime


def set_runtime(runtime: Optional[AgentRuntime]) -> None:
    """Override the shared runtime instance (primarily for isolated testing)."""
    global _runtime
    with _runtime_lock:
        _runtime = runtime


def reset_runtime() -> None:
    """Reset the shared runtime singleton so next call re-initializes it."""
    set_runtime(None)


def _format_skills_catalog(skills_index: SkillsIndex) -> str:
    """Format lightweight progressive-disclosure catalog of available skills."""
    items = skills_index.list_skills()
    if not items:
        return "No skills indexed."
    lines = ["Available skills (call skill_load(name) to load full instructions):"]
    for item in items:
        lines.append(f"- {item['name']}: {item['description']}")
    return "\n".join(lines)


@mcp.tool()
def rag_index(paths: List[str]) -> str:
    """Index document files, directories, or glob patterns into the local hybrid RAG store."""
    if not paths:
        return "Error: 'paths' must contain at least one file or directory path."
    runtime = get_runtime()
    stats = runtime.rag.index_documents(paths)
    summary = (
        f"Indexed {stats.indexed_files} file(s) "
        f"(total_files={stats.total_files}, skipped={stats.skipped_files}, "
        f"chunks={stats.total_chunks}, duration={stats.duration_seconds:.3f}s)."
    )
    if stats.errors:
        summary += f" Errors ({len(stats.errors)}): " + "; ".join(stats.errors[:5])
    return summary


@mcp.tool()
def rag_search(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.05,
    project: Optional[str] = None,
) -> str:
    """Search indexed documents across all projects or isolated by project name (e.g. 'Deckor', 'NIR', 'Элир', 'manage')."""
    runtime = get_runtime()
    results = runtime.rag.retrieve(query, top_k=top_k, score_threshold=score_threshold, project=project)
    if not results:
        proj_hint = f" in project '{project}'" if project else ""
        return f"No relevant documents found for query: '{query}'{proj_hint}."

    entries: List[str] = []
    for r in results:
        src = getattr(r.chunk, "doc_id", "doc")
        sec = f"#{r.chunk.section_title}" if getattr(r.chunk, "section_title", None) else ""
        entries.append(
            f"[{r.rank}] Source: {src}{sec} (Score: {r.score:.3f}, Mode: {r.retrieval_type})\n"
            f"{r.chunk.content}"
        )
    return "\n\n---\n\n".join(entries)


@mcp.tool()
def memory_recall(
    query: Optional[str] = None,
    namespace: Optional[str] = None,
    limit: int = 10,
    min_confidence: float = 0.0,
    include_expired: bool = False,
) -> str:
    """Recall persistent facts from long-term memory with namespace, confidence, and TTL filtering."""
    runtime = get_runtime()
    ltm = runtime.memory.long_term_memory
    facts = ltm.recall_facts(
        query=query if (query and query.strip()) else None,
        namespace=namespace if (namespace and namespace.strip()) else None,
        min_confidence=min_confidence,
        include_expired=include_expired,
        limit=limit,
    )
    if not facts:
        return (
            f"No memory facts found (query='{query or ''}', "
            f"namespace='{namespace or 'all'}')."
        )

    lines: List[str] = []
    for f in facts:
        ttl_info = f" | exp: {f.expires_at}" if f.expires_at else ""
        lines.append(
            f"[ID: {f.id} | v{f.version} | ns: {f.namespace} | conf: {f.confidence:.2f}{ttl_info}] "
            f"({f.subject}) -> [{f.predicate}] -> '{f.object_value}'"
        )
    return "\n".join(lines)


@mcp.tool()
def memory_upsert(
    subject: str,
    predicate: str,
    object_value: str,
    namespace: str = "manage",
    confidence: float = 1.0,
    ttl_days: Optional[int] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Store an atomic fact into persistent long-term memory under a project namespace."""
    runtime = get_runtime()
    ltm = runtime.memory.long_term_memory
    fact = FactTriple(
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        namespace=namespace,
        confidence=confidence,
        ttl_days=ttl_days,
        tags=tags or [],
        source_citation="mcp_server",
    )
    fact_id = ltm.store_fact(fact)
    exp_note = f", expires_at={fact.expires_at}" if fact.expires_at else ""
    return (
        f"Fact stored with ID {fact_id} (v{fact.version}) in namespace '{namespace}'"
        f"{exp_note}."
    )


@mcp.tool()
def memory_replace(
    old_fact_id: int,
    subject: str,
    predicate: str,
    object_value: str,
    namespace: str = "manage",
    confidence: float = 1.0,
    ttl_days: Optional[int] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Replace an active fact by ID, archiving the previous version in history."""
    runtime = get_runtime()
    ltm = runtime.memory.long_term_memory
    new_fact = FactTriple(
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        namespace=namespace,
        confidence=confidence,
        ttl_days=ttl_days,
        tags=tags or [],
        source_citation="mcp_server_replace",
    )
    try:
        fact_id = ltm.replace_fact(old_fact_id, new_fact)
    except (KeyError, ValueError) as exc:
        return f"Error replacing fact #{old_fact_id}: {exc}"

    return (
        f"Fact #{fact_id} updated to v{new_fact.version} in namespace '{namespace}' "
        f"(previous version archived in history)."
    )


@mcp.tool()
def skill_load(name: str) -> str:
    """Load full markdown instructions for a skill by name (or pass ''/'index' for catalog)."""
    runtime = get_runtime()
    skills_idx = getattr(runtime, "skills", None) or SkillsIndex(
        skills_root=runtime.workspace_root / "skills"
    )
    clean = (name or "").strip()
    if clean in ("", "index", "list", "__list__"):
        return _format_skills_catalog(skills_idx)

    try:
        return skills_idx.skill_load(clean)
    except SkillSecurityError as exc:
        return f"Security error loading skill '{name}': {exc}"
    except SkillNotFoundError:
        return f"Skill '{clean}' not found.\n{_format_skills_catalog(skills_idx)}"


@mcp.tool()
def skills_list() -> str:
    """Return lightweight progressive-disclosure index (name + 1-line description) of all skills."""
    runtime = get_runtime()
    skills_idx = getattr(runtime, "skills", None) or SkillsIndex(
        skills_root=runtime.workspace_root / "skills"
    )
    return _format_skills_catalog(skills_idx)


def main() -> None:
    """Run the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
