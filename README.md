# Antigravity Local Core (`manage-core`)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/Tests-Pytest%20(800%20passed)-brightgreen.svg)]()
[![Zero-Paid-API](https://img.shields.io/badge/API-Zero--Paid--API-orange.svg)]()

> **Zero-cost, local, hermetic agent runtime** providing hybrid dense/sparse RAG search, multi-project long-term memory with versioning, and progressive-disclosure skill loading via the **Model Context Protocol (MCP)** over `stdio`.

Developed for high-context AI agent environments (**Google Antigravity**, **Claude Code**, **Codex**) to eliminate external LLM/vector API costs and run 100% locally on CPU with zero RAM bloat in idle.

---

## Key Capabilities

1. **Hybrid RAG Pipeline (BM25 + Dense Vectors + RRF)**:
   - Dense semantic vector search stored as contiguous `float32 BLOB`s in SQLite with sub-5ms cosine similarity via `numpy`.
   - Sparse lexical search powered by pure-Python Okapi BM25.
   - Rank fusion via **Reciprocal Rank Fusion (RRF)** ensuring precision on exact variable names, formulas, and semantic concepts.
   - Built-in automatic embedding dimension adapter (graceful padding and truncation).
   - Instant project isolation filtering (`project_id`) preventing cross-repo context pollution across 70,000+ chunks.

2. **Self-Managed Multi-Project Memory (Letta/MemGPT-style)**:
   - Centralized SQLite database in **WAL mode** (`~/.rag_memory/global_memory.db`) with zero lock contention.
   - Full-text search via **SQLite FTS5** (`unicode61` tokenizer with full multilingual Cyrillic/Latin support).
   - Strict namespace isolation (`user`, `project_name`).
   - Atomic fact versioning (`replace_fact` with automated revision history archiving).
   - Time-To-Live (TTL) auto-expiration for ephemeral facts.

3. **Progressive Disclosure Skills Catalog**:
   - Lightweight catalog keeping agent context window clean.
   - Lazy on-demand loading of full instructions (`skill_load(name)`) with strict path-traversal jail protection.

4. **Zero-Paid-API Hermetic Execution**:
   - Delegates reasoning to host IDE subscription models through native **MCP stdio** transport (`core.mcp_server`).
   - Zero mandatory paid cloud subscriptions (no OpenAI, Pinecone, or Chroma lock-in).

---

## Architecture

```mermaid
graph TD
    User([IDE / Agent Prompt]) --> MCP[MCP Server: manage-core (stdio)]
    
    subgraph Tools ["Exposed MCP Tools"]
        MCP --> RAGSearch["rag_search(query, project)"]
        MCP --> RAGIndex["rag_index(paths)"]
        MCP --> MemRecall["memory_recall(query, namespace)"]
        MCP --> MemUpsert["memory_upsert(subject, predicate, object)"]
        MCP --> MemReplace["memory_replace(old_id, ...)"]
        MCP --> SkillLoad["skill_load(name)"]
    end

    subgraph Storage ["Local Zero-Cost Storage (~/.rag_memory/)"]
        RAGSearch & RAGIndex --> VectorDB[("SQLite Vectors (rag_vectors.db)<br/>BLOB + BM25 + RRF")]
        MemRecall & MemUpsert & MemReplace --> MemoryDB[("SQLite WAL (global_memory.db)<br/>FTS5 unicode61 + History")]
        SkillLoad --> SkillsDir["skills/<name>/SKILL.md (Jailed)"]
    end
```

---

## Quickstart

### 1. Installation
Clone the repository and install minimal dependencies:

```bash
git clone https://github.com/harvie016/antigravity-local-core.git
cd antigravity-local-core
pip install -r requirements.txt
```

### 2. Configure MCP in your IDE

#### For Antigravity (`~/.gemini/config/mcp_config.json`):
```json
{
  "mcpServers": {
    "manage-core": {
      "command": "python",
      "args": ["-m", "core.mcp_server"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/antigravity-local-core"
      }
    }
  }
}
```

#### For Claude Code (`.mcp.json`):
```json
{
  "mcpServers": {
    "manage-core": {
      "command": "python",
      "args": ["-m", "core.mcp_server"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

### 3. Run Tests
Execute the verification suite:
```bash
pytest tests/ -v
```

---

## License
MIT License. Created by [Filipp Shprengel (harvie016)](https://github.com/harvie016).
