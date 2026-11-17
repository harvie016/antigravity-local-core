"""
Pydantic v2 data models for the RAG subsystem.
Conforms to interface contracts specified in PROJECT.md and explorer survey handoff.
"""

from typing import List, Dict, Any, Optional
import time
import uuid
from pydantic import BaseModel, Field, ConfigDict


class DocumentMetadata(BaseModel):
    """Metadata describing a parent source document before or during chunking."""
    model_config = ConfigDict(extra="ignore")

    source: str = Field(..., description="File path, URI, or identifier of original document")
    file_type: str = Field("text", description="File extension or content format: md, txt, json, py, etc.")
    file_hash: str = Field("", description="SHA-256 hash for incremental indexing cache")
    title: Optional[str] = Field(None, description="Human-readable title or heading of the document")
    created_at: float = Field(default_factory=time.time, description="Unix timestamp of ingestion")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary user or file metadata")


class DocumentChunk(BaseModel):
    """An individual chunk extracted from a parent document with offset and section tracking."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique chunk UUID")
    doc_id: str = Field(..., description="Parent document identifier or source path")
    content: str = Field(..., description="Text content of the chunk")
    chunk_index: int = Field(0, description="Sequential zero-based index of this chunk in parent document")
    start_char: int = Field(0, description="Start character offset in parent document content")
    end_char: int = Field(0, description="End character offset in parent document content")
    section_title: Optional[str] = Field(None, description="Hierarchical breadcrumb or code symbol name")
    token_count: int = Field(0, description="Approximated or exact token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata inherited or computed")


class SearchResult(BaseModel):
    """Retrieved search result carrying chunk, score, rank, and retrieval method."""
    model_config = ConfigDict(extra="ignore")

    chunk: DocumentChunk
    score: float = Field(..., description="Normalized relevance score or RRF fused score")
    retrieval_type: str = Field("hybrid", description="Retrieval source: lexical, dense, or hybrid")
    rank: int = Field(1, description="1-based rank position in search results")


class ContextSnippet(BaseModel):
    """Extracted snippet packed into agent prompt context with verifiable citation reference."""
    model_config = ConfigDict(extra="ignore")

    source: str = Field(..., description="Source path or URI")
    section: str = Field("", description="Section breadcrumb or symbol name")
    content: str = Field(..., description="Extracted content text")
    score: float = Field(0.0, description="Relevance score")
    citation_id: str = Field(..., description="Formatted citation identifier, e.g. Source 1")


class RAGConfig(BaseModel):
    """Global configuration settings for RAG indexing, search, and context packing."""
    model_config = ConfigDict(extra="ignore")

    chunk_size: int = Field(600, ge=50, le=8000, description="Target chunk size in characters")
    chunk_overlap: int = Field(80, ge=0, le=2000, description="Overlap in characters between consecutive chunks")
    top_k: int = Field(5, ge=1, le=100, description="Default number of search results to return")
    score_threshold: float = Field(0.20, ge=0.0, le=1.0, description="Minimum score threshold for retrieved results")
    max_context_tokens: int = Field(2048, ge=64, le=65536, description="Token budget for context injection")
    hybrid_alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight: 0.0=pure lexical BM25, 1.0=pure dense vector")
    embedding_backend: str = Field("local_tfidf", description="local_tfidf, local_hf, or gigachat")
    vector_store_type: str = Field("sqlite", description="sqlite or memory")
    sqlite_db_path: str = Field("storage/rag.db", description="Path to SQLite vector database")


class IndexingStats(BaseModel):
    """Telemetry report produced after document indexing."""
    model_config = ConfigDict(extra="ignore")

    total_files: int = Field(0, description="Total files discovered for indexing")
    indexed_files: int = Field(0, description="Files newly indexed or updated")
    skipped_files: int = Field(0, description="Files skipped because hash was unchanged")
    total_chunks: int = Field(0, description="Total chunks produced and stored")
    duration_seconds: float = Field(0.0, description="Duration of indexing in seconds")
    errors: List[str] = Field(default_factory=list, description="List of error messages if any occurred")
