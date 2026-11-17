"""
Core RAG Subsystem package.
Provides document chunking, zero-paid local embeddings, pure Python BM25 lexical search,
SQLite BLOB & in-memory vector stores, hybrid RRF retrieval, context injection, and unified RAGPipeline.
"""

from core.rag.models import (
    DocumentMetadata,
    DocumentChunk,
    SearchResult,
    ContextSnippet,
    RAGConfig,
    IndexingStats,
)
from core.rag.chunker import (
    BaseChunker,
    RecursiveTextChunker,
    MarkdownChunker,
    CodeChunker,
    JSONChunker,
    get_chunker_for_file,
)
from core.rag.embeddings import (
    BaseEmbeddingProvider,
    LocalTFIDFEmbeddingProvider,
    LocalHFEmbeddingProvider,
    GigaChatEmbeddingProvider,
    get_embedding_provider,
)
from core.rag.bm25 import BM25Index, tokenize
from core.rag.vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    SQLiteVectorStore,
)
from core.rag.retriever import HybridRetriever
from core.rag.context_injector import ContextInjector
from core.rag.pipeline import RAGPipeline

__all__ = [
    # Models
    "DocumentMetadata",
    "DocumentChunk",
    "SearchResult",
    "ContextSnippet",
    "RAGConfig",
    "IndexingStats",
    # Chunkers
    "BaseChunker",
    "RecursiveTextChunker",
    "MarkdownChunker",
    "CodeChunker",
    "JSONChunker",
    "get_chunker_for_file",
    # Embeddings
    "BaseEmbeddingProvider",
    "LocalTFIDFEmbeddingProvider",
    "LocalHFEmbeddingProvider",
    "GigaChatEmbeddingProvider",
    "get_embedding_provider",
    # BM25
    "BM25Index",
    "tokenize",
    # Vector Stores
    "BaseVectorStore",
    "InMemoryVectorStore",
    "SQLiteVectorStore",
    # Retriever & Injector
    "HybridRetriever",
    "ContextInjector",
    # Pipeline Facade
    "RAGPipeline",
]
