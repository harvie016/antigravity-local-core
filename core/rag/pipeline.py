"""
Unified RAGPipeline facade connecting chunking, indexing, storage, hybrid retrieval,
and context injection conforming to the PROJECT.md interface contract.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import glob
import time
import hashlib
import logging

from core.rag.models import (
    RAGConfig,
    DocumentMetadata,
    DocumentChunk,
    SearchResult,
    ContextSnippet,
    IndexingStats,
)
from core.rag.chunker import get_chunker_for_file, RecursiveTextChunker
from core.rag.embeddings import (
    BaseEmbeddingProvider,
    LocalTFIDFEmbeddingProvider,
    get_embedding_provider,
)
from core.rag.bm25 import BM25Index
from core.rag.vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    SQLiteVectorStore,
)
from core.rag.retriever import HybridRetriever
from core.rag.context_injector import ContextInjector

logger = logging.getLogger("rag.pipeline")

SUPPORTED_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".text", ".py", ".json", ".jsonl",
    ".js", ".ts", ".html", ".css", ".sql", ".sh", ".yaml", ".yml"
}


def compute_file_hash(content: str) -> str:
    """Compute deterministic SHA-256 hash of document string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class RAGPipeline:
    """
    Unified end-to-end RAG facade providing document indexing,
    hybrid lexical/semantic retrieval, and token-budgeted prompt context injection.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
    ):
        self.config = config or RAGConfig()
        
        # 1. Embedding Provider
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            self.embedding_provider = get_embedding_provider(
                self.config.embedding_backend,
                dimension=128
            )

        # 2. Vector Store
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            if self.config.vector_store_type == "memory":
                self.vector_store = InMemoryVectorStore()
            else:
                self.vector_store = SQLiteVectorStore(self.config.sqlite_db_path)

        # 3. BM25 Lexical Index
        self.bm25 = bm25_index or BM25Index()

        # 4. Hybrid Retriever
        self.retriever = HybridRetriever(
            bm25_index=self.bm25,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            k_rrf=60,
            default_alpha=self.config.hybrid_alpha,
            default_score_threshold=self.config.score_threshold
        )

        # 5. Context Injector
        self.context_injector = ContextInjector(
            default_max_tokens=self.config.max_context_tokens
        )

        # In-memory document hash cache: doc_id -> hash
        self._doc_hashes: Dict[str, str] = {}

    def index_documents(self, paths: List[str]) -> IndexingStats:
        """
        Scan and index all files in given paths (supports file paths, directory paths, and globs).
        Uses SHA-256 hashes to skip unchanged files incrementally.
        """
        start_time = time.time()
        stats = IndexingStats()

        discovered_files: List[str] = []
        for p in paths:
            # Handle globs
            if "*" in p or "?" in p:
                matched = glob.glob(p, recursive=True)
                discovered_files.extend(m for m in matched if os.path.isfile(m))
            elif os.path.isdir(p):
                # Recursively walk directory
                for root, _, files in os.walk(p):
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in SUPPORTED_EXTENSIONS:
                            discovered_files.append(os.path.join(root, f))
            elif os.path.isfile(p):
                discovered_files.append(p)

        # Remove duplicate file paths while preserving order
        unique_files: List[str] = list(dict.fromkeys(discovered_files))
        stats.total_files = len(unique_files)

        all_new_texts: List[str] = []
        all_new_chunks: List[DocumentChunk] = []

        for fpath in unique_files:
            try:
                # Read file content safely
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                # Skip binary files containing null bytes
                if "\x00" in content:
                    stats.skipped_files += 1
                    continue

                clean_content = content.strip()
                if not clean_content:
                    stats.skipped_files += 1
                    continue

                fhash = compute_file_hash(clean_content)
                normalized_path = os.path.abspath(fpath)

                # Incremental caching check: skip if hash unchanged
                if self._doc_hashes.get(normalized_path) == fhash:
                    stats.skipped_files += 1
                    continue

                # Remove existing chunks for this document before re-indexing
                self.bm25.remove_document(normalized_path)
                self.vector_store.delete_document(normalized_path)

                ext = os.path.splitext(fpath)[1].lstrip(".") or "text"
                metadata = DocumentMetadata(
                    source=normalized_path,
                    file_type=ext,
                    file_hash=fhash,
                    title=os.path.basename(fpath),
                    created_at=os.path.getmtime(fpath),
                    extra={"file_size": len(clean_content)}
                )

                chunker = get_chunker_for_file(
                    fpath,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                chunks = chunker.chunk(clean_content, metadata)
                if not chunks:
                    stats.skipped_files += 1
                    continue

                for ch in chunks:
                    all_new_texts.append(ch.content)
                    all_new_chunks.append(ch)

                self._doc_hashes[normalized_path] = fhash
                stats.indexed_files += 1

            except Exception as e:
                err_msg = f"Failed to index '{fpath}': {str(e)}"
                logger.error(err_msg)
                stats.errors.append(err_msg)

        # If we have chunks, generate embeddings and write to stores
        if all_new_chunks:
            # If using LocalTFIDF and not yet fitted, fit vocabulary
            if isinstance(self.embedding_provider, LocalTFIDFEmbeddingProvider):
                if not getattr(self.embedding_provider, "_is_fitted", False):
                    self.embedding_provider.fit(all_new_texts)

            embeddings = self.embedding_provider.embed_batch(all_new_texts)
            self.vector_store.add_chunks(all_new_chunks, embeddings)
            self.bm25.add_documents(all_new_chunks)
            stats.total_chunks += len(all_new_chunks)

        stats.duration_seconds = round(time.time() - start_time, 4)
        return stats

    def index_text(
        self,
        content: str,
        source: str = "raw_text",
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Directly index raw text in memory without requiring a file on disk.
        """
        clean_content = content.strip() if content else ""
        if not clean_content:
            return []

        doc_meta = DocumentMetadata(
            source=source,
            file_type="text",
            file_hash=compute_file_hash(clean_content),
            title=title or source,
            extra=metadata or {}
        )

        chunker = RecursiveTextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        chunks = chunker.chunk(clean_content, doc_meta)
        if not chunks:
            return []

        # Remove previous version if exists
        self.bm25.remove_document(source)
        self.vector_store.delete_document(source)

        texts = [c.content for c in chunks]
        if isinstance(self.embedding_provider, LocalTFIDFEmbeddingProvider):
            if not getattr(self.embedding_provider, "_is_fitted", False):
                self.embedding_provider.fit(texts)

        embeddings = self.embedding_provider.embed_batch(texts)
        self.vector_store.add_chunks(chunks, embeddings)
        self.bm25.add_documents(chunks)
        self._doc_hashes[source] = doc_meta.file_hash

        return chunks

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        project: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Retrieve relevant document chunks using hybrid search (BM25 + vector RRF)
        with optional project isolation.
        """
        k = top_k if top_k is not None else self.config.top_k
        thresh = score_threshold if score_threshold is not None else self.config.score_threshold
        a = alpha if alpha is not None else self.config.hybrid_alpha
        return self.retriever.retrieve(query, top_k=k, score_threshold=thresh, alpha=a, project=project)

    def inject_context(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> Tuple[str, List[ContextSnippet]]:
        """
        Retrieve chunks and format them into a verifiable prompt context block.
        Returns:
            (context_xml_str, list_of_snippets)
        """
        results = self.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        budget = max_tokens if max_tokens is not None else self.config.max_context_tokens
        return self.context_injector.format_context(results, max_tokens=budget)

    def clear(self) -> None:
        """Purge all stored vectors, BM25 indices, and document hashes."""
        self.vector_store.clear()
        self.bm25.clear()
        self._doc_hashes.clear()

    def count(self) -> int:
        """Return total chunks currently indexed in vector store."""
        return self.vector_store.count()

    def close(self) -> None:
        """Release underlying vector store and resources."""
        self.vector_store.close()
