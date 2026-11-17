"""
Hybrid retrieval engine combining BM25 lexical search and dense vector search
via Reciprocal Rank Fusion (RRF, k=60) with score normalization and threshold filtering.
"""

from typing import List, Dict, Optional, Tuple
import logging

from core.rag.models import SearchResult, DocumentChunk
from core.rag.bm25 import BM25Index
from core.rag.vector_store import BaseVectorStore
from core.rag.embeddings import BaseEmbeddingProvider

logger = logging.getLogger("rag.retriever")


class HybridRetriever:
    """
    Combines exact-match lexical keyword search (BM25) with semantic vector search
    using Reciprocal Rank Fusion (RRF) to eliminate vocabulary mismatch.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_store: BaseVectorStore,
        embedding_provider: BaseEmbeddingProvider,
        k_rrf: int = 60,
        default_alpha: float = 0.5,
        default_score_threshold: float = 0.20
    ):
        self.bm25 = bm25_index
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.k_rrf = k_rrf
        self.default_alpha = default_alpha
        self.default_score_threshold = default_score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        project: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Execute hybrid search across BM25 and dense vector store with optional project isolation.
        alpha = 0.0 -> pure lexical BM25
        alpha = 1.0 -> pure dense vector
        alpha = 0.5 -> balanced hybrid RRF (default)
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        curr_alpha = self.default_alpha if alpha is None else max(0.0, min(1.0, alpha))
        threshold = self.default_score_threshold if score_threshold is None else score_threshold
        candidate_k = max(top_k * 3, 20)

        # 1. Lexical BM25 search
        lex_ranks: Dict[str, int] = {}
        if curr_alpha < 1.0:
            bm25_hits = self.bm25.search(clean_query, top_k=candidate_k)
            for rank_idx, (cid, _) in enumerate(bm25_hits, start=1):
                lex_ranks[cid] = rank_idx

        # 2. Dense vector search
        dense_ranks: Dict[str, int] = {}
        dense_chunk_map: Dict[str, DocumentChunk] = {}
        if curr_alpha > 0.0:
            q_vec = self.embedding_provider.embed_text(clean_query)
            filter_meta = {"project": project} if project else None
            dense_hits = self.vector_store.search(q_vec, top_k=candidate_k, filter_meta=filter_meta)
            for rank_idx, sres in enumerate(dense_hits, start=1):
                cid = sres.chunk.id
                dense_ranks[cid] = rank_idx
                dense_chunk_map[cid] = sres.chunk

        # 3. Reciprocal Rank Fusion
        all_candidate_ids = set(lex_ranks.keys()).union(set(dense_ranks.keys()))
        if not all_candidate_ids:
            return []

        # Maximum possible raw score: when a document is rank 1 in both (or single if alpha=0/1)
        w_lex = 1.0 - curr_alpha
        w_dense = curr_alpha
        max_possible_rrf = (w_lex / (self.k_rrf + 1)) + (w_dense / (self.k_rrf + 1))
        if max_possible_rrf <= 0.0:
            max_possible_rrf = 1.0 / (self.k_rrf + 1)

        fused_scores: List[Tuple[str, float, str]] = [] # (chunk_id, norm_score, type)

        for cid in all_candidate_ids:
            score = 0.0
            in_lex = cid in lex_ranks
            in_dense = cid in dense_ranks

            if in_lex:
                score += w_lex / (self.k_rrf + lex_ranks[cid])
            if in_dense:
                score += w_dense / (self.k_rrf + dense_ranks[cid])

            norm_score = float(min(1.0, score / max_possible_rrf))

            if in_lex and in_dense:
                retrieval_type = "hybrid"
            elif in_lex:
                retrieval_type = "lexical"
            else:
                retrieval_type = "dense"

            if norm_score >= threshold:
                fused_scores.append((cid, norm_score, retrieval_type))

        # Sort by normalized score descending
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        # Construct SearchResult items with optional project filter
        results: List[SearchResult] = []
        rank_num = 1
        norm_proj = project.strip().lower().replace("\\", "/") if project else None

        for cid, score, rtype in fused_scores:
            chunk = dense_chunk_map.get(cid) or self.bm25.get_chunk(cid) or self.vector_store.get_chunk(cid)
            if not chunk:
                continue

            # Optional project path isolation filter
            if norm_proj and (norm_proj not in chunk.doc_id.lower().replace("\\", "/")):
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    retrieval_type=rtype,
                    rank=rank_num
                )
            )
            rank_num += 1
            if len(results) >= top_k:
                break

        return results
