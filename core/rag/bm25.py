"""
Pure Python Okapi BM25 lexical search engine.
Implements standard Okapi BM25 (k1=1.5, b=0.75) with Lucene-style non-negative IDF,
supporting multilingual (Latin/Cyrillic) tokenization and incremental document management.
"""

import math
import re
from typing import List, Dict, Tuple, Optional
from core.rag.models import DocumentChunk


# Multilingual word regex capturing alphanumeric tokens across scripts
TOKEN_PATTERN = re.compile(r"(?u)\b[\w-]+\b")


def tokenize(text: str) -> List[str]:
    """Tokenize string into lowercase terms."""
    if not text:
        return []
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


class BM25Index:
    """
    Okapi BM25 index maintaining document frequencies, term frequencies,
    and average document length for rapid lexical ranking.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # doc_id (chunk.id) -> DocumentChunk
        self.chunks: Dict[str, DocumentChunk] = {}
        # doc_id (chunk.id) -> parent doc_id
        self.chunk_to_parent: Dict[str, str] = {}
        # doc_id -> list of tokens
        self.doc_tokens: Dict[str, List[str]] = {}
        # doc_id -> length in tokens
        self.doc_len: Dict[str, int] = {}
        # term -> {doc_id: count}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        # term -> document frequency (count of docs containing term)
        self.doc_freq: Dict[str, int] = {}
        
        self.total_docs: int = 0
        self.total_tokens: int = 0
        self.avgdl: float = 0.0

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Index a batch of DocumentChunk objects into the BM25 index.
        """
        for chunk in chunks:
            if not chunk or not chunk.id:
                continue
            
            # If chunk already indexed, remove old version first
            if chunk.id in self.chunks:
                self.remove_chunk(chunk.id)

            tokens = tokenize(chunk.content)
            # Optionally incorporate section title into tokens for better lexical hit
            if chunk.section_title:
                tokens.extend(tokenize(chunk.section_title))

            chunk_len = len(tokens)
            self.chunks[chunk.id] = chunk
            self.chunk_to_parent[chunk.id] = chunk.doc_id
            self.doc_tokens[chunk.id] = tokens
            self.doc_len[chunk.id] = chunk_len

            term_counts: Dict[str, int] = {}
            for t in tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                    self.doc_freq[term] = 0
                self.inverted_index[term][chunk.id] = count
                self.doc_freq[term] += 1

            self.total_docs += 1
            self.total_tokens += chunk_len

        if self.total_docs > 0:
            self.avgdl = self.total_tokens / self.total_docs
        else:
            self.avgdl = 0.0

    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove an individual chunk from the BM25 index."""
        if chunk_id not in self.chunks:
            return False

        tokens = self.doc_tokens.pop(chunk_id, [])
        chunk_len = self.doc_len.pop(chunk_id, 0)
        self.chunks.pop(chunk_id, None)
        self.chunk_to_parent.pop(chunk_id, None)

        unique_terms = set(tokens)
        for term in unique_terms:
            if term in self.inverted_index and chunk_id in self.inverted_index[term]:
                del self.inverted_index[term][chunk_id]
                self.doc_freq[term] -= 1
                if self.doc_freq[term] <= 0:
                    self.doc_freq.pop(term, None)
                    self.inverted_index.pop(term, None)

        self.total_docs = max(0, self.total_docs - 1)
        self.total_tokens = max(0, self.total_tokens - chunk_len)
        self.avgdl = (self.total_tokens / self.total_docs) if self.total_docs > 0 else 0.0
        return True

    def remove_document(self, doc_id: str) -> int:
        """
        Remove all chunks associated with a parent document ID.
        Returns count of removed chunks.
        """
        chunk_ids_to_remove = [
            cid for cid, pid in self.chunk_to_parent.items() if pid == doc_id
        ]
        count = 0
        for cid in chunk_ids_to_remove:
            if self.remove_chunk(cid):
                count += 1
        return count

    def clear(self) -> None:
        """Clear all indexed documents and reset statistics."""
        self.chunks.clear()
        self.chunk_to_parent.clear()
        self.doc_tokens.clear()
        self.doc_len.clear()
        self.inverted_index.clear()
        self.doc_freq.clear()
        self.total_docs = 0
        self.total_tokens = 0
        self.avgdl = 0.0

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Fetch chunk by its unique ID."""
        return self.chunks.get(chunk_id)

    def count(self) -> int:
        """Return total number of chunks currently indexed."""
        return self.total_docs

    def _compute_idf(self, term: str) -> float:
        """
        Compute Robertson-Spärck Jones Okapi IDF with Lucene smoothing:
        IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
        Guaranteed to be strictly positive (> 0.0).
        """
        n_q = self.doc_freq.get(term, 0)
        if n_q == 0 or self.total_docs == 0:
            return 0.0
        return math.log(1.0 + (self.total_docs - n_q + 0.5) / (n_q + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search indexed chunks using Okapi BM25.
        Returns ranked list of (chunk_id, score) pairs.
        """
        query_tokens = tokenize(query)
        if not query_tokens or self.total_docs == 0 or self.avgdl == 0.0:
            return []

        # Candidate accumulation
        scores: Dict[str, float] = {}
        for q in query_tokens:
            if q not in self.inverted_index:
                continue
            
            idf = self._compute_idf(q)
            if idf <= 0.0:
                continue

            posting = self.inverted_index[q]
            for doc_id, tf in posting.items():
                doc_l = self.doc_len.get(doc_id, self.avgdl)
                # BM25 term weight
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_l / self.avgdl))
                if denom > 0:
                    term_score = idf * (tf * (self.k1 + 1.0)) / denom
                    scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        if not scores:
            return []

        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
