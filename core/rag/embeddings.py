"""
Pluggable embedding providers for RAG subsystem.
Provides zero-paid deterministic LocalTFIDFEmbeddingProvider, optional LocalHFEmbeddingProvider,
and GigaChatEmbeddingProvider with automatic HTTP 402 / error circuit-breaker.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import json
import logging
import urllib.request
import hashlib
import numpy as np

logger = logging.getLogger("rag.embeddings")

# Default base bilingual seed corpus to ensure vocabulary stability and sufficient SVD rank
DEFAULT_SEED_CORPUS = [
    "python programming language function class method module async await typing pydantic",
    "retrieval augmented generation rag vector store search embeddings index chunking text",
    "okapi bm25 lexical keyword matching inverted index term frequency inverse document frequency",
    "sqlite database query table blob float32 memory persistence transaction wal mode journal",
    "hybrid search reciprocal rank fusion rrf score ranking normalization threshold context",
    "context injection citation prompt token budget greedy packing deduplication source",
    "multi-agent system orchestrator planner executor reviewer intent router tool execution",
    "session memory long term episodic fact triple fts5 markdown vault history state",
    "environment adapter antigravity claude code codex openai gemini mcp tools configuration",
    "разработка архитектура система мультиагентная память контекст индексация документов",
    "поиск семантический лексический гибридный векторная база данных хранилище документов",
    "цитирование верификация тесты качество надежность автоматизация фабрика компонентов",
    "клиент сервер запрос ответ параметры валидация обработка ошибок фаллбэк провайдер",
    "машинное обучение искусственный интеллект эмбеддинги проекция сингулярное разложение svd",
    "самарский университет экономика информатика системный аналитик проект разработка код",
    "gigachat sberbank api completions embeddings token authorization header bearer ssl",
    "markdown headers code ast json flattening parser delimiters recursive sliding window",
    "fastapi starlette endpoints rest api routing pydantic v2 settings configuration models",
    "performance benchmark latency throughput millisecond optimization clean vanilla python",
    "unit test integration test pytest assertions fixture mock verification coverage zero paid"
]


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense text embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into a dense float vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings into a list of dense float vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Fixed dimensionality of generated embedding vectors."""
        pass


class LocalTFIDFEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic zero-cost embedding provider using scikit-learn
    TfidfVectorizer + TruncatedSVD with L2 normalization.
    Guarantees fixed dimension and sub-millisecond offline execution.
    """

    def __init__(self, dimension: int = 128):
        self._dim = dimension
        self._vectorizer = None
        self._svd = None
        self._is_fitted = False
        self._corpus: List[str] = list(DEFAULT_SEED_CORPUS)
        self._fit_internal()

    @property
    def dimension(self) -> int:
        return self._dim

    def fit(self, additional_texts: Optional[List[str]] = None) -> None:
        """Fit or refit TF-IDF and SVD on seed corpus plus additional document texts."""
        if additional_texts:
            clean = [t.strip() for t in additional_texts if t and t.strip()]
            self._corpus.extend(clean)
        self._fit_internal()

    def _fit_internal(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                token_pattern=r"(?u)\b[\w-]+\b",
                lowercase=True,
                ngram_range=(1, 2)
            )
            tfidf_mat = self._vectorizer.fit_transform(self._corpus)
            n_features = tfidf_mat.shape[1]
            n_samples = tfidf_mat.shape[0]

            # SVD components must be < min(n_samples, n_features)
            target_components = min(self._dim, max(2, min(n_samples - 1, n_features - 1)))
            if target_components > 1:
                self._svd = TruncatedSVD(n_components=target_components, random_state=42)
                self._svd.fit(tfidf_mat)
            else:
                self._svd = None

            self._is_fitted = True
        except Exception as e:
            logger.warning(f"Error during TF-IDF fitting: {e}. Fallback to deterministic projection.")
            self._is_fitted = False

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float]] = []

        if not self._is_fitted or self._vectorizer is None:
            # Fallback deterministic hash projection if sklearn unavailable
            for t in texts:
                vec = self._hash_projection(t)
                results.append(vec)
            return results

        try:
            tfidf_mat = self._vectorizer.transform(texts)
            if self._svd is not None:
                reduced = self._svd.transform(tfidf_mat)
            else:
                reduced = tfidf_mat.toarray()

            for i in range(len(texts)):
                row = reduced[i].astype(np.float32)
                # Pad or truncate to self._dim
                if len(row) < self._dim:
                    padded = np.zeros(self._dim, dtype=np.float32)
                    padded[:len(row)] = row
                    row = padded
                elif len(row) > self._dim:
                    row = row[:self._dim]

                # L2 normalize
                norm = np.linalg.norm(row)
                if norm > 1e-12:
                    row = row / norm
                else:
                    # Deterministic hash projection fallback for OOV / zero-norm
                    row = np.array(self._hash_projection(texts[i]), dtype=np.float32)

                results.append(row.tolist())
        except Exception as e:
            logger.warning(f"TF-IDF embedding transform error: {e}. Using hash projection.")
            for t in texts:
                results.append(self._hash_projection(t))

        return results

    def _hash_projection(self, text: str) -> List[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            vec[0] = 1.0
            return vec.tolist()
        for t in tokens:
            seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec += rng.randn(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()


class LocalHFEmbeddingProvider(BaseEmbeddingProvider):
    """
    Optional Hugging Face transformers embedder running locally on CPU.
    Requires transformers and torch.
    """

    def __init__(self, model_name: str = "cointegrated/rubert-tiny2", dimension: int = 312):
        self._model_name = model_name
        self._dim = dimension
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._fallback = LocalTFIDFEmbeddingProvider(dimension=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        try:
            from transformers import AutoTokenizer, AutoModel

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
            self._model.eval()
            self._loaded = True
            return True
        except Exception as e:
            logger.info(f"LocalHF model '{self._model_name}' not available ({e}). Using LocalTFIDF fallback.")
            return False

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._ensure_loaded():
            return self._fallback.embed_batch(texts)

        try:
            import torch

            inputs = self._tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Mean pooling
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = torch.sum(outputs.last_hidden_state * attention_mask, dim=1) / torch.clamp(
                    attention_mask.sum(dim=1), min=1e-9
                )
                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.warning(f"HF embedding error: {e}. Falling back to TFIDF.")
            return self._fallback.embed_batch(texts)


class GigaChatEmbeddingProvider(BaseEmbeddingProvider):
    """
    GigaChat remote embedding provider calling /api/v1/embeddings.
    Equipped with an automatic circuit breaker: if HTTP 402, 401, 429, or network error occurs,
    it automatically falls back to LocalTFIDFEmbeddingProvider with zero user intervention.
    """

    def __init__(self, model: str = "Embeddings", fallback_dimension: int = 128):
        self._model = model
        self._fallback = LocalTFIDFEmbeddingProvider(dimension=fallback_dimension)
        self._circuit_broken = False
        self._warned = False
        self._client = None
        try:
            from core.gigachat_client import gigachat
            self._client = gigachat
        except ImportError:
            self._circuit_broken = True

    @property
    def dimension(self) -> int:
        if self._circuit_broken:
            return self._fallback.dimension
        # Standard GigaChat embeddings dimension
        return 1024

    @property
    def is_fallback_active(self) -> bool:
        return self._circuit_broken

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._circuit_broken:
            return self._fallback.embed_batch(texts)

        token = self._client._get_token() if self._client else None
        if not token:
            self._trigger_fallback("No valid GigaChat access token available")
            return self._fallback.embed_batch(texts)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
        }
        payload = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            "https://gigachat.devices.sberbank.ru/api/v1/embeddings",
            data=payload,
            headers=headers
        )

        try:
            with urllib.request.urlopen(req, context=self._client._ssl_ctx, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("data", [])
                vectors = [item["embedding"] for item in items]
                if len(vectors) == len(texts):
                    return vectors
                else:
                    self._trigger_fallback(f"Mismatched embedding count: got {len(vectors)}, expected {len(texts)}")
                    return self._fallback.embed_batch(texts)
        except urllib.error.HTTPError as e:
            if e.code == 402:
                self._trigger_fallback("GigaChat Embeddings returned HTTP 402 Payment Required (free scope lacks embedding credits)")
            else:
                self._trigger_fallback(f"GigaChat Embeddings HTTP Error: {e.code} {e.reason}")
            return self._fallback.embed_batch(texts)
        except Exception as e:
            self._trigger_fallback(f"GigaChat Embeddings network/request exception: {e}")
            return self._fallback.embed_batch(texts)

    def _trigger_fallback(self, reason: str) -> None:
        self._circuit_broken = True
        if not self._warned:
            logger.warning(f"[RAG CIRCUIT-BREAKER] {reason}. Gracefully fallen back to LocalTFIDFEmbeddingProvider.")
            self._warned = True


class LocalFastEmbedProvider(BaseEmbeddingProvider):
    """
    Local CPU semantic embedding provider powered by fastembed (quantized ONNX models).
    Default model: 'BAAI/bge-small-en-v1.5' (384 dimensions).
    Equipped with graceful fallback to LocalTFIDFEmbeddingProvider(dimension=384)
    if fastembed is not installed, model weights are not downloaded, or inference fails.
    Guarantees dimension is strictly 384 under both fastembed and fallback, avoiding
    dimension mismatch errors in SQLiteVectorStore.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        local_files_only: bool = True,
        fallback_dimension: int = 384,
        **kwargs
    ):
        self._model_name = model_name
        self._dim = dimension
        self._cache_dir = cache_dir
        self._threads = threads
        self._local_files_only = local_files_only
        self._fallback = LocalTFIDFEmbeddingProvider(dimension=fallback_dimension)
        self._model = None
        self._is_loaded = False
        self._fallback_active = False

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_fallback_active(self) -> bool:
        return self._fallback_active

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> bool:
        """
        Dynamically loads fastembed.TextEmbedding on demand.
        Falls back to LocalTFIDF on ImportError, ModuleNotFoundError, or init exception.
        """
        if self._is_loaded:
            return not self._fallback_active
        try:
            from fastembed import TextEmbedding

            kwargs = {}
            if self._cache_dir:
                kwargs["cache_dir"] = self._cache_dir
            if self._threads:
                kwargs["threads"] = self._threads
            if self._local_files_only:
                kwargs["local_files_only"] = True

            self._model = TextEmbedding(model_name=self._model_name, **kwargs)
            self._is_loaded = True
            self._fallback_active = False
            return True
        except (ImportError, ModuleNotFoundError) as e:
            logger.info(
                f"fastembed package not installed ({e}). Gracefully falling back to LocalTFIDF (dim={self._fallback.dimension})."
            )
            self._fallback_active = True
            self._is_loaded = True
            return False
        except Exception as e:
            logger.info(
                f"fastembed initialization error for '{self._model_name}' ({e}). Gracefully falling back to LocalTFIDF (dim={self._fallback.dimension})."
            )
            self._fallback_active = True
            self._is_loaded = True
            return False

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if not self._ensure_loaded():
            return self._fallback.embed_batch(texts)

        try:
            embeddings_gen = self._model.embed(texts)
            results: List[List[float]] = []
            for vec in embeddings_gen:
                row = np.asarray(vec, dtype=np.float32)
                # Pad or truncate to self._dim
                if len(row) < self._dim:
                    padded = np.zeros(self._dim, dtype=np.float32)
                    padded[:len(row)] = row
                    row = padded
                elif len(row) > self._dim:
                    row = row[:self._dim]

                # L2 normalize
                norm = np.linalg.norm(row)
                if norm > 1e-12:
                    row = row / norm
                results.append(row.tolist())

            if len(results) == len(texts):
                return results

            self._fallback_active = True
            return self._fallback.embed_batch(texts)
        except Exception as e:
            logger.warning(f"fastembed inference error: {e}. Gracefully falling back to LocalTFIDF.")
            self._fallback_active = True
            return self._fallback.embed_batch(texts)


def get_embedding_provider(
    name: str = "local_tfidf",
    dimension: int = 128,
    **kwargs
) -> BaseEmbeddingProvider:
    """
    Factory function returning the configured embedding provider.
    Supported backends: 'local_tfidf', 'local_fastembed', 'local_hf', 'gigachat'.
    """
    norm_name = name.lower().strip()
    if norm_name in ("local_fastembed", "fastembed", "onnx", "local_onnx"):
        dim = kwargs.get("dimension") or (384 if dimension == 128 else dimension)
        model_name = kwargs.get("model_name", "BAAI/bge-small-en-v1.5")
        return LocalFastEmbedProvider(model_name=model_name, dimension=dim, **kwargs)
    elif norm_name in ("local_tfidf", "tfidf"):
        return LocalTFIDFEmbeddingProvider(dimension=dimension)
    elif norm_name in ("local_hf", "hf", "transformers"):
        model_name = kwargs.get("model_name", "cointegrated/rubert-tiny2")
        return LocalHFEmbeddingProvider(model_name=model_name, dimension=kwargs.get("dimension", 312))
    elif norm_name in ("gigachat", "sber"):
        return GigaChatEmbeddingProvider(model=kwargs.get("model", "Embeddings"), fallback_dimension=dimension)
    else:
        logger.warning(f"Unknown embedding backend '{name}', defaulting to LocalTFIDFEmbeddingProvider.")
        return LocalTFIDFEmbeddingProvider(dimension=dimension)
