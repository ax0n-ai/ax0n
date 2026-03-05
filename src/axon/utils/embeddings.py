from collections import OrderedDict
from typing import List, Optional
import structlog

logger = structlog.get_logger(__name__)

_st_available = False
_SentenceTransformer = None
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _st_available = True
except Exception:
    pass


class EmbeddingProvider:
    """
    Semantic similarity provider.

    Uses sentence-transformers when installed for high-quality cosine similarity.
    Falls back to weighted n-gram Jaccard when the library is not available.
    """

    _DEFAULT_MODEL = "all-MiniLM-L6-v2"
    _DEFAULT_CACHE_SIZE = 512

    def __init__(self, model_name: Optional[str] = None, cache_size: int = _DEFAULT_CACHE_SIZE):
        self._model = None
        self._model_name = model_name or self._DEFAULT_MODEL
        self._cache_size = cache_size
        self._embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger = logger.bind(component="embedding_provider")

        if _st_available:
            try:
                self._model = _SentenceTransformer(self._model_name)
                self.logger.info(
                    "Sentence-transformers loaded",
                    model=self._model_name,
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to load sentence-transformers model, falling back to n-gram",
                    error=str(e),
                )
                self._model = None
        else:
            self.logger.info("sentence-transformers not installed, using n-gram fallback")

    @property
    def uses_embeddings(self) -> bool:
        """Whether this provider uses real embeddings (vs n-gram fallback)."""
        return self._model is not None

    @property
    def cache_stats(self) -> dict:
        """Return embedding cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "size": len(self._embedding_cache),
            "max_size": self._cache_size,
        }

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache, updating LRU order."""
        if text in self._embedding_cache:
            self._cache_hits += 1
            self._embedding_cache.move_to_end(text)
            return self._embedding_cache[text]
        self._cache_misses += 1
        return None

    def _put_cached_embedding(self, text: str, embedding: List[float]) -> None:
        """Store embedding in LRU cache."""
        self._embedding_cache[text] = embedding
        self._embedding_cache.move_to_end(text)
        while len(self._embedding_cache) > self._cache_size:
            self._embedding_cache.popitem(last=False)

    def encode(self, text: str) -> Optional[List[float]]:
        """
        Encode text into a dense embedding vector.

        Returns None when sentence-transformers is not available.
        Uses LRU cache to avoid recomputing embeddings for repeated texts.
        """
        if self._model is None:
            return None

        cached = self._get_cached_embedding(text)
        if cached is not None:
            return cached

        try:
            embedding = self._model.encode(text, convert_to_numpy=True).tolist()
            self._put_cached_embedding(text, embedding)
            return embedding
        except Exception as e:
            self.logger.warning("Embedding encode failed", error=str(e))
            return None

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Uses cosine similarity on sentence-transformers embeddings when available,
        otherwise falls back to weighted n-gram Jaccard similarity.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if self._model is not None:
            return self._cosine_similarity(text1, text2)
        return self._ngram_similarity(text1, text2)


    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity using sentence-transformers with embedding cache."""
        try:
            import numpy as np

            emb1 = self._get_cached_embedding(text1)
            emb2 = self._get_cached_embedding(text2)

            if emb1 is not None and emb2 is not None:
                a, b = np.array(emb1), np.array(emb2)
            elif emb1 is not None:
                a = np.array(emb1)
                b = self._model.encode(text2, convert_to_numpy=True)
                self._put_cached_embedding(text2, b.tolist())
            elif emb2 is not None:
                a = self._model.encode(text1, convert_to_numpy=True)
                self._put_cached_embedding(text1, a.tolist())
                b = np.array(emb2)
            else:
                embeddings = self._model.encode([text1, text2], convert_to_numpy=True)
                a, b = embeddings[0], embeddings[1]
                self._put_cached_embedding(text1, a.tolist())
                self._put_cached_embedding(text2, b.tolist())

            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            cosine = float(np.dot(a, b) / (norm_a * norm_b))
            return max(0.0, min(1.0, cosine))
        except Exception as e:
            self.logger.warning("Cosine similarity failed, using n-gram fallback", error=str(e))
            return self._ngram_similarity(text1, text2)


    @staticmethod
    def _ngram_similarity(text1: str, text2: str) -> float:
        """
        Weighted n-gram Jaccard similarity (unigrams 30%, bigrams 40%, trigrams 30%).
        """
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        if not words1 or not words2:
            return 0.0

        def _jaccard(w1: List[str], w2: List[str], n: int) -> float:
            if len(w1) < n or len(w2) < n:
                return 0.0
            s1 = set(tuple(w1[i:i + n]) for i in range(len(w1) - n + 1))
            s2 = set(tuple(w2[i:i + n]) for i in range(len(w2) - n + 1))
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)

        uni = _jaccard(words1, words2, 1)
        bi = _jaccard(words1, words2, 2)
        tri = _jaccard(words1, words2, 3)

        return 0.3 * uni + 0.4 * bi + 0.3 * tri
