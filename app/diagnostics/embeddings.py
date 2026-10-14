
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def load_embedding_model(
    model_name: str = DEFAULT_MODEL,
) -> SentenceTransformer:
    """Load and cache a local sentence-embedding model."""
    return SentenceTransformer(model_name, device="cpu")


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """Convert a list of text documents into normalized embeddings."""

    if not texts:
        raise ValueError("texts must contain at least one document")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every document must be a non-empty string")

    model = load_embedding_model(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise RuntimeError("Unexpected embedding output shape")

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embeddings contain non-finite values")

    return embeddings
