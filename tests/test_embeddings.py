
import numpy as np
import pytest

from app.diagnostics.embeddings import embed_texts


def test_embed_texts_returns_normalized_vectors():
    embeddings = embed_texts(
        [
            "The customer cannot access their account.",
            "The user has a problem signing in.",
            "The weather is sunny today.",
        ]
    )

    assert embeddings.shape == (3, 384)
    assert np.isfinite(embeddings).all()

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_texts_rejects_empty_input():
    with pytest.raises(ValueError):
        embed_texts([])


def test_embed_texts_rejects_blank_documents():
    with pytest.raises(ValueError):
        embed_texts(["valid text", "   "])


def test_embed_texts_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        embed_texts(["some text"], batch_size=0)
