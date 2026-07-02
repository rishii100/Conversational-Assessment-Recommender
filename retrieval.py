"""
Retrieval module — semantic search over the SHL catalog using
sentence-transformers embeddings and numpy dot product.

At startup, all 377 catalog items are embedded and indexed.
At query time, the user's latest message is embedded and the top-K
most similar assessments are returned for LLM context injection.
"""

import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from catalog import load_catalog, format_assessment_for_context
from config import EMBEDDING_MODEL, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

# Module-level singletons
_model: SentenceTransformer | None = None
_embeddings_matrix: np.ndarray | None = None
_catalog_items: list[dict] | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def build_index() -> None:
    """Build the embeddings matrix from all catalog items. Call once at startup."""
    global _embeddings_matrix, _catalog_items

    _catalog_items = load_catalog()

    if os.path.exists("embeddings.npy"):
        logger.info("Loading precomputed embeddings from embeddings.npy...")
        _embeddings_matrix = np.load("embeddings.npy")
        logger.info("Embeddings loaded: %d vectors", _embeddings_matrix.shape[0])
        return

    model = _get_model()
    # Create search texts for embedding
    texts = [item["search_text"] for item in _catalog_items]

    logger.info("Embedding %d catalog items...", len(texts))
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    _embeddings_matrix = np.array(embeddings, dtype="float32")
    logger.info("Embeddings built: %d vectors of dim %d", _embeddings_matrix.shape[0], _embeddings_matrix.shape[1])


def retrieve_relevant(query: str, top_k: int | None = None) -> list[dict]:
    """
    Retrieve the top-K most relevant catalog items for a given query.
    """
    if _embeddings_matrix is None or _catalog_items is None:
        build_index()

    if top_k is None:
        top_k = TOP_K_RETRIEVAL

    model = _get_model()
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec[0], dtype="float32")

    # Cosine similarity (dot product since normalized)
    scores = _embeddings_matrix @ query_vec

    # Get top-K indices
    top_indices = np.argsort(scores)[::-1][: min(top_k, len(_catalog_items))]

    results = []
    for idx in top_indices:
        item = _catalog_items[idx].copy()
        item["relevance_score"] = float(scores[idx])
        results.append(item)

    return results


def retrieve_context_block(query: str, top_k: int | None = None) -> str:
    """
    Retrieve relevant assessments and format them as a text block
    suitable for injection into the LLM prompt.
    """
    items = retrieve_relevant(query, top_k)
    if not items:
        return "No relevant assessments found in the catalog."

    lines = [f"=== {len(items)} RELEVANT ASSESSMENTS FROM SHL CATALOG ===\n"]
    for item in items:
        lines.append(format_assessment_for_context(item))
    return "\n".join(lines)
