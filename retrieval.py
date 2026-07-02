"""
Retrieval module — lightweight semantic search over the SHL catalog 
using TF-IDF vectorization from scikit-learn.

Optimized for Vercel deployment (removes 5GB torch dependency).
"""

import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from catalog import load_catalog, format_assessment_for_context
from config import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

# Module-level singletons
_vectorizer: TfidfVectorizer | None = None
_tfidf_matrix = None
_catalog_items: list[dict] | None = None


def build_index() -> None:
    """Build the TF-IDF index from all catalog items."""
    global _vectorizer, _tfidf_matrix, _catalog_items

    _catalog_items = load_catalog()
    
    logger.info("Building TF-IDF index for %d catalog items...", len(_catalog_items))
    
    # We use a simple vectorizer with English stop words
    _vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2), # Use unigrams and bigrams for better keyword matching
        max_features=5000
    )
    
    texts = [item["search_text"] for item in _catalog_items]
    _tfidf_matrix = _vectorizer.fit_transform(texts)
    
    logger.info("TF-IDF index built successfully.")


def retrieve_relevant(query: str, top_k: int | None = None) -> list[dict]:
    """
    Retrieve the top-K most relevant catalog items for a given query.
    """
    if _tfidf_matrix is None or _catalog_items is None:
        build_index()

    if top_k is None:
        top_k = TOP_K_RETRIEVAL

    # Transform query to TF-IDF vector
    query_vec = _vectorizer.transform([query])
    
    # Calculate cosine similarity
    scores = cosine_similarity(query_vec, _tfidf_matrix).flatten()
    
    # Get top-K indices
    top_indices = scores.argsort()[::-1][: min(top_k, len(_catalog_items))]

    results = []
    for idx in top_indices:
        # Only return items with some similarity (score > 0)
        # but for RAG context, we usually want at least a few items
        item = _catalog_items[idx].copy()
        item["relevance_score"] = float(scores[idx])
        results.append(item)

    return results


def retrieve_context_block(query: str, top_k: int | None = None) -> str:
    """
    Retrieve relevant assessments and format them as a text block.
    """
    items = retrieve_relevant(query, top_k)
    if not items:
        return "No relevant assessments found in the catalog."

    lines = [f"=== {len(items)} RELEVANT ASSESSMENTS FROM SHL CATALOG ===\n"]
    for item in items:
        lines.append(format_assessment_for_context(item))
    return "\n".join(lines)
