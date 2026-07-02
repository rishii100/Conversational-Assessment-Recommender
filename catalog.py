"""
Catalog module — loads, cleans, and indexes the SHL product catalog.

Responsibilities:
- Parse the JSON catalog (handling invalid control characters)
- Derive test_type codes from the 'keys' field
- Provide lookup and search helpers
"""

import json
from pathlib import Path
from typing import Optional

# Mapping from catalog 'keys' values to single-letter test type codes
# Derived by studying the sample conversations (C1-C10)
KEYS_TO_TEST_TYPE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

CATALOG_PATH = Path(__file__).parent / "shl_product_catalog.json"

# Module-level cache
_catalog: list[dict] | None = None
_catalog_by_url: dict[str, dict] | None = None


def _derive_test_type(keys: list[str]) -> str:
    """Convert a list of key categories to comma-separated test type codes."""
    codes = []
    for key in keys:
        code = KEYS_TO_TEST_TYPE.get(key)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"


def load_catalog() -> list[dict]:
    """Load and enrich the SHL product catalog from JSON."""
    global _catalog, _catalog_by_url

    if _catalog is not None:
        return _catalog

    raw = CATALOG_PATH.read_text(encoding="utf-8")
    data = json.loads(raw, strict=False)

    enriched = []
    for item in data:
        # Derive test_type from keys
        item["test_type"] = _derive_test_type(item.get("keys", []))

        # Build a combined text blob for embedding/search
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("keys", [])),
            " ".join(item.get("job_levels", [])),
        ]
        item["search_text"] = " ".join(p for p in parts if p)

        enriched.append(item)

    _catalog = enriched
    _catalog_by_url = {item["link"]: item for item in _catalog}
    return _catalog


def get_catalog_by_url() -> dict[str, dict]:
    """Return a dict mapping URL -> catalog item for validation."""
    if _catalog_by_url is None:
        load_catalog()
    return _catalog_by_url  # type: ignore


def is_valid_catalog_url(url: str) -> bool:
    """Check if a URL exists in the catalog."""
    return url in get_catalog_by_url()


def get_assessment_by_url(url: str) -> Optional[dict]:
    """Look up a catalog item by its URL."""
    return get_catalog_by_url().get(url)


def format_assessment_for_context(item: dict) -> str:
    """Format a single catalog item as a compact text block for LLM context."""
    languages = item.get("languages", [])
    lang_str = ", ".join(languages[:4])
    if len(languages) > 4:
        lang_str += f" (+{len(languages) - 4} more)"

    duration = item.get("duration", "") or "—"

    return (
        f"• {item['name']}\n"
        f"  URL: {item['link']}\n"
        f"  Test Type: {item['test_type']}\n"
        f"  Keys: {', '.join(item.get('keys', []))}\n"
        f"  Job Levels: {', '.join(item.get('job_levels', []))}\n"
        f"  Duration: {duration}\n"
        f"  Languages: {lang_str}\n"
        f"  Remote: {item.get('remote', '—')} | Adaptive: {item.get('adaptive', '—')}\n"
        f"  Description: {item.get('description', 'N/A')}\n"
    )
