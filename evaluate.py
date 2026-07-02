"""
Evaluation script — replays the 10 sample conversations against the
running /chat endpoint and measures:

1. Schema compliance (every response has reply, recommendations, end_of_conversation)
2. Catalog compliance (all recommended URLs exist in the catalog)
3. Recall@10 per trace and mean across all traces
4. Turn count adherence (max 8 turns)
"""

import json
import re
import sys
import requests
from pathlib import Path

from catalog import load_catalog, get_catalog_by_url

BASE_URL = "http://localhost:8000"
CONVERSATIONS_DIR = Path(__file__).parent / "sample_conversations" / "GenAI_SampleConversations"


def parse_conversation(md_path: Path) -> dict:
    """
    Parse a sample conversation markdown file.

    Returns:
        {
            "user_messages": ["msg1", "msg2", ...],
            "expected_recommendations": [{"name": "...", "url": "..."}, ...]
        }
    """
    text = md_path.read_text(encoding="utf-8")

    # Extract user messages (lines starting with >)
    user_messages = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("> "):
            user_messages.append(line[2:])

    # Extract expected recommendations from the LAST table in the file
    # Tables have rows like: | 1 | Name | Type | Keys | Duration | Languages | <URL> |
    url_pattern = re.compile(r"<(https://www\.shl\.com/[^>]+)>")
    name_pattern = re.compile(
        r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|"
    )

    # Find the last table block (the final recommendation)
    tables = []
    current_table = []
    in_table = False

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and "---" not in stripped and "#" not in stripped:
            in_table = True
            current_table.append(stripped)
        else:
            if in_table:
                tables.append(current_table)
                current_table = []
                in_table = False

    if current_table:
        tables.append(current_table)

    expected_recs = []
    if tables:
        last_table = tables[-1]
        for row in last_table:
            urls = url_pattern.findall(row)
            names = name_pattern.findall(row)
            if urls:
                rec = {"url": urls[0]}
                if names:
                    rec["name"] = names[0].strip()
                expected_recs.append(rec)

    return {
        "user_messages": user_messages,
        "expected_recommendations": expected_recs,
    }


def replay_conversation(user_messages: list[str]) -> list[dict]:
    """
    Replay a conversation against the /chat endpoint.
    Returns the list of all responses.
    """
    messages = []
    responses = []

    for user_msg in user_messages:
        messages.append({"role": "user", "content": user_msg})

        try:
            resp = requests.post(
                f"{BASE_URL}/chat",
                json={"messages": messages},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ❌ API error: {e}")
            data = {"reply": "", "recommendations": [], "end_of_conversation": False}

        responses.append(data)

        import time
        time.sleep(13)  # Respect free tier rate limits (< 5 RPM)
        # Add assistant response to history for next turn
        messages.append({"role": "assistant", "content": data.get("reply", "")})

    return responses


def compute_recall_at_k(predicted_urls: list[str], expected_urls: list[str], k: int = 10) -> float:
    """Compute Recall@K."""
    if not expected_urls:
        return 1.0  # No expected = trivially correct

    predicted_set = set(predicted_urls[:k])
    expected_set = set(expected_urls)

    hits = len(predicted_set & expected_set)
    return hits / len(expected_set)


def check_schema_compliance(response: dict) -> list[str]:
    """Check if a response complies with the required schema."""
    errors = []
    if "reply" not in response:
        errors.append("Missing 'reply' field")
    if "recommendations" not in response:
        errors.append("Missing 'recommendations' field")
    if "end_of_conversation" not in response:
        errors.append("Missing 'end_of_conversation' field")

    recs = response.get("recommendations", [])
    if not isinstance(recs, list):
        errors.append("'recommendations' is not a list")
    else:
        for i, rec in enumerate(recs):
            if "name" not in rec:
                errors.append(f"Rec {i}: missing 'name'")
            if "url" not in rec:
                errors.append(f"Rec {i}: missing 'url'")
            if "test_type" not in rec:
                errors.append(f"Rec {i}: missing 'test_type'")

    return errors


def main():
    # Check health
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        health.raise_for_status()
        print(f"✅ Health check: {health.json()}\n")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("   Make sure the server is running: uvicorn main:app --port 8000")
        sys.exit(1)

    catalog_urls = set(get_catalog_by_url().keys())

    # Process each conversation
    conv_files = sorted(CONVERSATIONS_DIR.glob("C*.md"))
    if not conv_files:
        print("❌ No conversation files found")
        sys.exit(1)

    results = []
    for conv_file in conv_files:
        conv_name = conv_file.stem
        print(f"{'='*60}")
        print(f"📋 {conv_name}: {conv_file.name}")
        print(f"{'='*60}")

        parsed = parse_conversation(conv_file)
        print(f"   User messages: {len(parsed['user_messages'])}")
        print(f"   Expected recs: {len(parsed['expected_recommendations'])}")

        # Replay
        responses = replay_conversation(parsed["user_messages"])

        # Schema compliance
        all_schema_ok = True
        for i, resp in enumerate(responses):
            errors = check_schema_compliance(resp)
            if errors:
                all_schema_ok = False
                print(f"   ❌ Turn {i+1} schema errors: {errors}")

        # Catalog compliance
        all_catalog_ok = True
        for i, resp in enumerate(responses):
            for rec in resp.get("recommendations", []):
                url = rec.get("url", "")
                if url and url not in catalog_urls:
                    all_catalog_ok = False
                    print(f"   ❌ Turn {i+1}: URL not in catalog: {url}")

        # Get final recommendations
        final_recs = []
        for resp in reversed(responses):
            if resp.get("recommendations"):
                final_recs = resp["recommendations"]
                break

        predicted_urls = [rec["url"] for rec in final_recs]
        expected_urls = [rec["url"] for rec in parsed["expected_recommendations"]]

        recall = compute_recall_at_k(predicted_urls, expected_urls, k=10)

        total_turns = len(parsed["user_messages"]) * 2  # user + assistant
        turn_ok = total_turns <= 8

        print(f"   Schema compliance: {'✅' if all_schema_ok else '❌'}")
        print(f"   Catalog compliance: {'✅' if all_catalog_ok else '❌'}")
        print(f"   Turn count: {total_turns}/8 {'✅' if turn_ok else '❌'}")
        print(f"   Recall@10: {recall:.2f}")
        print(f"   Final recs: {[r['name'] for r in final_recs]}")
        print()

        results.append({
            "conversation": conv_name,
            "schema_ok": all_schema_ok,
            "catalog_ok": all_catalog_ok,
            "turn_ok": turn_ok,
            "recall_at_10": recall,
            "num_predicted": len(final_recs),
            "num_expected": len(expected_urls),
        })

    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Conv':<8} {'Schema':<8} {'Catalog':<9} {'Turns':<7} {'Recall@10':<10}")
    print(f"{'-'*8} {'-'*8} {'-'*9} {'-'*7} {'-'*10}")

    total_recall = 0
    for r in results:
        schema = "✅" if r["schema_ok"] else "❌"
        catalog = "✅" if r["catalog_ok"] else "❌"
        turns = "✅" if r["turn_ok"] else "❌"
        total_recall += r["recall_at_10"]
        print(f"{r['conversation']:<8} {schema:<8} {catalog:<9} {turns:<7} {r['recall_at_10']:<10.2f}")

    mean_recall = total_recall / len(results) if results else 0
    print(f"\nMean Recall@10: {mean_recall:.4f}")


if __name__ == "__main__":
    main()
