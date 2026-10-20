
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import mock_llm

ROOT = Path(__file__).resolve().parent
TICKETS_PATH = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_PATH = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_INPUT_TOKENS = 400
TOKEN_BUDGET = 100000
BASE_DELAY = 0.05
MAX_RETRIES = 3

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

PII_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\b[6-9]\d{9}\b"),
    re.compile(r"ORD-\d{6}"),
]

FEW_SHOT_PROMPT = """
CLASSIFY customer support tickets.
Choose exactly one category from: billing, delivery, technical, fraud.
Return JSON only, using exactly this schema:
{"category":"...", "reason":"..."}
The category must be one of the four allowed categories.
The reason should briefly explain the classification.

EXAMPLES:
Ticket: "I see an unauthorized transaction that I did not make."
Output: {"category":"fraud","reason":"Unauthorized transaction reported"}

Ticket: "My invoice shows an incorrect amount charged."
Output: {"category":"billing","reason":"Incorrect invoice amount"}

Ticket: "The app freezes when I try to log in."
Output: {"category":"technical","reason":"Application functionality problem"}
""".strip()


def redact_text(text):
    replacements = ["[EMAIL]", "[PHONE]", "[ORDER_ID]"]
    redacted = text
    for pattern, replacement in zip(PII_PATTERNS, replacements):
        redacted = pattern.sub(replacement, redacted)
    return redacted


def contains_pii(text):
    return any(pattern.search(text) for pattern in PII_PATTERNS)


def parse_response(content):
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (content or "").lower())
        found = [word for word in words if word in CATEGORIES]
        if len(found) == 1:
            return found[0], "classified_keyword"
        return "", "flagged_invalid_output"

    if (
        isinstance(data, dict)
        and isinstance(data.get("category"), str)
        and data["category"] in CATEGORIES
    ):
        return data["category"], "classified_json"

    return "", "flagged_invalid_output"


def classify_ticket(text, ticket_id, running_tokens):
    """Classify one eligible ticket; return result and updated token total."""
    redacted = redact_text(text)
    assert not contains_pii(redacted), (
        f"PII remains after redaction in {ticket_id}"
    )

    # The model receives only the redacted ticket as its user message.
    messages = [
        {"role": "system", "content": FEW_SHOT_PROMPT},
        {"role": "user", "content": redacted},
    ]
    assert messages[-1]["content"] == redacted
    assert not contains_pii(messages[-1]["content"])

    retries = 0

    while True:
        if running_tokens >= TOKEN_BUDGET:
            return {
                "category": "",
                "status": "budget_exhausted",
                "tokens_used": 0,
            }, running_tokens

        try:
            response = mock_llm.chat(messages)
            content = response.choices[0].message.content or ""
            used = response.usage.total_tokens
            running_tokens += used

            category, status = parse_response(content)

            return {
                "category": category,
                "status": status,
                "tokens_used": used,
            }, running_tokens

        except mock_llm.TransientLLMError:
            if retries >= MAX_RETRIES:
                return {
                    "category": "",
                    "status": "flagged_invalid_output",
                    "tokens_used": 0,
                }, running_tokens

            wait = (
                BASE_DELAY * (2 ** retries) * random.uniform(0.9, 1.15)
            )
            print(
                f"[final] Retry: ticket={ticket_id}, "
                f"retry_number={retries}, wait={wait:.4f}s"
            )
            time.sleep(wait)
            retries += 1


def main():
    tickets = pd.read_csv(TICKETS_PATH, keep_default_na=False)
    segments = pd.read_csv(SEGMENTS_PATH, keep_default_na=False)

    required = {
        "ticket_id", "customer_id", "text", "true_category"
    }
    assert required.issubset(tickets.columns)
    assert {
        "customer_id", "segment_name", "is_premium"
    }.issubset(segments.columns)
    assert tickets["ticket_id"].is_unique
    assert segments["customer_id"].is_unique

    # Parse premium values robustly from the CSV.
    segments["is_premium"] = (
        segments["is_premium"].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )
    segment_lookup = segments.set_index("customer_id").to_dict("index")

    mock_llm.reset()
    running_tokens = 0
    output_rows = []

    for _, ticket in tickets.iterrows():
        text = ticket["text"]
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]

        category = ""
        tokens_used = 0

        # These guardrails precede all model calls.
        if not text.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        else:
            result, running_tokens = classify_ticket(
                text, ticket_id, running_tokens
            )
            category = result["category"]
            status = result["status"]
            tokens_used = result["tokens_used"]

        customer = segment_lookup.get(customer_id, {})
        segment_name = customer.get("segment_name", "")
        if pd.isna(segment_name):
            segment_name = ""

        # Apply human review first, then fraud, then premium routing.
        if status in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        }:
            route = "human_review"
            priority = "NORMAL"
            category = ""
        elif category == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif customer.get("is_premium", False):
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        output_rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
        })

    final_df = pd.DataFrame(
        output_rows,
        columns=[
            "ticket_id", "customer_id", "status", "category",
            "segment_name", "priority", "route", "tokens_used",
        ],
    )

    output_path = OUTPUT_DIR / "triage_results.csv"
    final_df.to_csv(output_path, index=False)

    assert len(final_df) == len(tickets) == 164
    assert final_df["ticket_id"].is_unique
    assert final_df["category"].eq("").sum() >= 17

    print("\nFINAL PART C OUTPUT")
    print("Rows:", len(final_df))
    print("\nStatus counts:")
    print(final_df["status"].value_counts().to_string())
    print("\nRouting counts:")
    print(final_df["route"].value_counts().to_string())
    print("\nPriority counts:")
    print(final_df["priority"].value_counts().to_string())
    print("\nTotal response tokens:", int(final_df["tokens_used"].sum()))
    print("\nOutput saved to:", output_path)


if __name__ == "__main__":
    main()
