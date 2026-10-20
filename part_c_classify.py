
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import mock_llm

ROOT = Path(__file__).resolve().parent
RAW_TICKETS = ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET = 6000

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("phone", re.compile(r"\b[6-9]\d{9}\b")),
    ("order_id", re.compile(r"ORD-\d{6}")),
]

ZERO_SHOT_PROMPT = """
CLASSIFY customer support tickets.
Choose exactly one category from: billing, delivery, technical, fraud.
Return JSON only, using exactly this schema:
{"category":"...", "reason":"..."}
The category must be one of the four allowed categories.
The reason should briefly explain the classification.
""".strip()

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
    """Redact PII in the assignment's required order."""
    redacted = text
    for _, pattern in PII_PATTERNS:
        redacted = pattern.sub(
            {
                "email": "[EMAIL]",
                "phone": "[PHONE]",
                "order_id": "[ORDER_ID]",
            }[_],
            redacted,
        )
    return redacted


def has_remaining_pii(text):
    return any(pattern.search(text) for _, pattern in PII_PATTERNS)


def parse_response(content):
    """Return (category, status) after validating model output."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (content or "").lower())
        found = [word for word in words if word in CATEGORIES]
        if len(found) == 1:
            return found[0], "classified_keyword"
        return "", "flagged_invalid_output"

    # A valid JSON response with an unsupported category is invalid.
    if (
        isinstance(data, dict)
        and isinstance(data.get("category"), str)
        and data["category"] in CATEGORIES
    ):
        return data["category"], "classified_json"

    return "", "flagged_invalid_output"


def run_classification(prompt, run_name, token_budget=None):
    """Run one independent batch; token budget applies across the batch."""
    mock_llm.reset()

    tickets = pd.read_csv(RAW_TICKETS, keep_default_na=False)
    results = []
    total_tokens = 0
    retry_events = 0

    for _, row in tickets.iterrows():
        original = row["text"]
        redacted = redact_text(original)

        result = {
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "category": "",
            "reason": "",
            "status": "",
            "tokens_used": 0,
        }

        # Guardrails run before any model call.
        if not original.strip():
            result["status"] = "skipped_empty"
            results.append(result)
            continue

        if mock_llm.est_tokens(original) > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            results.append(result)
            continue

        # The user message contains ONLY the redacted ticket text.
        assert not has_remaining_pii(redacted), (
            f"PII remains in ticket {row['ticket_id']}"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": redacted},
        ]
        assert messages[-1]["content"] == redacted
        assert not has_remaining_pii(messages[-1]["content"])

        retry_number = 0
        completed = False

        while True:
            # Do not call the model once the accumulated budget is reached.
            if token_budget is not None and total_tokens >= token_budget:
                result["status"] = "budget_exhausted"
                completed = True
                break

            try:
                response = mock_llm.chat(messages)
                content = response.choices[0].message.content or ""
                used = response.usage.total_tokens
                total_tokens += used
                result["tokens_used"] = used

                category, status = parse_response(content)
                result["category"] = category
                result["status"] = status

                if status == "classified_json":
                    try:
                        result["reason"] = json.loads(content).get(
                            "reason", ""
                        )
                    except (json.JSONDecodeError, AttributeError):
                        result["reason"] = ""

                completed = True
                break

            except mock_llm.TransientLLMError:
                if retry_number >= MAX_RETRIES:
                    result["status"] = "flagged_invalid_output"
                    completed = True
                    break

                wait = (
                    BASE_DELAY
                    * (2 ** retry_number)
                    * random.uniform(0.9, 1.15)
                )
                retry_events += 1
                print(
                    f"[{run_name}] Retry: ticket={row['ticket_id']}, "
                    f"retry_number={retry_number}, wait={wait:.4f}s"
                )
                time.sleep(wait)
                retry_number += 1

        if completed:
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / f"triage_{run_name}.csv", index=False
    )

    classified = results_df[
        results_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]
    correct = (
        classified["category"] == classified["true_category"]
    )
    accuracy = correct.mean() * 100 if len(classified) else 0

    print(f"\n--- {run_name.upper()} RESULTS ---")
    print("Tickets processed:", len(results_df))
    print("\nStatus counts:")
    print(results_df["status"].value_counts().to_string())
    print("Total response tokens:", total_tokens)
    print("Classified tickets:", len(classified))
    print("Correct classifications:", int(correct.sum()))
    print(f"Accuracy: {accuracy:.2f}%")
    print(
        "Tokens per classified ticket:",
        round(total_tokens / len(classified), 2)
        if len(classified) else "N/A",
    )
    print("Transient retry events:", retry_events)

    if run_name == "zero_shot" and len(classified):
        print("\nZero-shot confusion table:")
        print(
            pd.crosstab(
                classified["true_category"],
                classified["category"],
                rownames=["True category"],
                colnames=["Predicted category"],
            ).to_string()
        )

    return results_df


def main():
    # Full independent evaluation runs.
    zero_shot = run_classification(
        ZERO_SHOT_PROMPT, "zero_shot"
    )
    few_shot = run_classification(
        FEW_SHOT_PROMPT, "few_shot"
    )

    print("\n--- PROMPT COMPARISON ---")
    for name, frame in [
        ("Zero-shot", zero_shot),
        ("Few-shot", few_shot),
    ]:
        classified = frame[
            frame["status"].isin(
                ["classified_json", "classified_keyword"]
            )
        ]
        accuracy = (
            (classified["category"] == classified["true_category"]).mean()
            * 100
            if len(classified) else 0
        )
        print(
            f"{name}: classified={len(classified)}, "
            f"accuracy={accuracy:.2f}%, "
            f"tokens={frame['tokens_used'].sum()}"
        )

    # Independent zero-shot run with a token budget.
    budget_results = run_classification(
        ZERO_SHOT_PROMPT,
        "budget_demo",
        token_budget=TOKEN_BUDGET,
    )
    budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
    budget_results.to_csv(budget_path, index=False)

    print("\nBudget limit:", TOKEN_BUDGET)
    print("Budget demo saved to:", budget_path)


if __name__ == "__main__":
    main()
