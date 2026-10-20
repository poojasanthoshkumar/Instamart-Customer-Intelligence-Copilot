
import re
import pandas as pd
from pathlib import Path
import mock_llm

# Project paths
ROOT = Path(__file__).resolve().parent
RAW_TICKETS = ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Assignment constants
MAX_INPUT_TOKENS = 400
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

# Apply the exact required patterns in the specified order.
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact_text(text):
    """Return redacted text and substitution counts for each PII type."""
    redacted = text
    counts = {}

    for name, pattern, replacement in PII_PATTERNS:
        redacted, count = pattern.subn(replacement, redacted)
        counts[name] = count

    return redacted, counts


def remaining_pii(text):
    """Count any sensitive patterns that remain after redaction."""
    return {
        name: len(pattern.findall(text))
        for name, pattern, _ in PII_PATTERNS
    }


def inspect_tickets():
    # Preserve empty strings rather than converting them to NaN.
    df = pd.read_csv(RAW_TICKETS, keep_default_na=False)

    assert len(df) == 164, f"Expected 164 tickets, found {len(df)}"
    assert {"ticket_id", "customer_id", "order_id", "text",
            "true_category"}.issubset(df.columns)

    # Apply redaction to all records for the audit.
    redacted_texts = []
    totals = {name: 0 for name, _, _ in PII_PATTERNS}

    for text in df["text"]:
        redacted, counts = redact_text(text)
        redacted_texts.append(redacted)

        for name, count in counts.items():
            totals[name] += count

    # Verify no configured PII pattern remains in any redacted text.
    for text in redacted_texts:
        assert sum(remaining_pii(text).values()) == 0, (
            "PII remains after redaction"
        )

    df["redacted_text"] = redacted_texts

    # Guardrails: determine eligibility before any model call.
    df["estimated_tokens"] = df["text"].map(mock_llm.est_tokens)

    df["input_status"] = "eligible"
    empty_mask = df["text"].str.strip().eq("")
    long_mask = (
        ~empty_mask
        & (df["estimated_tokens"] > MAX_INPUT_TOKENS)
    )

    df.loc[empty_mask, "input_status"] = "skipped_empty"
    df.loc[long_mask, "input_status"] = "flagged_long"

    print("Total tickets:", len(df))
    print("\nInput status counts:")
    print(df["input_status"].value_counts().to_string())

    print("\nPII substitution counts:")
    for name, count in totals.items():
        print(f"{name}: {count}")
    print("Total substitutions:", sum(totals.values()))
    print("Remaining PII matches:", sum(
        sum(remaining_pii(text).values()) for text in redacted_texts
    ))

    print("\nEligible for classification:",
          int((df["input_status"] == "eligible").sum()))

    # Save this initial audit for inspection.
    audit_path = OUTPUT_DIR / "triage_input_audit.csv"
    df[[
        "ticket_id", "customer_id", "order_id", "true_category",
        "input_status", "estimated_tokens", "redacted_text"
    ]].to_csv(audit_path, index=False)

    print("\nInput audit saved to:", audit_path)


if __name__ == "__main__":
    inspect_tickets()
