
"""evaluate_agent.py
Official scenario evaluation for Instamart refund agent.
"""

import csv
import math
import re
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from policy_retriever import load_policy_chunks, build_retriever
from refund_agent import run_refund_agent

DB_FILE = BASE_DIR / "data" / "instamart.db"
SCENARIO_FILE = BASE_DIR / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_FILE = (
    BASE_DIR / "outputs" / "part_d" / "agent_evaluation_results.csv"
)


def get_order_details(message):
    """Retrieve order details for scenario validation."""

    match = re.search(r"ORD-\d{6}", message)

    if not match:
        return None

    conn = sqlite3.connect(DB_FILE)

    try:
        row = conn.execute(
            """
            SELECT customer_id, order_amount,
                   delivery_minutes, status
            FROM orders
            WHERE order_id = ?
            """,
            (match.group(0),),
        ).fetchone()

        return row

    finally:
        conn.close()


def parse_actual_response(result):
    """Extract outcome and amount from the agent response."""

    result = (result or "").strip()
    outcome = result.split(":", 1)[0].strip().upper()
    amount = 0.0

    if outcome == "REFUND":

        match = re.search(
            r"REFUND:\s*INR\s*([\d.]+)",
            result,
            re.IGNORECASE,
        )

        if match:
            amount = float(match.group(1))
        else:
            outcome = "INVALID_REFUND_RESPONSE"

    elif outcome == "ESCALATE":

        match = re.search(
            r"ESCALATE:\s*refund of INR\s*([\d.]+)",
            result,
            re.IGNORECASE,
        )

        if match:
            amount = float(match.group(1))
        else:
            outcome = "INVALID_ESCALATION_RESPONSE"

    elif outcome == "NEEDS_INFO":
        outcome = "FALLBACK"

    return outcome, amount


def main():

    print("=" * 65)
    print("INSTAMART REFUND AGENT - OFFICIAL SCENARIO EVALUATION")
    print("=" * 65)

    print("\nLoading policy information...")

    chunks = load_policy_chunks()

    if not chunks:
        raise RuntimeError("No policy chunks found.")

    vectorizer, document_vectors = build_retriever(chunks)

    print(f"Policy chunks loaded: {len(chunks)}")

    if not SCENARIO_FILE.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {SCENARIO_FILE}"
        )

    with open(
        SCENARIO_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        scenarios = list(csv.DictReader(file))

    if not scenarios:
        raise RuntimeError("No evaluation scenarios found.")

    print(f"Total official scenarios: {len(scenarios)}")

    results = []
    passed = 0

    for index, scenario in enumerate(scenarios, start=1):

        scenario_id = scenario["scenario_id"]
        customer_id = scenario["customer_id"]
        message = scenario["message"]

        expected_outcome = (
            scenario["expected_outcome"].strip().upper()
        )

        expected_amount = float(
            scenario["expected_refund_inr"]
        )

        print("\n" + "=" * 65)
        print(f"TEST {index}: {scenario_id}")
        print("=" * 65)

        print(f"Customer: {customer_id}")
        print(f"Expected outcome: {expected_outcome}")
        print(f"Expected amount: INR {expected_amount:.2f}")

        fixture_valid = True

        order_details = get_order_details(message)

        if order_details:

            (
                db_customer,
                db_amount,
                delivery_minutes,
                order_status,
            ) = order_details

            if db_customer != customer_id:
                fixture_valid = False
                print("WARNING: Customer ID mismatch.")

            if expected_outcome in ("REFUND", "ESCALATE"):

                if not math.isclose(
                    float(db_amount),
                    expected_amount,
                    abs_tol=0.01,
                ):
                    fixture_valid = False
                    print(
                        "WARNING: Expected amount does not match "
                        "the database order amount."
                    )

        result = run_refund_agent(
            message,
            chunks,
            vectorizer,
            document_vectors,
        )

        actual_outcome, actual_amount = parse_actual_response(result)

        outcome_pass = actual_outcome == expected_outcome

        amount_pass = math.isclose(
            actual_amount,
            expected_amount,
            abs_tol=0.01,
        )

        test_pass = (
            outcome_pass
            and amount_pass
            and fixture_valid
        )

        print("\nActual response:")
        print(result)

        print(f"\nActual outcome: {actual_outcome}")
        print(f"Actual amount: INR {actual_amount:.2f}")

        print(
            f"Outcome check: {'PASS' if outcome_pass else 'FAIL'}"
        )
        print(
            f"Amount check: {'PASS' if amount_pass else 'FAIL'}"
        )

        if test_pass:
            print("\nTEST RESULT: PASS")
            passed += 1
            status = "PASS"
        else:
            print("\nTEST RESULT: FAIL")
            status = "FAIL"

        results.append({
            "scenario_id": scenario_id,
            "customer_id": customer_id,
            "expected_outcome": expected_outcome,
            "actual_outcome": actual_outcome,
            "expected_refund_inr": f"{expected_amount:.2f}",
            "actual_refund_inr": f"{actual_amount:.2f}",
            "result": result,
            "status": status,
        })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    failed = len(scenarios) - passed

    print("\n" + "=" * 65)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Total scenarios: {len(scenarios)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {(passed / len(scenarios)) * 100:.2f}%")

    print(f"\nDetailed results saved to:\n{OUTPUT_FILE}")

    if failed == 0:
        print("\nALL OFFICIAL SCENARIOS PASSED!")
    else:
        print("\nSome scenarios failed. Review the CSV report.")


if __name__ == "__main__":
    main()
