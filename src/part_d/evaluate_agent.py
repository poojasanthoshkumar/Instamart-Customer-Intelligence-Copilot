import csv
import math
import re
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from policy_retriever import load_policy_chunks, build_retriever
from refund_agent import run_refund_agent

DB_FILE = BASE_DIR / "data" / "instamart.db"
SCENARIO_FILE = BASE_DIR / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "part_d" / "agent_evaluation_results.csv"


def get_order_details(message):
    match = re.search(r"ORD-\d{6}", message)

    if not match:
        return None

    conn = sqlite3.connect(DB_FILE)

    try:
        row = conn.execute(
            """
            SELECT customer_id, order_amount, delivery_minutes, status
            FROM orders
            WHERE order_id = ?
            """,
            (match.group(0),)
        ).fetchone()

        return row

    finally:
        conn.close()


def main():
    print("=" * 60)
    print("INSTAMART REFUND AGENT - OFFICIAL SCENARIO EVALUATION")
    print("=" * 60)

    print("\nLoading policy information...")

    chunks = load_policy_chunks()

    (
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors
    ) = build_retriever(chunks)

    with open(SCENARIO_FILE, "r", encoding="utf-8-sig", newline="") as file:
        scenarios = list(csv.DictReader(file))

    print(f"Total official scenarios: {len(scenarios)}")

    results = []
    passed = 0

    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario["scenario_id"]
        customer_id = scenario["customer_id"]
        message = scenario["message"]
        expected_outcome = scenario["expected_outcome"].strip()
        expected_amount = float(scenario["expected_refund_inr"])

        print("\n" + "=" * 60)
        print(f"TEST {index}: {scenario_id}")
        print("=" * 60)
        print(f"Customer: {customer_id}")
        print(f"Expected outcome: {expected_outcome}")
        print(f"Expected amount: INR {expected_amount:.2f}")

        order_details = get_order_details(message)
        fixture_valid = True
        actual_amount = 0.0

        if order_details:
            db_customer, db_amount, delivery_minutes, order_status = order_details

            if db_customer != customer_id:
                fixture_valid = False
                print("WARNING: Customer ID does not match the order.")

            if expected_outcome in ("REFUND", "ESCALATE"):
                if not math.isclose(
                    float(db_amount),
                    expected_amount,
                    abs_tol=0.01
                ):
                    fixture_valid = False
                    print("WARNING: Expected amount does not match database.")

                if expected_outcome == "ESCALATE":
                    actual_amount = float(db_amount)

        result = run_refund_agent(
            message,
            chunks,
            heading_vectorizer,
            heading_vectors,
            content_vectorizer,
            document_vectors
        )

        result = result or ""
        actual_outcome = result.split(":", 1)[0].strip()

        if actual_outcome == "REFUND":
            amount_match = re.search(
                r"REFUND:\s*INR\s*([\d.]+)",
                result
            )

            if amount_match:
                actual_amount = float(amount_match.group(1))
            else:
                actual_outcome = "INVALID_REFUND_RESPONSE"

        outcome_pass = actual_outcome == expected_outcome

        if expected_outcome == "REFUND":
            amount_pass = math.isclose(
                actual_amount,
                expected_amount,
                abs_tol=0.01
            )
        elif expected_outcome == "ESCALATE":
            amount_pass = math.isclose(
                actual_amount,
                expected_amount,
                abs_tol=0.01
            )
        else:
            amount_pass = math.isclose(actual_amount, 0.0, abs_tol=0.01)

        test_pass = outcome_pass and amount_pass and fixture_valid

        print("\nActual response:")
        print(result)
        print(f"\nActual outcome: {actual_outcome}")
        print(f"Actual amount checked: INR {actual_amount:.2f}")

        if test_pass:
            print("TEST RESULT: PASS")
            passed += 1
            status = "PASS"
        else:
            print("TEST RESULT: FAIL")
            status = "FAIL"

        results.append({
            "scenario_id": scenario_id,
            "customer_id": customer_id,
            "expected_outcome": expected_outcome,
            "actual_outcome": actual_outcome,
            "expected_refund_inr": f"{expected_amount:.2f}",
            "actual_refund_inr": f"{actual_amount:.2f}",
            "result": result,
            "status": status
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    failed = len(scenarios) - passed

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if scenarios:
        print(f"Pass rate: {(passed / len(scenarios)) * 100:.2f}%")

    print(f"\nDetailed results saved to: {OUTPUT_FILE}")

    if failed:
        print("\nSome scenarios failed. Review the results before submission.")
    else:
        print("\nAll official scenarios passed!")


if __name__ == "__main__":
    main()
