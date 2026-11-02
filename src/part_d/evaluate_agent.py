import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from policy_retriever import load_policy_chunks, build_retriever
from refund_agent import run_refund_agent

DB_FILE = BASE_DIR / "data" / "instamart.db"


def find_order_id(condition):
    """Find a suitable order ID from the database."""

    conn = sqlite3.connect(DB_FILE)

    try:
        row = conn.execute(
            f"""
            SELECT order_id
            FROM orders
            WHERE {condition}
            LIMIT 1
            """
        ).fetchone()

        return row[0] if row else None

    finally:
        conn.close()


def create_test_scenarios():

    scenarios = []

    # Test 1: Successful refund
    order_id = find_order_id(
        "status = 'completed' "
        "AND delivery_minutes > 45 "
        "AND order_amount <= 1000"
    )

    if order_id:
        scenarios.append({
            "name": "Successful refund",
            "query": f"I want a refund for order {order_id} because delivery was late.",
            "expected": "REFUND:"
        })

    # Test 2: Cancelled order
    order_id = find_order_id(
        "status = 'cancelled'"
    )

    if order_id:
        scenarios.append({
            "name": "Cancelled order",
            "query": f"I want a refund for order {order_id}.",
            "expected": "DENIED: only completed orders"
        })

    # Test 3: Invalid order ID
    invalid_id = "ORD-999999"

    while find_order_id(
        f"order_id = '{invalid_id}'"
    ) is not None:
        invalid_id = "ORD-" + str(int(invalid_id[4:]) - 1)

    scenarios.append({
        "name": "Invalid order ID",
        "query": f"I want a refund for order {invalid_id}.",
        "expected": "NOT_FOUND:"
    })

    # Test 4: Missing order ID
    scenarios.append({
        "name": "Missing order ID",
        "query": "I want a refund because my delivery was late.",
        "expected": "NEEDS_INFO:"
    })

    # Test 5: Malformed order ID
    scenarios.append({
        "name": "Malformed order ID",
        "query": "I want a refund for order ORD-123 because delivery was late.",
        "expected": "NEEDS_INFO:"
    })

    # Test 6: Delivery within 45 minutes
    order_id = find_order_id(
        "status = 'completed' "
        "AND delivery_minutes <= 45"
    )

    if order_id:
        scenarios.append({
            "name": "Delivery within 45 minutes",
            "query": f"I want a refund for order {order_id} because delivery was late.",
            "expected": "DENIED: delivery was within 45 minutes"
        })

    # Test 7: High-value refund escalation
    order_id = find_order_id(
        "status = 'completed' "
        "AND delivery_minutes > 45 "
        "AND order_amount > 1000"
    )

    if order_id:
        scenarios.append({
            "name": "High-value refund escalation",
            "query": f"I want a refund for order {order_id} because delivery was late.",
            "expected": "ESCALATE:"
        })

    return scenarios


def main():

    print("=" * 55)
    print("INSTAMART REFUND AGENT - EVALUATION")
    print("=" * 55)

    print("\nLoading policy information...")

    chunks = load_policy_chunks()

    (
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors
    ) = build_retriever(chunks)

    scenarios = create_test_scenarios()

    print(f"\nTotal test scenarios available: {len(scenarios)}")

    passed = 0
    failed = 0

    for index, scenario in enumerate(scenarios, start=1):

        print("\n" + "=" * 55)
        print(f"TEST {index}: {scenario['name']}")
        print("=" * 55)

        print("\nCustomer request:")
        print(scenario["query"])

        result = run_refund_agent(
            scenario["query"],
            chunks,
            heading_vectorizer,
            heading_vectors,
            content_vectorizer,
            document_vectors
        )

        print("\nFinal response:")
        print(result)

        if result and result.startswith(scenario["expected"]):
            print("\nTEST RESULT: PASS")
            passed += 1
        else:
            print("\nTEST RESULT: FAIL")
            failed += 1

    print("\n" + "=" * 55)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 55)

    print(f"Total tests: {len(scenarios)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if passed + failed > 0:
        accuracy = (passed / (passed + failed)) * 100
        print(f"Test pass rate: {accuracy:.2f}%")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()