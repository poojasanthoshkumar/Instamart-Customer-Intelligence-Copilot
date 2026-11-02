import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from policy_retriever import (
    load_policy_chunks,
    build_retriever,
    retrieve_policy
)


def main():

    print("=" * 55)
    print("INSTAMART POLICY RETRIEVAL EVALUATION")
    print("=" * 55)

    chunks = load_policy_chunks()

    (
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors
    ) = build_retriever(chunks)

    test_cases = [
        ("How can I report a damaged product?", "POLICY_002"),
        ("Do I need a photograph of a damaged item?", "POLICY_002"),
        ("What should I do if an item is missing?", "POLICY_003"),
        ("I received the wrong product.", "POLICY_004"),
        ("How do I report an expired product?", "POLICY_005"),
        ("What determines whether my refund request is eligible?", "POLICY_006"),
        ("Where will my approved refund be credited?", "POLICY_007"),
        ("Can I cancel an order before dispatch?", "POLICY_008"),
        ("Can I cancel an order after dispatch?", "POLICY_008"),
        ("What happens if I request the same refund twice?", "POLICY_009"),
        ("When is a refund request sent for manual review?", "POLICY_010"),
        ("How should customer support communicate refund status?", "POLICY_011"),
        ("How are refund requests verified?", "POLICY_006"),
        ("What should support do when a complaint is unclear?", "POLICY_010")
    ]

    passed = 0
    failed = 0

    for index, (query, expected_id) in enumerate(
        test_cases, start=1
    ):

        results = retrieve_policy(
            query,
            chunks,
            heading_vectorizer,
            heading_vectors,
            content_vectorizer,
            document_vectors,
            top_k=1
        )

        retrieved_id = (
            results[0]["chunk_id"] if results else "NONE"
        )

        print("\n" + "-" * 55)
        print(f"TEST {index}")
        print("Query:", query)
        print("Expected section:", expected_id)
        print("Retrieved section:", retrieved_id)

        if retrieved_id == expected_id:
            print("RESULT: PASS")
            passed += 1
        else:
            print("RESULT: FAIL")
            failed += 1

    total = len(test_cases)
    accuracy = (passed / total) * 100

    print("\n" + "=" * 55)
    print("POLICY RETRIEVAL EVALUATION SUMMARY")
    print("=" * 55)

    print("Total queries:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print(f"Retrieval accuracy: {accuracy:.2f}%")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()