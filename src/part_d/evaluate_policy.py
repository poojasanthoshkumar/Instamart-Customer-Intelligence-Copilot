import csv
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src" / "part_d"))

from policy_retriever import (
    load_policy_chunks,
    build_retriever,
    retrieve_policy
)

EVALUATION_FILE = BASE_DIR / "data" / "raw" / "policy_eval.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "part_d" / "policy_evaluation_results.csv"

# Diagnostic results showed the lowest supported-query score was 0.1597,
# while both unsupported questions scored 0.0000.
ABSTAIN_THRESHOLD = 0.05


def normalize(text):
    """Normalize text for reliable fact matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def main():

    print("=" * 65)
    print("INSTAMART POLICY RETRIEVAL EVALUATION")
    print("=" * 65)

    chunks = load_policy_chunks()

    hv, hvec, cv, dvec = build_retriever(chunks)

    with open(EVALUATION_FILE, "r", encoding="utf-8", newline="") as file:
        test_cases = list(csv.DictReader(file))

    if not test_cases:
        raise ValueError("Policy evaluation CSV is empty.")

    passed = 0
    failed = 0
    evaluation_results = []

    for index, case in enumerate(test_cases, start=1):

        query = case["question"].strip()
        expected_doc = case["expected_doc"].strip()
        expected_fact = case["expected_fact"].strip()

        results = retrieve_policy(
            query,
            chunks,
            hv,
            hvec,
            cv,
            dvec,
            top_k=3
        )

        best_score = (
            results[0]["similarity_score"] if results else 0.0
        )

        best_chunk = None

        if results:
            best_chunk = next(
                (
                    chunk for chunk in chunks
                    if chunk["chunk_id"] == results[0]["chunk_id"]
                ),
                None
            )

        if expected_doc == "NONE":

            abstained = best_score < ABSTAIN_THRESHOLD
            document_correct = abstained
            fact_found = abstained

        else:

            document_correct = (
                best_chunk is not None
                and best_chunk["source_file"] == expected_doc
                and best_score >= ABSTAIN_THRESHOLD
            )

            retrieved_expected_chunks = []

            for result in results:
                matching_chunk = next(
                    (
                        chunk for chunk in chunks
                        if chunk["chunk_id"] == result["chunk_id"]
                    ),
                    None
                )

                if (
                    matching_chunk is not None
                    and matching_chunk["source_file"] == expected_doc
                ):
                    retrieved_expected_chunks.append(matching_chunk["text"])

            retrieved_text = normalize(
                " ".join(retrieved_expected_chunks)
            )

            fact_found = (
                normalize(expected_fact) in retrieved_text
                if expected_fact else False
            )

        test_passed = document_correct and fact_found

        if test_passed:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        retrieved_doc = (
            best_chunk["source_file"]
            if best_chunk is not None
            else "NONE"
        )

        print("\n" + "-" * 65)
        print(f"TEST {index}")
        print("Question:", query)
        print("Expected document:", expected_doc)
        print("Retrieved document:", retrieved_doc)
        print("Top similarity score:", best_score)
        print("Expected fact:", expected_fact or "NONE")
        print("Fact found:", fact_found)
        print("RESULT:", status)

        evaluation_results.append({
            "question": query,
            "expected_doc": expected_doc,
            "retrieved_doc": retrieved_doc,
            "top_similarity_score": best_score,
            "expected_fact": expected_fact,
            "fact_found": fact_found,
            "result": status
        })

    total = len(test_cases)
    accuracy = (passed / total) * 100

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=evaluation_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(evaluation_results)

    print("\n" + "=" * 65)
    print("POLICY EVALUATION SUMMARY")
    print("=" * 65)
    print("Total questions:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print(f"Evaluation accuracy: {accuracy:.2f}%")
    print("Results saved to:", OUTPUT_FILE)

    if failed:
        print("\nSome tests failed. Review the individual results above.")
    else:
        print("\nAll policy evaluation tests passed!")


if __name__ == "__main__":
    main()