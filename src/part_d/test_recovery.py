
"""Isolated recovery and safety-limit tests for the refund agent."""

import csv
import json
import tempfile
from pathlib import Path

import refund_agent as agent
from policy_retriever import load_policy_chunks, build_retriever


BASE_DIR = Path(__file__).resolve().parents[2]
SCENARIO_FILE = BASE_DIR / "data" / "raw" / "agent_scenarios.csv"


def load_s1_message():
    with open(SCENARIO_FILE, "r", encoding="utf-8-sig", newline="") as file:
        scenarios = list(csv.DictReader(file))

    scenario = next(
        item for item in scenarios
        if item["scenario_id"] == "S1"
    )
    return scenario["message"]


def read_jsonl(path):
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    print("=" * 65)
    print("ISOLATED REFUND AGENT RECOVERY TESTS")
    print("=" * 65)

    message = load_s1_message()
    chunks = load_policy_chunks()

    if not chunks:
        raise RuntimeError("Policy chunks are missing.")

    vectorizer, document_vectors = build_retriever(chunks)

    with tempfile.TemporaryDirectory(
        prefix="instamart_recovery_test_"
    ) as temporary_directory:

        temporary = Path(temporary_directory)

        # Redirect runtime files to temporary storage.
        agent.OUTPUT_DIR = temporary / "outputs"
        agent.STATE_DIR = temporary / "state"
        agent.RUN_LOG = agent.OUTPUT_DIR / "agent_runs.jsonl"
        agent.REFUND_LEDGER = agent.OUTPUT_DIR / "refund_ledger.jsonl"

        agent.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        agent.STATE_DIR.mkdir(parents=True, exist_ok=True)

        # TEST 1: Crash after refund tool and resume.
        print("\nTEST 1: Crash and checkpoint recovery")

        run_id = "recovery_crash_test"

        try:
            agent.run_refund_agent(
                message,
                chunks,
                vectorizer,
                document_vectors,
                run_id=run_id,
                crash_after_tool=4,
            )
        except RuntimeError as error:
            if "Simulated crash" not in str(error):
                raise
            print("Simulated crash detected.")
        else:
            raise AssertionError("Expected simulated crash did not occur.")

        checkpoint = agent.load_checkpoint(run_id)

        assert checkpoint["status"] == "running"
        assert checkpoint["tool_sequence"][-1] == "issue_refund"

        ledger_before = read_jsonl(agent.REFUND_LEDGER)

        matching_before = [
            entry for entry in ledger_before
            if entry.get("run_id") == run_id
        ]

        assert len(matching_before) == 1

        print("Checkpoint saved.")
        print("Refund recorded before simulated crash.")

        result = agent.resume_agent(
            run_id,
            chunks,
            vectorizer,
            document_vectors,
        )

        resumed_checkpoint = agent.load_checkpoint(run_id)

        assert resumed_checkpoint["status"] == "completed"
        assert resumed_checkpoint["final_result"] == result
        assert result.startswith("REFUND:"), result

        ledger_after = read_jsonl(agent.REFUND_LEDGER)

        matching_after = [
            entry for entry in ledger_after
            if entry.get("run_id") == run_id
        ]

        assert len(matching_after) == 1

        print("Agent resumed successfully.")
        print("Final result:", result)
        print("Refund ledger entries:", len(matching_after))

        repeated_result = agent.resume_agent(
            run_id,
            chunks,
            vectorizer,
            document_vectors,
        )

        assert repeated_result == result

        final_ledger = read_jsonl(agent.REFUND_LEDGER)

        assert sum(
            entry.get("run_id") == run_id
            for entry in final_ledger
        ) == 1

        print("Completed-run resume did not duplicate the refund.")
        print("TEST 1: PASS")

        # TEST 2: Timeout fallback.
        print("\nTEST 2: Timeout fallback")

        timeout_id = "recovery_timeout_test"

        timeout_result = agent.run_refund_agent(
            message,
            chunks,
            vectorizer,
            document_vectors,
            run_id=timeout_id,
            time_limit_seconds=0,
        )

        timeout_state = agent.load_checkpoint(timeout_id)

        assert timeout_state["status"] == "fallback_timeout"
        assert timeout_result.startswith("FALLBACK_TIMEOUT:")

        print("TEST 2: PASS")

        # TEST 3: Maximum-iteration fallback.
        print("\nTEST 3: Maximum-iteration fallback")

        iteration_id = "recovery_iteration_test"

        iteration_result = agent.run_refund_agent(
            message,
            chunks,
            vectorizer,
            document_vectors,
            run_id=iteration_id,
            max_iterations=0,
        )

        iteration_state = agent.load_checkpoint(iteration_id)

        assert iteration_state["status"] == "fallback_max_iterations"
        assert iteration_result.startswith("FALLBACK_MAX_ITERATIONS:")

        print("TEST 3: PASS")

        # TEST 4: Token-limit fallback.
        print("\nTEST 4: Token-limit fallback")

        token_id = "recovery_token_test"

        token_result = agent.run_refund_agent(
            message,
            chunks,
            vectorizer,
            document_vectors,
            run_id=token_id,
            token_limit=0,
        )

        token_state = agent.load_checkpoint(token_id)

        assert token_state["status"] == "fallback_token_limit"
        assert token_result.startswith("FALLBACK_TOKEN_LIMIT:")

        print("TEST 4: PASS")

        print("\n" + "=" * 65)
        print("RECOVERY TEST SUMMARY")
        print("=" * 65)
        print("Crash and resume: PASS")
        print("Refund idempotency within resumed run: PASS")
        print("Timeout fallback: PASS")
        print("Maximum-iteration fallback: PASS")
        print("Token-limit fallback: PASS")
        print("\nAll isolated recovery tests passed.")
        print("Temporary test files will be removed automatically.")


if __name__ == "__main__":
    main()
