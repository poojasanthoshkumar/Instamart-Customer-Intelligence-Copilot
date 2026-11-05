
"""refund_agent.py
Offline policy-grounded Instamart refund agent.

Features:
- Five simulated agent tools
- Repeated-tool-call detection
- Maximum model-call, token and time limits
- Run logging in outputs/agent_runs.jsonl
- Refund ledger in outputs/refund_ledger.jsonl
- Per-run checkpoints in state/
- Checkpoint resume support
- Crash simulation for recovery testing

Compatible with the original supplied mock_llm.py.
"""

import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mock_llm import chat
from policy_retriever import (
    load_policy_chunks,
    build_retriever,
    retrieve_policy,
)

DB_FILE = BASE_DIR / "data" / "instamart.db"

OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"

RUN_LOG = OUTPUT_DIR / "agent_runs.jsonl"
REFUND_LEDGER = OUTPUT_DIR / "refund_ledger.jsonl"

MAX_ITERATIONS = 6
TOKEN_LIMIT = 6000
TIME_LIMIT_SECONDS = 10


# ---------------------------------------------------------
# FILE AND CHECKPOINT UTILITIES
# ---------------------------------------------------------

def append_jsonl(path, record):
    """Append one JSON record to a JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(run_id, state):
    """Save the current agent state safely."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    path = STATE_DIR / f"{run_id}.json"
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary.replace(path)


def load_checkpoint(run_id):
    """Load a previously saved agent checkpoint."""
    path = STATE_DIR / f"{run_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def finish_run(run_id, status, result, state):
    """Save final run status and append a run-log record."""
    state["status"] = status
    state["final_result"] = result

    save_checkpoint(run_id, state)

    append_jsonl(
        RUN_LOG,
        {
            "run_id": run_id,
            "status": status,
            "result": result,
            "model_calls": state.get("model_calls", 0),
            "total_tokens": state.get("total_tokens", 0),
            "tool_sequence": state.get("tool_sequence", []),
            "timestamp": time.time(),
        },
    )

    return result


# ---------------------------------------------------------
# TOOL 1: GET ORDER
# ---------------------------------------------------------

def get_order(order_id):
    """Retrieve order details from SQLite."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            """
            SELECT order_id, customer_id, order_amount,
                   delivery_minutes, status, order_ts
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

        if row is None:
            return {"error": "Order not found"}

        return dict(row)

    finally:
        conn.close()


# ---------------------------------------------------------
# TOOL 2: SEARCH POLICY
# ---------------------------------------------------------

def search_policy(
    query,
    chunks,
    vectorizer,
    document_vectors,
):
    """Retrieve relevant policy sections."""
    results = retrieve_policy(
        query,
        chunks,
        vectorizer,
        document_vectors,
        top_k=len(chunks),
    )

    return {
        "policy_sections": [
            {
                "heading": item.get("heading", ""),
                "content": item.get("text", ""),
                "source": item.get("source", ""),
                "chunk_id": item.get("chunk_id", ""),
            }
            for item in results
        ]
    }


# ---------------------------------------------------------
# TOOL 3: CALCULATE REFUND
# ---------------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):
    """Calculate the simulated late-delivery refund."""
    if delivery_minutes is None:
        return {
            "refund_inr": 0.0,
            "reason": "Delivery time unavailable.",
        }

    if float(delivery_minutes) <= 45:
        return {
            "refund_inr": 0.0,
            "reason": "Delivery completed within 45 minutes.",
        }

    return {
        "refund_inr": round(float(order_amount), 2),
        "reason": "Delivery exceeded 45 minutes.",
    }


# ---------------------------------------------------------
# TOOL 4: ISSUE REFUND
# ---------------------------------------------------------

def issue_refund(order_id, refund_inr, run_id=None):
    """
    Record a simulated refund in the refund ledger.

    This does not perform a real payment transaction.
    Duplicate ledger entries for the same run and order
    are prevented.
    """
    amount = round(float(refund_inr), 2)

    REFUND_LEDGER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if REFUND_LEDGER.exists():
        with open(
            REFUND_LEDGER,
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                try:
                    old = json.loads(line)

                    if (
                        old.get("run_id") == run_id
                        and old.get("order_id") == order_id
                    ):
                        return old["result"]

                except (json.JSONDecodeError, KeyError):
                    continue

    result = {
        "status": "simulated",
        "order_id": order_id,
        "refund_inr": amount,
        "payment_method": "original payment method",
        "credit_timeline": "within 5 working days",
    }

    append_jsonl(
        REFUND_LEDGER,
        {
            "run_id": run_id,
            "order_id": order_id,
            "refund_inr": amount,
            "result": result,
            "timestamp": time.time(),
        },
    )

    return result


# ---------------------------------------------------------
# TOOL 5: ESCALATE TO HUMAN
# ---------------------------------------------------------

def escalate_to_human(order_id, refund_inr):
    """Simulate escalation for human approval."""
    return {
        "status": "escalated",
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "reason": "Human support lead approval required.",
    }


# ---------------------------------------------------------
# TOOL EXECUTION ROUTER
# ---------------------------------------------------------

def execute_tool(
    name,
    arguments,
    chunks,
    vectorizer,
    document_vectors,
    run_id,
):
    """Route a model tool call to its corresponding function."""

    if name == "get_order":
        return get_order(arguments["order_id"])

    if name == "search_policy":
        return search_policy(
            arguments["query"],
            chunks,
            vectorizer,
            document_vectors,
        )

    if name == "calculate_refund":
        return calculate_refund(
            arguments["order_amount"],
            arguments["delivery_minutes"],
        )

    if name == "issue_refund":
        return issue_refund(
            arguments["order_id"],
            arguments["refund_inr"],
            run_id,
        )

    if name == "escalate_to_human":
        return escalate_to_human(
            arguments["order_id"],
            arguments["refund_inr"],
        )

    return {
        "error": f"Unknown tool: {name}"
    }


# ---------------------------------------------------------
# TOKEN ACCOUNTING
# ---------------------------------------------------------

def get_token_count(response, messages):
    """
    Read token usage from the mock response when available.

    Otherwise use a reproducible character-based estimate.
    """
    usage = getattr(response, "usage", None)

    if isinstance(usage, dict):
        reported = usage.get("total_tokens")
    else:
        reported = getattr(
            usage,
            "total_tokens",
            None,
        )

    if reported is not None:
        return int(reported)

    serialized = json.dumps(
        messages,
        ensure_ascii=False,
        default=str,
    )

    return max(1, len(serialized) // 4)


# ---------------------------------------------------------
# MAIN AGENT EXECUTION ENGINE
# ---------------------------------------------------------

def _run(
    customer_query,
    chunks,
    vectorizer,
    document_vectors,
    run_id,
    state,
    max_iterations,
    token_limit,
    time_limit_seconds,
    crash_after_tool=None,
):
    """Execute or resume the agent loop."""
    started = time.monotonic()

    while True:

        if state.get("status") == "completed":
            return state["final_result"]

        # TIME LIMIT
        if (
            time.monotonic() - started
            >= time_limit_seconds
        ):
            return finish_run(
                run_id,
                "fallback_timeout",
                (
                    "FALLBACK_TIMEOUT: Time limit exceeded. "
                    "Manual review required."
                ),
                state,
            )

        # MODEL-CALL LIMIT
        if state["model_calls"] >= max_iterations:
            return finish_run(
                run_id,
                "fallback_max_iterations",
                (
                    "FALLBACK_MAX_ITERATIONS: "
                    "Maximum model calls reached. "
                    "Manual review required."
                ),
                state,
            )

        # MODEL CALL
        response = chat(state["messages"])
        state["model_calls"] += 1

        choice = response.choices[0]
        assistant_message = choice.message

        # FINAL RESPONSE
        if choice.finish_reason != "tool_calls":

            tokens = get_token_count(
                response,
                state["messages"],
            )

            state["total_tokens"] += tokens

            if state["total_tokens"] > token_limit:
                return finish_run(
                    run_id,
                    "fallback_token_limit",
                    (
                        "FALLBACK_TOKEN_LIMIT: "
                        "Token budget exceeded. "
                        "Manual review required."
                    ),
                    state,
                )

            result = (
                assistant_message.content
                or "FALLBACK: No final response generated."
            )

            if state.get("escalation_result"):
                amount = state["escalation_result"]["refund_inr"]

                result = (
                    f"ESCALATE: refund of INR {amount:.2f} "
                    "sent to a human support lead for approval."
                )

            return finish_run(
                run_id,
                "completed",
                result,
                state,
            )

        # TOOL CALL RESPONSE
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            return finish_run(
                run_id,
                "fallback_empty_tool_calls",
                "FALLBACK: Empty tool-call response.",
                state,
            )

        # REPEATED TOOL CALL DETECTION
        # This check is deliberately performed before token fallback.
        for call in tool_calls:

            signature = (
                call.name,
                json.dumps(
                    call.arguments,
                    sort_keys=True,
                ),
            )

            if signature in state["seen_tool_calls"]:
                return finish_run(
                    run_id,
                    "fallback_repeated_tool_call",
                    (
                        "FALLBACK: Repeated tool call detected. "
                        "Manual support required."
                    ),
                    state,
                )

            state["seen_tool_calls"].append(signature)

        # TOKEN LIMIT
        tokens = get_token_count(
            response,
            state["messages"],
        )

        state["total_tokens"] += tokens

        if state["total_tokens"] > token_limit:
            return finish_run(
                run_id,
                "fallback_token_limit",
                (
                    "FALLBACK_TOKEN_LIMIT: "
                    "Token budget exceeded. "
                    "Manual review required."
                ),
                state,
            )

        # STORE ASSISTANT TOOL REQUEST
        state["messages"].append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    for call in tool_calls
                ],
            }
        )

        # EXECUTE EACH TOOL
        for call in tool_calls:

            print(
                f"\nExecuting tool: {call.name}"
            )

            try:
                tool_result = execute_tool(
                    call.name,
                    call.arguments,
                    chunks,
                    vectorizer,
                    document_vectors,
                    run_id,
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:

                return finish_run(
                    run_id,
                    "fallback_tool_error",
                    f"FALLBACK: Tool execution error: {exc}",
                    state,
                )

            print("Tool result:", tool_result)

            if (
                call.name == "escalate_to_human"
                and tool_result.get("status") == "escalated"
            ):
                state["escalation_result"] = tool_result

            state["tool_sequence"].append(
                call.name
            )

            # SAVE TOOL OBSERVATION
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(tool_result),
                }
            )

            # CHECKPOINT AFTER EVERY TOOL
            save_checkpoint(
                run_id,
                state,
            )

            # CRASH SIMULATION
            if (
                crash_after_tool is not None
                and len(state["tool_sequence"])
                == crash_after_tool
            ):
                raise RuntimeError(
                    "Simulated crash after tool "
                    f"{crash_after_tool}"
                )

            # TIME CHECK AFTER TOOL
            if (
                time.monotonic() - started
                >= time_limit_seconds
            ):
                return finish_run(
                    run_id,
                    "fallback_timeout",
                    (
                        "FALLBACK_TIMEOUT: "
                        "Time limit exceeded. "
                        "Manual review required."
                    ),
                    state,
                )


# ---------------------------------------------------------
# START A NEW AGENT RUN
# ---------------------------------------------------------

def run_refund_agent(
    customer_query,
    chunks,
    vectorizer,
    document_vectors,
    max_iterations=MAX_ITERATIONS,
    token_limit=TOKEN_LIMIT,
    time_limit_seconds=TIME_LIMIT_SECONDS,
    run_id=None,
    crash_after_tool=None,
):
    """Create and execute a new refund-agent run."""

    run_id = run_id or str(uuid.uuid4())

    system_message = (
        "AGENT: You are the Instamart refund support agent. "
        "Use the available tools to retrieve orders, retrieve "
        "refund policies, calculate refunds and process or "
        "escalate requests. Only completed orders are eligible. "
        "Delivery above 45 minutes qualifies for a full refund. "
        "Refunds above INR 1000 require human approval. "
        "Refund processing is simulated, not a real transaction."
    )

    state = {
        "run_id": run_id,
        "status": "running",
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": customer_query,
            },
        ],
        "model_calls": 0,
        "total_tokens": 0,
        "tool_sequence": [],
        "seen_tool_calls": [],
        "escalation_result": None,
    }

    save_checkpoint(
        run_id,
        state,
    )

    return _run(
        customer_query,
        chunks,
        vectorizer,
        document_vectors,
        run_id,
        state,
        max_iterations,
        token_limit,
        time_limit_seconds,
        crash_after_tool,
    )


# ---------------------------------------------------------
# RESUME AN EXISTING AGENT RUN
# ---------------------------------------------------------

def resume_agent(
    run_id,
    chunks,
    vectorizer,
    document_vectors,
    max_iterations=MAX_ITERATIONS,
    token_limit=TOKEN_LIMIT,
    time_limit_seconds=TIME_LIMIT_SECONDS,
):
    """Resume an interrupted run using its saved checkpoint."""

    state = load_checkpoint(run_id)

    if state.get("status") == "completed":
        return state["final_result"]

    state["status"] = "running"

    save_checkpoint(
        run_id,
        state,
    )

    return _run(
        "",
        chunks,
        vectorizer,
        document_vectors,
        run_id,
        state,
        max_iterations,
        token_limit,
        time_limit_seconds,
    )


# ---------------------------------------------------------
# STANDALONE EXECUTION
# ---------------------------------------------------------

def main():

    print("Loading policy information...")

    chunks = load_policy_chunks()

    if not chunks:
        raise RuntimeError(
            "No policy chunks found."
        )

    vectorizer, document_vectors = build_retriever(
        chunks
    )

    print(
        f"Policy chunks loaded: {len(chunks)}"
    )

    print("Refund agent is ready.")

    customer_query = (
        "I want a refund for my order ORD-111469 "
        "because the delivery was late."
    )

    print("\nCustomer request:")
    print(customer_query)

    print("\nStarting refund agent...")

    result = run_refund_agent(
        customer_query,
        chunks,
        vectorizer,
        document_vectors,
    )

    print("\nFinal agent response:")
    print(result)


if __name__ == "__main__":
    main()
