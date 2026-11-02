import json
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from mock_llm import chat
from policy_retriever import (
    load_policy_chunks,
    build_retriever,
    retrieve_policy
)

DB_FILE = BASE_DIR / "data" / "instamart.db"


# 1. Retrieve order details from SQLite database
def get_order(order_id):

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            """
            SELECT order_id, customer_id, order_amount,
                   delivery_minutes, status
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

        if row is None:
            return {"error": "Order not found"}

        return dict(row)

    finally:
        conn.close()


# 2. Search relevant refund policy
def search_policy(
    query,
    chunks,
    heading_vectorizer,
    heading_vectors,
    content_vectorizer,
    document_vectors
):

    results = retrieve_policy(
        query,
        chunks,
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors,
        top_k=3
    )

    return {
        "policy_sections": [
            {
                "heading": item["heading"],
                "content": item["content"]
            }
            for item in results
        ]
    }


# 3. Calculate simulated refund
def calculate_refund(order_amount, delivery_minutes):

    if delivery_minutes is None or delivery_minutes <= 45:
        refund_amount = 0
    else:
        refund_amount = float(order_amount)

    return {"refund_inr": refund_amount}


# 4. Simulate refund processing
def issue_refund(order_id, refund_inr):

    return {
        "status": "simulated",
        "order_id": order_id,
        "refund_inr": refund_inr
    }


# 5. Simulate escalation to human support
def escalate_to_human(order_id, refund_inr):

    return {
        "status": "escalated",
        "order_id": order_id,
        "refund_inr": refund_inr
    }


# 6. Execute agent tool calls
def execute_tool(
    name,
    arguments,
    chunks,
    heading_vectorizer,
    heading_vectors,
    content_vectorizer,
    document_vectors
):

    if name == "get_order":
        return get_order(arguments["order_id"])

    elif name == "search_policy":
        return search_policy(
            arguments["query"],
            chunks,
            heading_vectorizer,
            heading_vectors,
            content_vectorizer,
            document_vectors
        )

    elif name == "calculate_refund":
        return calculate_refund(
            arguments["order_amount"],
            arguments["delivery_minutes"]
        )

    elif name == "issue_refund":
        return issue_refund(
            arguments["order_id"],
            arguments["refund_inr"]
        )

    elif name == "escalate_to_human":
        return escalate_to_human(
            arguments["order_id"],
            arguments["refund_inr"]
        )

    return {"error": "Unknown tool"}


# 7. Main refund agent workflow
def run_refund_agent(
    customer_query,
    chunks,
    heading_vectorizer,
    heading_vectors,
    content_vectorizer,
    document_vectors
):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AGENT responsible for handling "
                "customer refund requests. Use the available "
                "tools to retrieve orders, search policies, "
                "calculate refunds, and process or escalate requests."
            )
        },
        {
            "role": "user",
            "content": customer_query
        }
    ]

    max_steps = 10

    for step in range(max_steps):

        response = chat(messages)

        assistant_message = response.choices[0].message

        if response.choices[0].finish_reason != "tool_calls":
            return assistant_message.content

        tool_calls = assistant_message.tool_calls

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in tool_calls
            ]
        })

        for call in tool_calls:

            print(f"\nExecuting tool: {call.name}")

            result = execute_tool(
                call.name,
                call.arguments,
                chunks,
                heading_vectorizer,
                heading_vectors,
                content_vectorizer,
                document_vectors
            )

            print("Tool result:", result)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(result)
            })

    return (
        "ESCALATE: Maximum agent steps reached. "
        "Manual review required."
    )


# 8. Main program
def main():

    print("Loading policy information...")

    chunks = load_policy_chunks()

    (
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors
    ) = build_retriever(chunks)

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
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors
    )

    print("\nFinal agent response:")
    print(result)


if __name__ == "__main__":
    main()