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


def generate_policy_answer(
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
        top_k=1
    )

    context = "\n\n".join(
        result["content"] for result in results
    )

    messages = [
        {
            "role": "system",
            "content": "POLICY ASSISTANT"
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{query}"
            )
        }
    ]

    response = chat(messages)

    return response.choices[0].message.content


def main():

    print("Loading policy information...")

    chunks = load_policy_chunks()

    (
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors
    ) = build_retriever(chunks)

    print("Policy retrieval system is ready.")

    query = "I received a damaged product. Can I get a refund?"

    print("\nCustomer query:")
    print(query)

    print("\nGenerating policy-based answer...")

    answer = generate_policy_answer(
        query,
        chunks,
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors
    )

    print("\nPolicy assistant response:")
    print(answer)


if __name__ == "__main__":
    main()