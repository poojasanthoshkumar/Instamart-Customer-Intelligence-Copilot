
"""rag_pipeline.py
Policy-based Retrieval Augmented Generation for Instamart.
"""

import sys
from pathlib import Path

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mock_llm import chat

from src.part_d.policy_retriever import (
    load_policy_chunks,
    build_retriever,
    retrieve_policy
)

TOP_K = 3

POLICY_DIR = PROJECT_ROOT / "data" / "policies"

FALLBACK_ANSWER = (
    "I could not find specific information about this issue "
    "in the available company policies. Please contact customer "
    "support for further assistance."
)


# --------------------------------------------------
# INITIALIZE RAG
# --------------------------------------------------

def initialize_rag():

    print("Loading policy information...")

    chunks = load_policy_chunks()

    if not chunks:
        raise ValueError("No policy chunks found.")

    vectorizer, document_vectors = build_retriever(chunks)

    print(f"Total policy chunks: {len(chunks)}")
    print("Policy retrieval system is ready.")

    return chunks, vectorizer, document_vectors


# --------------------------------------------------
# PREPARE POLICY CONTEXT
# --------------------------------------------------

def prepare_context(results):

    if not results:
        return "", []

    best_score = max(
        result["score"] for result in results
    )

    relevant_results = [
        result for result in results
        if result["score"] >= max(0.02, best_score * 0.30)
    ]

    sources = []

    for result in relevant_results:

        source = Path(result["source"]).name

        if source not in sources:
            sources.append(source)

    context_parts = []

    for source in sources:

        policy_path = POLICY_DIR / source

        if policy_path.exists():

            policy_text = policy_path.read_text(
                encoding="utf-8"
            ).strip()

        else:

            policy_text = "\n".join(
                result["text"]
                for result in relevant_results
                if Path(result["source"]).name == source
            )

        if not policy_text:
            continue

        context_parts.append(
            f"Source: {source}\n"
            f"Policy: {policy_text}"
        )

    return "\n\n".join(context_parts), sources


# --------------------------------------------------
# IDENTIFY SOURCES USED IN THE FINAL ANSWER
# --------------------------------------------------

def identify_answer_sources(answer, available_sources):

    answer_lower = answer.lower()

    matched_sources = []

    for source in available_sources:

        title = (
            Path(source)
            .stem
            .replace("_", " ")
            .lower()
        )

        if title in answer_lower:
            matched_sources.append(source)

    # Preserve available source information if the
    # original policy does not contain a title.
    return matched_sources if matched_sources else available_sources


# --------------------------------------------------
# GENERATE POLICY ANSWER
# --------------------------------------------------

def generate_policy_answer(
    query,
    chunks,
    vectorizer,
    document_vectors
):

    results = retrieve_policy(
        query,
        chunks,
        vectorizer,
        document_vectors,
        top_k=TOP_K
    )

    print("\nRetrieved policy chunks:")

    for result in results:

        print(
            f"\nChunk ID: {result['chunk_id']}"
            f"\nSource: {result['source']}"
            f"\nSimilarity: {result['score']}"
            f"\nContent: {result['text'][:300]}"
        )

    if not results:

        return {
            "answer": FALLBACK_ANSWER,
            "sources": []
        }

    context, available_sources = prepare_context(results)

    if not context:

        return {
            "answer": FALLBACK_ANSWER,
            "sources": []
        }

    messages = [

        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT\n"
                "You are an Instamart customer support assistant.\n"
                "Answer customer questions using only the supplied "
                "company policy documents.\n"
                "Identify the policy that directly addresses the "
                "customer's question.\n"
                "Read the complete relevant policy before answering.\n"
                "Include all applicable conditions, instructions, "
                "eligibility rules and time limits.\n"
                "For multiple-part questions, address every part.\n"
                "Do not select only one sentence when several "
                "sentences are needed for a complete answer.\n"
                "Do not mix unrelated policies.\n"
                "Do not invent information or make unsupported promises.\n"
                "If information is unavailable, clearly say so."
            )
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

    answer = response.choices[0].message.content

    if not answer or not answer.strip():

        answer = FALLBACK_ANSWER

    answer = answer.strip()

    sources = identify_answer_sources(
        answer,
        available_sources
    )

    return {
        "answer": answer,
        "sources": sources
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    try:

        chunks, vectorizer, document_vectors = initialize_rag()

        print("\n" + "=" * 55)

        query = input("\nCustomer query:\n").strip()

        if not query:
            print("Please enter a valid query.")
            return

        print("\nGenerating policy-based answer...")

        result = generate_policy_answer(
            query,
            chunks,
            vectorizer,
            document_vectors
        )

        print("\n" + "=" * 55)
        print("POLICY ASSISTANT RESPONSE")
        print("=" * 55)

        print("\n" + result["answer"])

        print("\nSources:")

        for source in result["sources"]:
            print(f"- {source}")

        print("\n" + "=" * 55)

    except Exception as error:

        print("\nRAG pipeline error:")
        print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
