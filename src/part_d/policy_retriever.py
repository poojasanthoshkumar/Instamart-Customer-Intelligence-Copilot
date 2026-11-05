
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_FILE = BASE_DIR / "outputs" / "part_d" / "policy_chunks.json"

# Expand customer language into terms used in company policies.
QUERY_EXPANSIONS = {
    "damaged": "defective wrong missing product damage complaint",
    "defective": "damaged wrong missing product",
    "wrong": "incorrect damaged missing product",
    "missing": "wrong damaged product item",

    "privacy": "personal information phone email masked address deleted",
    "protected": "privacy personal information masked secure",
    "personal": "privacy phone email address masked information",

    "payment": "failed deducted money returned refund transaction",
    "deducted": "payment failed money returned automatically 48 hours",
    "placed": "payment failed order not created returned",

    "escalate": "escalation ticket human agent unresolved complaint handoff",
    "unresolved": "escalation cannot resolve ticket human agent",
    "complaint": "ticket support issue escalation human agent",

    "late": "delivery delivered delay 45 minutes refund",
    "delivery": "late delivered minutes rider order",
    "refund": "completed cancelled amount payment returned",

    "cancelled": "canceled refund charged completed order"
}


def load_policy_chunks():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Policy chunks not found: {CHUNKS_FILE}"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("Policy chunks must be a JSON list.")

    return chunks


def expand_query(query):

    words = query.lower().split()

    expanded_terms = [query]

    for word in words:
        clean_word = word.strip(".,?!:;")

        if clean_word in QUERY_EXPANSIONS:
            expanded_terms.append(
                QUERY_EXPANSIONS[clean_word]
            )

    return " ".join(expanded_terms)


def build_retriever(chunks):
    """
    Build the TF-IDF policy retrieval system.
    """

    if not chunks:
        raise ValueError("No policy chunks available.")

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    document_vectors = vectorizer.fit_transform(documents)

    return vectorizer, document_vectors


def retrieve_policy(
    query,
    chunks,
    vectorizer,
    document_vectors,
    top_k=3,
    min_score=0.02
):
    """
    Retrieve relevant policy chunks using cosine similarity.
    """

    if not query.strip():
        return []

    expanded = expand_query(query)

    query_vector = vectorizer.transform([expanded])

    similarity_scores = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    ranked_indices = similarity_scores.argsort()[::-1]

    results = []

    for index in ranked_indices:

        score = float(similarity_scores[index])

        # Ignore unrelated and zero-similarity chunks.
        if score < min_score:
            continue

        chunk = chunks[index]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source_file"],
            "score": round(score, 4),
            "text": chunk["text"]
        })

        if len(results) >= top_k:
            break

    return results


def main():

    print("Loading policy chunks...")

    chunks = load_policy_chunks()

    print(f"Total chunks loaded: {len(chunks)}")

    print("\nBuilding TF-IDF retrieval system...")

    vectorizer, document_vectors = build_retriever(chunks)

    query = input("\nEnter a customer query: ").strip()

    results = retrieve_policy(
        query,
        chunks,
        vectorizer,
        document_vectors,
        top_k=3
    )

    print("\nRetrieved Policy Chunks")

    for result in results:

        print("\n" + "=" * 60)
        print("Chunk ID:", result["chunk_id"])
        print("Source:", result["source"])
        print("Similarity:", result["score"])
        print("Policy:", result["text"])


if __name__ == "__main__":
    main()
