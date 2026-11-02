import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_FILE = BASE_DIR / "outputs" / "part_d" / "policy_chunks.json"


def load_policy_chunks():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Policy chunks not found: {CHUNKS_FILE}"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def expand_query(query):

    query_lower = query.lower()
    expanded_terms = [query]

    if any(term in query_lower for term in [
        "eligible", "eligibility", "qualify",
        "verified", "verification", "valid"
    ]):
        expanded_terms.append(
            "refund eligibility eligible valid verified "
            "verification supporting evidence requirements"
        )

    if any(term in query_lower for term in [
        "twice", "duplicate", "same refund",
        "again", "repeated", "multiple"
    ]):
        expanded_terms.append(
            "duplicate refund requests repeated same request "
            "multiple refund twice"
        )

    if any(term in query_lower for term in [
        "damaged", "broken", "defective"
    ]):
        expanded_terms.append(
            "damaged product photograph evidence condition"
        )

    if any(term in query_lower for term in [
        "missing", "absent", "not received"
    ]):
        expanded_terms.append(
            "missing products order details items"
        )

    if any(term in query_lower for term in [
        "wrong", "incorrect", "different product"
    ]):
        expanded_terms.append(
            "wrong incorrect products delivered"
        )

    if any(term in query_lower for term in [
        "expired", "expiry", "expiration"
    ]):
        expanded_terms.append(
            "expired product expiry date photograph"
        )

    if any(term in query_lower for term in [
        "unclear", "conflicting", "manual review",
        "escalation", "escalate"
    ]):
        expanded_terms.append(
            "manual review escalation unclear conflicting requests"
        )

    return " ".join(expanded_terms)


def build_retriever(chunks):

    headings = [chunk["heading"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]

    heading_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    content_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    heading_vectors = heading_vectorizer.fit_transform(headings)
    document_vectors = content_vectorizer.fit_transform(documents)

    return (
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors
    )


def retrieve_policy(
    query,
    chunks,
    heading_vectorizer,
    heading_vectors,
    content_vectorizer,
    document_vectors,
    top_k=3
):

    expanded_query = expand_query(query)

    query_heading_vector = heading_vectorizer.transform(
        [expanded_query]
    )

    query_content_vector = content_vectorizer.transform(
        [expanded_query]
    )

    heading_scores = cosine_similarity(
        query_heading_vector,
        heading_vectors
    ).flatten()

    content_scores = cosine_similarity(
        query_content_vector,
        document_vectors
    ).flatten()

    combined_scores = (
        0.65 * heading_scores +
        0.35 * content_scores
    )

    ranked_indices = combined_scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:

        results.append({
            "chunk_id": chunks[index]["chunk_id"],
            "heading": chunks[index]["heading"],
            "content": chunks[index]["text"],
            "similarity_score": round(
                float(combined_scores[index]), 4
            )
        })

    return results


def main():

    print("Loading policy chunks...")

    chunks = load_policy_chunks()

    print("Building improved TF-IDF retrieval system...")

    (
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors
    ) = build_retriever(chunks)

    query = "I received a damaged product. Can I get a refund?"

    results = retrieve_policy(
        query,
        chunks,
        heading_vectorizer,
        heading_vectors,
        content_vectorizer,
        document_vectors,
        top_k=3
    )

    for result in results:

        print("\n" + "=" * 50)
        print("Policy ID:", result["chunk_id"])
        print("Section:", result["heading"])
        print("Similarity:", result["similarity_score"])
        print("Policy:", result["content"])


if __name__ == "__main__":
    main()