
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

POLICY_DIR = BASE_DIR / "data" / "policies"
OUTPUT_FILE = BASE_DIR / "outputs" / "part_d" / "policy_chunks.json"

CHUNK_SIZE = 40
OVERLAP_WORDS = 10
STRIDE = CHUNK_SIZE - OVERLAP_WORDS


def load_policy(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Policy file not found: {file_path}"
        )

    text = file_path.read_text(encoding="utf-8-sig")

    return text.split()


def split_policy_into_chunks(words, source_file):
    """
    Split policy text into 40-word chunks
    with a 10-word overlap.
    """

    chunks = []

    policy_name = Path(source_file).stem.upper()
    prefix = policy_name.replace("_POLICY", "")

    start = 0
    total_words = len(words)

    while start < total_words:

        end = min(start + CHUNK_SIZE, total_words)

        chunk_words = words[start:end]
        content = " ".join(chunk_words)

        chunks.append({
            "chunk_id": f"{prefix}_{len(chunks) + 1:03d}",
            "heading": content[:100],
            "content": content,
            "text": content,
            "source_file": source_file,
            "policy_type": prefix.lower()
        })

        # Stop when the final word has been included.
        if end >= total_words:
            break

        # Move forward by 30 words,
        # retaining the previous 10 words as overlap.
        start += STRIDE

    return chunks


def save_chunks(chunks, output_file):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )


def main():

    print("Loading all policy documents...")

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    if not policy_files:
        raise FileNotFoundError(
            f"No policy documents found in: {POLICY_DIR}"
        )

    all_chunks = []

    for policy_file in policy_files:

        print(f"\nProcessing: {policy_file.name}")

        words = load_policy(policy_file)

        chunks = split_policy_into_chunks(
            words,
            policy_file.name
        )

        all_chunks.extend(chunks)

        print(f"Total words: {len(words)}")
        print(f"Chunks created: {len(chunks)}")

    save_chunks(all_chunks, OUTPUT_FILE)

    print("\nPolicy processing completed.")
    print(f"Total documents: {len(policy_files)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
