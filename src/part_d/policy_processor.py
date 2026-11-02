import re
import json
from pathlib import Path


# Get the main project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Define file paths
POLICY_FILE = BASE_DIR / "data" / "policies" / "refund_policy.txt"
OUTPUT_FILE = BASE_DIR / "outputs" / "part_d" / "policy_chunks.json"


def load_policy(file_path):
    """Read and clean the refund policy."""

    if not file_path.exists():
        raise FileNotFoundError(f"Policy file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def split_policy_into_chunks(text):
    """Split policy into individual sections."""

    sections = re.split(r"\n(?=\d+\.\s)", text)

    chunks = []

    for index, section in enumerate(sections, start=1):

        section = section.strip()

        if not section:
            continue

        lines = section.split("\n", 1)

        heading = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""

        chunks.append({
            "chunk_id": f"POLICY_{index:03d}",
            "heading": heading,
            "content": content,
            "text": section
        })

    return chunks


def save_chunks(chunks, output_file):
    """Save policy chunks as JSON."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)


def main():

    print("Loading refund policy...")

    policy_text = load_policy(POLICY_FILE)

    print("Splitting policy into chunks...")

    chunks = split_policy_into_chunks(policy_text)

    save_chunks(chunks, OUTPUT_FILE)

    print("\nPolicy processing completed successfully!")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nGenerated policy chunks:")

    for chunk in chunks:
        print(f"{chunk['chunk_id']}: {chunk['heading']}")


if __name__ == "__main__":
    main()