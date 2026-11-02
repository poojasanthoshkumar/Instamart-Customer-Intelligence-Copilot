import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
POLICY_DIR = BASE_DIR / "data" / "policies"
OUTPUT_FILE = BASE_DIR / "outputs" / "part_d" / "policy_chunks.json"


def load_policy(file_path):
    """Read and clean a policy document."""

    if not file_path.exists():
        raise FileNotFoundError(f"Policy file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def split_policy_into_chunks(text, source_file):
    """Split numbered refund sections or retain other policies as one chunk."""

    policy_name = Path(source_file).stem.upper()
    chunks = []

    if policy_name == "REFUND_POLICY":
        sections = re.split(r"(?m)(?=^\s*\d+\.\s+)", text)
    else:
        sections = [text]

    for index, section in enumerate(sections, start=1):
        section = section.strip()

        if not section:
            continue

        lines = section.split("\n", 1)
        heading = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""

        # Give each policy its own unique chunk ID.
        prefix = policy_name.replace("_POLICY", "")

        chunks.append({
            "chunk_id": f"{prefix}_{index:03d}",
            "heading": heading,
            "content": content,
            "text": section,
            "source_file": source_file,
            "policy_type": prefix.lower()
        })

    return chunks


def save_chunks(chunks, output_file):
    """Save all policy chunks as JSON."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)


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

        policy_text = load_policy(policy_file)

        chunks = split_policy_into_chunks(
            policy_text,
            policy_file.name
        )

        all_chunks.extend(chunks)

        print(f"Chunks created: {len(chunks)}")

    save_chunks(all_chunks, OUTPUT_FILE)

    print("\nAll policy documents processed successfully!")
    print(f"Total policy documents: {len(policy_files)}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nGenerated policy chunks:")

    for chunk in all_chunks:
        print(
            f"{chunk['chunk_id']} | "
            f"{chunk['source_file']} | "
            f"{chunk['heading']}"
        )


if __name__ == "__main__":
    main()