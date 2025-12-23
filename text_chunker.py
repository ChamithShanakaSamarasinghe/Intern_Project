import os
import re
from run_config import PATHS

INPUT_FOLDER = PATHS["extracted_text"]
OUTPUT_FOLDER = PATHS["chunks"]


def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def split_into_chunks(text, max_tokens=300):
    pages = re.split(r'\[PAGE_BREAK_\d+\]', text)
    chunks = []

    for page in pages:
        if not page.strip():
            continue

        sentences = re.split(r'(?<=[.!?])\s+', page)
        current = ""

        for s in sentences:
            if len((current + s).split()) <= max_tokens:
                current += " " + s
            else:
                chunks.append(current.strip())
                current = s

        if current.strip():
            chunks.append(current.strip())

    return chunks


def run():
    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".txt"):
            text = open(os.path.join(INPUT_FOLDER, file), encoding="utf-8").read()
            text = clean_text(text)
            chunks = split_into_chunks(text)

            out_file = os.path.join(
                OUTPUT_FOLDER,
                file.replace(".txt", "_chunks.txt")
            )

            with open(out_file, "w", encoding="utf-8") as f:
                for i, c in enumerate(chunks):
                    if len(c.split()) < 5:
                        continue
                    f.write(f"--- CHUNK {i+1} ---\n{c}\n\n")

            print(f"✔ Chunked: {file}")


if __name__ == "__main__":
    run()
