import os
import json
from datetime import datetime

CHUNK_FOLDER = "data/chunks"
IMAGE_FOLDER = "data/extracted_images"
METADATA_FOLDER = "data/metadata"

os.makedirs(METADATA_FOLDER, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

TEXT_METADATA_FILE = os.path.join(
    METADATA_FOLDER, f"text_metadata_{timestamp}.json"
)
IMAGE_LINK_FILE = os.path.join(
    METADATA_FOLDER, f"image_text_links_{timestamp}.json"
)

# -----------------------------
# STEP 1: Generate Text Metadata
# -----------------------------
text_metadata = []
global_chunk_id = 1

for file in os.listdir(CHUNK_FOLDER):
    if not file.endswith("_chunks.txt"):
        continue

    source_doc = file.replace("_chunks.txt", "")
    file_path = os.path.join(CHUNK_FOLDER, file)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_chunk = []

    for line in lines:
        if line.startswith("--- CHUNK"):
            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    text_metadata.append({
                        "chunk_id": global_chunk_id,
                        "source_document": source_doc,
                        "chunk_text": chunk_text,
                        "token_count": len(chunk_text.split())
                    })
                    global_chunk_id += 1
                current_chunk = []
        else:
            current_chunk.append(line.strip())

    # Last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            text_metadata.append({
                "chunk_id": global_chunk_id,
                "source_document": source_doc,
                "chunk_text": chunk_text,
                "token_count": len(chunk_text.split())
            })
            global_chunk_id += 1

with open(TEXT_METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(text_metadata, f, indent=4)

print(f"Text metadata created → {TEXT_METADATA_FILE}")

# -----------------------------
# STEP 2: Link Images to Text
# -----------------------------
image_links = []

for image in os.listdir(IMAGE_FOLDER):
    if image.lower().endswith((".png", ".jpg", ".jpeg")):
        source_doc = image.split("_page")[0]

        linked_chunks = [
            item["chunk_id"]
            for item in text_metadata
            if item["source_document"] == source_doc
        ]

        image_links.append({
            "image_file": image,
            "source_document": source_doc,
            "linked_chunks": linked_chunks
        })

with open(IMAGE_LINK_FILE, "w", encoding="utf-8") as f:
    json.dump(image_links, f, indent=4)

print(f"Image–text linking created → {IMAGE_LINK_FILE}")
