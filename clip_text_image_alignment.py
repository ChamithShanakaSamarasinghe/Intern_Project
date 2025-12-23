import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from run_config import PATHS

CHUNK_FOLDER = PATHS["chunks"]
IMAGE_FOLDER = PATHS["extracted_images"]
OUTPUT_FOLDER = PATHS["embeddings"]

TEXT_OUT = os.path.join(OUTPUT_FOLDER, "clip_text_embeddings.json")
IMAGE_OUT = os.path.join(OUTPUT_FOLDER, "clip_image_embeddings.json")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text_embeddings = []

for file in os.listdir(CHUNK_FOLDER):
    if file.endswith("_chunks.txt"):
        chunks = open(os.path.join(CHUNK_FOLDER, file), encoding="utf-8").read().split("--- CHUNK")[1:]

        for i, ch in enumerate(chunks):
            parts = ch.split("\n", 1)
            if len(parts) < 2:
                continue

            text = parts[1].strip()
            if not text:
                continue

            inputs = processor(text=[text], return_tensors="pt", truncation=True)
            with torch.no_grad():
                emb = model.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            text_embeddings.append({
                "source_file": file,
                "chunk_id": i + 1,
                "embedding": emb[0].tolist()
            })

json.dump(text_embeddings, open(TEXT_OUT, "w"), indent=4)

image_embeddings = []

for img in os.listdir(IMAGE_FOLDER):
    if img.lower().endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(os.path.join(IMAGE_FOLDER, img)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        image_embeddings.append({
            "image_file": img,
            "embedding": emb[0].tolist()
        })

json.dump(image_embeddings, open(IMAGE_OUT, "w"), indent=4)

print("✔ CLIP embeddings generated")
