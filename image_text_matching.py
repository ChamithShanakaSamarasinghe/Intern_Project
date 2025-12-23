import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configuring the Path
CHUNK_FOLDER = "data/chunks"
IMAGE_FOLDER = "data/extracted_images"
OUTPUT_FOLDER = "data/embeddings"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TEXT_OUTPUT = os.path.join(OUTPUT_FOLDER, "clip_text_embeddings.json")
IMAGE_OUTPUT = os.path.join(OUTPUT_FOLDER, "clip_image_embeddings.json")

# Loading the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Generating the text Embeddings
text_embeddings = []

for file in os.listdir(CHUNK_FOLDER):
    if file.endswith("_chunks.txt"):
        with open(os.path.join(CHUNK_FOLDER, file), "r", encoding="utf-8") as f:
            chunks = f.read().split("--- CHUNK")[1:]

        for idx, chunk in enumerate(chunks):
            text = chunk.split("\n", 1)[1].strip()

            inputs = processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            with torch.no_grad():
                embedding = model.get_text_features(**inputs)

            text_embeddings.append({
                "source_file": file,
                "chunk_id": idx + 1,
                "embedding": embedding[0].tolist()
            })

# Saving the text embeddings
with open(TEXT_OUTPUT, "w") as f:
    json.dump(text_embeddings, f, indent=4)

print("CLIP text embeddings generated.")


# Generating the Image Embeddings 
image_embeddings = []

for image_file in os.listdir(IMAGE_FOLDER):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(
            os.path.join(IMAGE_FOLDER, image_file)
        ).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            embedding = model.get_image_features(**inputs)

        image_embeddings.append({
            "image_file": image_file,
            "embedding": embedding[0].tolist()
        })

# Saving the image embeddings
with open(IMAGE_OUTPUT, "w") as f:
    json.dump(image_embeddings, f, indent=4)

print("CLIP image embeddings generated.")
