import os
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
IMAGE_FOLDER = "data/extracted_images"
OUTPUT_FOLDER = "data/captions"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CAPTION_OUTPUT = os.path.join(OUTPUT_FOLDER, "image_captions.json")

# -----------------------------
# LOAD BLIP MODEL
# -----------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# -----------------------------
# GENERATE CAPTIONS
# -----------------------------
captions = []

for image_file in os.listdir(IMAGE_FOLDER):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        image = Image.open(image_path).convert("RGB")

        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        captions.append({
            "image_file": image_file,
            "caption": caption
        })

# Save captions
with open(CAPTION_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(captions, f, indent=4)

print("Image auto-captions generated successfully.")
