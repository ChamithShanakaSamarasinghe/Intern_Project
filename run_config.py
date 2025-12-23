import os
from datetime import datetime

RUN_ID = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
BASE_RUN_DIR = os.path.join("data", "runs", RUN_ID)

PATHS = {
    "extracted_text": os.path.join(BASE_RUN_DIR, "extracted_text"),
    "extracted_images": os.path.join(BASE_RUN_DIR, "extracted_images"),
    "chunks": os.path.join(BASE_RUN_DIR, "chunks"),
    "metadata": os.path.join(BASE_RUN_DIR, "metadata"),
    "embeddings": os.path.join(BASE_RUN_DIR, "embeddings")
}

for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

print(f"🚀 Pipeline Run Started → {RUN_ID}")
