import fitz
import os
from PIL import Image
import io
from run_config import PATHS

PDF_FOLDER = "data/manuals"
OUTPUT_TEXT = PATHS["extracted_text"]
OUTPUT_IMAGES = PATHS["extracted_images"]


def extract_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    text_output_path = os.path.join(OUTPUT_TEXT, f"{pdf_name}.txt")

    with open(text_output_path, "w", encoding="utf-8") as f:
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            f.write(f"\n[PAGE_BREAK_{page_num + 1}]\n{text}\n")

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base = doc.extract_image(xref)
                image = Image.open(io.BytesIO(base["image"]))
                name = f"{pdf_name}_page{page_num+1}_{img_index+1}.{base['ext']}"
                image.save(os.path.join(OUTPUT_IMAGES, name))

    print(f"✔ Parsed: {pdf_name}")


def run():
    for file in os.listdir(PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            extract_from_pdf(os.path.join(PDF_FOLDER, file))


if __name__ == "__main__":
    run()
