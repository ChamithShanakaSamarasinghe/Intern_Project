📄 Phase 3 – Document Processing (Multimodal RAG)
📌 Overview

This phase focuses on document processing for a multimodal Retrieval-Augmented Generation (RAG) system. The objective is to extract structured information from technical manuals, including text, images, and metadata, and prepare them for downstream embedding and retrieval stages.

The implementation follows a practical, code-first approach, ensuring real-world applicability rather than theoretical concepts.

🎯 Phase Objectives

Parse PDFs containing text and images

Chunk extracted text for efficient retrieval

Maintain metadata to preserve document structure

Link images with their relevant text sections

Prepare clean, structured outputs for multimodal embedding

📂 Project Structure
PDF PARSING/
│
├── data/
│   ├── manuals/              # Input PDF manuals
│   ├── extracted_text/       # Raw extracted text
│   ├── extracted_images/     # Images extracted from PDFs
│   ├── chunks/               # Text chunks for retrieval
│   └── metadata/             # Metadata and image-text links
│
├── pdf_parser.py             # PDF parsing (text + images)
├── text_chunker.py           # Text chunking logic
├── metadata_and_linking.py   # Metadata generation & image-text linking
└── README.md

🔹 Phase 3 Tasks Breakdown
1️⃣ PDF Parsing (Text + Images)

Extracts text and images from technical manuals

Saves text and images separately for processing

Supports real-world manuals with mixed content

Output:

Extracted text files

Extracted image files

2️⃣ Text Chunking

Splits large text into manageable chunks

Improves retrieval accuracy in RAG systems

Preserves semantic meaning across chunks

Output:

Chunked text files stored per document

3️⃣ Metadata & Image–Text Linking

Generates structured metadata for text chunks

Links images to relevant text using document-level association

Maintains layout and contextual consistency

Output:

text_metadata.json

image_text_links.json

▶️ How to Run the Phase 3 Pipeline
Step 1: Place Manuals
data/manuals/

Step 2: Run PDF Parsing
python pdf_parser.py

Step 3: Run Text Chunking
python text_chunker.py

Step 4: Generate Metadata & Linking
python metadata_and_linking.py

🧪 Practical Testing

Tested using sample technical manuals containing both text and images

Verified correct extraction of text, images, and metadata

Confirmed proper linkage between visual and textual components

🚀 Outcome

By the end of Phase 3:

Documents are fully structured

Visual and textual information is preserved

Data is ready for multimodal embedding and retrieval (Phase 4)

🔜 Next Phase

Phase 4 – Multimodal Embeddings

CLIP/SigLIP text–image alignment

Image caption generation

Image–text similarity search

🧠 Technologies Used

Python

PDF processing libraries

JSON for structured metadata storage

👤 Author

Chamith Shanaka Samarasinghe
AI/ML Intern – JW Infotech