#Multimodal RAG Capstone Project

This repository documents the **end‑to‑end implementation of a Multimodal Retrieval‑Augmented Generation (RAG) system**, developed as a final capstone project. The project is organized into **six progressive phases**, each building toward a production‑ready AI system capable of understanding and answering questions from text and images.

The approach is **practical, code‑first, and industry‑oriented**, focusing on real‑world implementation rather than theoretical concepts.

---

## 📌 Phase 1 – Project Setup & Environment Configuration

### 📄 Overview

Phase 1 establishes the foundation for the Multimodal RAG system. This phase focuses on project structure, environment setup, dependency management, and validating that the development environment is ready for subsequent phases.

### 🎯 Phase Objectives

* Create a clean, modular project structure
* Set up Python virtual environment
* Install required dependencies
* Validate basic script execution

### 📂 Project Structure

```
project-root/
│
├── data/
├── logs/
├── rag_pipeline/
├── api.py
├── logger_config.py
├── requirements.txt
└── README.md
```

### ▶️ How to Run Phase 1

```bash
python --version
pip install -r requirements.txt
python -c "print('Environment Ready')"
```

### 🚀 Outcome

* Development environment successfully configured
* Project structure ready for incremental development

---

## 📌 Phase 2 – Text‑Only RAG Pipeline

### 📄 Overview

This phase implements a **basic Retrieval‑Augmented Generation pipeline** using text documents only. It validates the core RAG workflow before introducing multimodal data.

### 🎯 Phase Objectives

* Load and preprocess text documents
* Generate embeddings for text
* Store embeddings in a vector database
* Retrieve relevant chunks for a query
* Generate answers using an LLM

### 📂 Key Files

```
rag_pipeline/
├── text_loader.py
├── text_chunker.py
├── embeddings.py
├── vector_store.py
└── rag_text_only.py
```

### ▶️ How to Run Phase 2

```bash
python rag_pipeline/rag_text_only.py
```

### 🚀 Outcome

* Functional text‑only RAG pipeline
* Validated retrieval + generation flow

---

## 📌 Phase 3 – Document Processing (Multimodal RAG)

### 📄 Overview

This phase focuses on **document processing for multimodal inputs**. Text, images, and metadata are extracted from technical manuals and prepared for downstream embedding and retrieval.

### 🎯 Phase Objectives

* Parse PDFs containing text and images
* Chunk extracted text
* Preserve document metadata
* Link images with relevant text sections

### 📂 Project Structure

```
PDF_PARSING/
│
├── data/
│   ├── manuals/
│   ├── extracted_text/
│   ├── extracted_images/
│   ├── chunks/
│   └── metadata/
│
├── pdf_parser.py
├── text_chunker.py
├── metadata_and_linking.py
└── README.md
```

### ▶️ How to Run Phase 3

```bash
python pdf_parser.py
python text_chunker.py
python metadata_and_linking.py
```

### 🚀 Outcome

* Fully structured multimodal documents
* Text–image relationships preserved

---

## 📌 Phase 4 – Multimodal Embeddings

### 📄 Overview

This phase converts **text and images into a shared embedding space** using multimodal models, enabling cross‑modal retrieval.

### 🎯 Phase Objectives

* Generate embeddings for text chunks
* Generate embeddings for images
* Align text and image representations
* Store embeddings in a vector database

### 📂 Key Files

```
rag_pipeline/
├── multimodal_embeddings.py
├── image_encoder.py
├── text_encoder.py
└── vector_store.py
```

### ▶️ How to Run Phase 4

```bash
python rag_pipeline/multimodal_embeddings.py
```

### 🚀 Outcome

* Text and images embedded in a unified vector space
* Ready for multimodal retrieval

---

## 📌 Phase 5 – Multimodal RAG Pipeline

### 📄 Overview

This phase integrates **retrieval and generation across both text and images**, completing the multimodal RAG logic.

### 🎯 Phase Objectives

* Retrieve relevant text and images for a query
* Combine multimodal context
* Generate grounded responses using LLM

### 📂 Key Files

```
rag_pipeline/
├── retriever.py
├── multimodal_rag.py
└── answer_generator.py
```

### ▶️ How to Run Phase 5

```bash
python rag_pipeline/multimodal_rag.py
```

### 🚀 Outcome

* End‑to‑end multimodal RAG pipeline
* Accurate responses using text + images

---

## 📌 Phase 6 – API, Logging & Final Integration (Capstone)

### 📄 Overview

The final phase exposes the Multimodal RAG system via a **FastAPI service**, adds structured logging, and prepares the project for demonstration and evaluation.

### 🎯 Phase Objectives

* Build REST API using FastAPI
* Implement structured logging
* Add health checks
* Enable real‑time query handling

### 📂 Key Files

```
api.py
logger_config.py
rag_pipeline/
└── answer_question.py
```

### ▶️ How to Run Phase 6

```bash
uvicorn api:app --reload
```

### 🔗 Available Endpoints

* `POST /ask` – Query the Multimodal RAG system
* `GET /health` – System health check
* `GET /docs` – Interactive Swagger UI

### 🚀 Outcome

* Production‑ready Multimodal RAG API
* Fully logged and testable system

---

## 🧪 Testing & Validation

* Tested with real technical manuals
* Verified text and image retrieval
* Confirmed API responses and logging

---

## 🧠 Technologies Used

* Python
* FastAPI
* Vector Databases (Milvus)
* Multimodal Embedding Models
* LLMs (via API)
* PDF Processing Libraries

---

## 👤 Author

**Chamith Shanaka Samarasinghe**
AI/ML Intern – JW Infotech

---

## ✅ Final Note

This project demonstrates a **complete, real‑world Multimodal RAG system**, from raw documents to an API‑based intelligent assistant, following industry‑standard practices and modular design.
