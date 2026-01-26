import os
import logging
import traceback
from typing import List, Any
from functools import lru_cache

from groq import Groq
from retriever import retrieve_context
from language_utils import normalize_question, translate_answer


# Logger Configuration
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for deeper tracing
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("RAGPipeline")

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_CONTEXT_CHARS = 3500

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Helper Functions
def _safe_extract_text(item: Any) -> str:
    """
    Safely extract readable text from:
    - strings
    - dicts (text chunks, metadata)
    - image references
    """
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        for key in ("text", "content", "page_content"):
            value = item.get(key)
            if value:
                return str(value).strip()

        if "image_path" in item:
            return f"[Related image: {item['image_path']}]"

    return ""


def _build_context(results: List[Any]) -> str:
    """
    Build a clean context string with length control
    """
    parts = []

    for item in results:
        text = _safe_extract_text(item)
        if text:
            parts.append(text)

    context = "\n\n".join(parts)
    return context[:MAX_CONTEXT_CHARS]


def _local_fallback_answer(context: str) -> str:
    """
    Used when LLM is unavailable or fails
    """
    if not context:
        return "I could not find relevant information in the indexed documents."

    return (
        "Based on the retrieved documents:\n\n"
        f"{context[:1000]}...\n\n"
        "(LLM unavailable — showing retrieved context only.)"
    )

# Main RAG Pipeline
@lru_cache(maxsize=128)
def answer_question(question: str) -> str:
    """
    Production-ready RAG pipeline

    Features:
    - Multilingual support
    - Performance caching
    - Structured logging
    - Safe fallbacks
    - Context length control
    """

    logger.info("Received question")

    # Language normalization
    normalized_question, original_lang = normalize_question(question)
    logger.debug("Normalized question: %s", normalized_question)

    # Retrieval
    try:
        results = retrieve_context(normalized_question)
        logger.info("Retrieved %d context items", len(results))
    except Exception as e:
        logger.error("Retriever failed", exc_info=True)
        results = []

    context = _build_context(results)

    prompt = f"""
You are an AI assistant.
Answer ONLY using the provided context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{normalized_question}
"""

    # LLM call
    if client:
        try:
            logger.info("Calling Groq model: %s", GROQ_MODEL)

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )

            answer = response.choices[0].message.content.strip()
            logger.info("LLM response generated successfully")

            return translate_answer(answer, original_lang)

        except Exception:
            logger.error("Groq LLM call failed", exc_info=True)

    # Safe fallback
    logger.warning("Using fallback answer (LLM unavailable)")
    fallback = _local_fallback_answer(context)
    return translate_answer(fallback, original_lang)
