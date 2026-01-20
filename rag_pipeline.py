import os
import traceback
from typing import List, Any
from functools import lru_cache
from groq import Groq
from retriever import retrieve_context

# Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_CONTEXT_CHARS = 3500

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# Helpers
def _safe_extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("text") or item.get("content") or "").strip()
    return str(item).strip()


def _build_context(results: List[Any]) -> str:
    parts = []
    for item in results:
        text = _safe_extract_text(item)
        if text:
            parts.append(text)

    context = "\n\n".join(parts)
    return context[:MAX_CONTEXT_CHARS]


def _local_fallback_answer(context: str, question: str) -> str:
    if not context:
        return "I could not find relevant information in the indexed documents."
    return (
        "Based on the retrieved documents:\n\n"
        f"{context[:1000]}...\n\n"
        "(LLM unavailable — showing retrieved context only.)"
    )


# Cache full RAG calls
@lru_cache(maxsize=128)
def answer_question(question: str) -> str:
    """
    Production-ready RAG pipeline:
    1. Retrieve cached context
    2. Generate answer via Groq
    3. Safe fallback if anything fails
    """

    try:
        results = retrieve_context(question)
    except Exception as e:
        print("Retriever error:", e)
        traceback.print_exc()
        results = []

    context = _build_context(results)

    prompt = f"""
You are an AI assistant.
Answer ONLY using the provided context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

    if client:
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
                timeout=15
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Groq failure:", e)
            traceback.print_exc()
            return _local_fallback_answer(context, question)

    return _local_fallback_answer(context, question)
