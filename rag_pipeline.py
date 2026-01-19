# rag_pipeline.py
import os
import traceback
from typing import List, Any
from groq import Groq
from milvus_search import search_text_and_images

# Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

def _safe_extract_text(item: Any) -> str:
    """
    Robust extraction of human-readable text from many possible result item shapes:
    - dicts with keys 'content', 'text', 'page_content', 'chunk', 'caption', 'image_path', ...
    - objects (e.g. Milvus hit) with `.entity` attr that is a dict-like with fields
    - plain strings
    Returns empty string if nothing useful is found.
    """
    # 1) plain string
    if isinstance(item, str):
        return item.strip()

    # 2) dict-like
    if isinstance(item, dict):
        # prefer explicit content-like fields
        for k in ("content", "text", "page_content", "chunk", "caption", "summary"):
            v = item.get(k)
            if v:
                return str(v).strip()
        # sometimes payload is inside meta/entity
        meta = item.get("meta") or item.get("entity") or item.get("metadata")
        if isinstance(meta, dict):
            for k in ("content", "text", "page_content", "chunk"):
                v = meta.get(k)
                if v:
                    return str(v).strip()
        # fallback: look for any string-valued field
        for k, v in item.items():
            if isinstance(v, str) and len(v.strip()) > 0:
                return v.strip()
        return ""

    # 3) object with .entity or attributes (Milvus Hit object)
    # try common access patterns, but guard with hasattr
    try:
        # milvus Hit has .entity which can be dict-like
        ent = getattr(item, "entity", None)
        if ent:
            # ent may behave like a dict or have .get
            if isinstance(ent, dict):
                return _safe_extract_text(ent)
            # try attribute access
            # try common attribute names
            for attr in ("content", "text", "page_content", "chunk", "caption"):
                if hasattr(ent, attr):
                    v = getattr(ent, attr)
                    if v:
                        return str(v).strip()
            # try dictionary-like interface
            try:
                # some milvus .entity object supports .get
                v = ent.get("content") or ent.get("text")
                if v:
                    return str(v).strip()
            except Exception:
                pass
    except Exception:
        pass

    # 4) fallback: try generic str()
    try:
        s = str(item)
        if s and len(s.strip()) > 0:
            return s.strip()
    except Exception:
        pass

    return ""

def _build_context(results: List[Any]) -> str:
    parts = []
    # debug: log the first time to see actual shape (remove later if you want)
    if getattr(_build_context, "_logged", False) is False:
        try:
            print("DEBUG: sample search results (first 3):")
            for i, r in enumerate(results[:3]):
                print(f"  RESULT[{i}]: type={type(r)} repr={repr(r)[:500]}")
        except Exception:
            pass
        _build_context._logged = True

    for item in results:
        text = _safe_extract_text(item)
        if text:
            parts.append(text)
    return "\n\n".join(parts)

def _local_fallback_answer(context: str, question: str) -> str:
    if not context or context.strip() == "":
        return "I couldn't find any relevant information in the indexed manuals for that question."
    excerpt = context[:1200].strip()
    return f"Based on the retrieved manual excerpts:\n\n{excerpt}\n\n(If you want the exact page/images, I can retrieve sources.)"

def answer_question(question: str) -> str:
    """
    1) retrieve results via milvus_search.search_text_and_images
    2) build context safely without KeyError
    3) call Groq if available, otherwise fallback locally
    """
    try:
        results = search_text_and_images(question) or []
    except Exception as e:
        print("Error calling search_text_and_images:", e)
        traceback.print_exc()
        results = []

    context = _build_context(results)
    if not context:
        context = ""  # let fallback handle empty context

    prompt = f"""You are an AI assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

    # try LLM if configured
    if client:
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            # safe extraction of response text
            try:
                return response.choices[0].message.content.strip()
            except Exception:
                return str(response)
        except Exception as e:
            print("Groq call failed:", e)
            traceback.print_exc()
            # fallback to local generator
            return _local_fallback_answer(context, question)

    # no LLM configured, local fallback
    return _local_fallback_answer(context, question)
