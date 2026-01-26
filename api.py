from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
import uuid
import os

from rag_pipeline import answer_question
from logger_config import setup_logger

app = FastAPI(title="JW Infotech Multimodal RAG")
logger = setup_logger()

logger.info("api.py loaded")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[REQUEST_START] id={request_id} method={request.method} path={request.url.path} client={request.client.host if request.client else 'unknown'}")

    response = await call_next(request)

    duration = round(time.time() - start_time, 3)
    logger.info(f"[REQUEST_END] id={request_id} status={response.status_code} duration={duration}s")

    return response


class Query(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


@app.get("/health")
def health():
    logger.info("[HEALTH] called")
    status = {"api": "ok", "milvus": "down", "llm": "down"}

    try:
        from pymilvus import connections
        if connections.has_connection("default"):
            status["milvus"] = "ok"
    except Exception:
        logger.debug("Milvus check failed or not configured")

    if os.getenv("GROQ_API_KEY"):
        status["llm"] = "ok"

    return status


@app.post("/ask", response_model=Answer)
def ask(query: Query):
    logger.info("🔥 /ask endpoint hit")
    start_time = time.time()
    logger.info(f"[RAG_QUERY] question='{query.question[:200]}'")

    try:
        answer = answer_question(query.question)
        duration = round(time.time() - start_time, 3)
        logger.info(f"[RAG_SUCCESS] duration={duration}s")
        return {"answer": answer}
    except Exception as e:
        logger.error("[RAG_ERROR] Exception while answering", exc_info=True)
        raise