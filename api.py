from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import answer_question

app = FastAPI(title="JW Infotech Multimodal RAG")

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.post("/ask", response_model=Answer)
def ask(query: Query):
    return {"answer": answer_question(query.question)}
