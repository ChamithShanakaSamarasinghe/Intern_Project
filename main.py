from fastapi import FastAPI
from pydantic import BaseModel
import sys

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: AskRequest):
    print("🔥 Ask endpoint hit", flush=True)
    print(f"Question received: {request.question}", flush=True)
    sys.stdout.flush()

    return {
        "answer": f"You asked: {request.question}"
    }
