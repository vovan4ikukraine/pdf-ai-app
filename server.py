from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional
import os
import uvicorn
from dotenv import load_dotenv

# импортируем твою функцию
# ⚠️ тут "main.py" = имя твоего файла с функцией
from pdfsRerank import rerank_and_answer

app = FastAPI(title="CV Q&A API")

load_dotenv()
API_KEY = os.getenv("API_KEY")


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401, detail="Invalid or missing API Key")

# --- Определяем структуру запроса ---


class QuestionRequest(BaseModel):
    question: str
    candidates: Optional[int] = 10
    top_k: Optional[int] = 3

# --- Структура ответа ---


class AnswerResponse(BaseModel):
    answer: str
    debug: dict

# --- Сам эндпоинт ---


# --- Эндпоинт ---
@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    answer, debug = rerank_and_answer(
        question=req.question,
        candidates=req.candidates,
        top_k=req.top_k,
    )
    return {"answer": answer, "debug": debug}



# --- Чтобы запускалось просто python server.py ---
if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
