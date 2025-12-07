"""
FastAPI бекенд для обработки запросов к RAG системе
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from rag_system import AgenticRAGSystem

# Глобальная переменная для хранения RAG системы
rag_system: Optional[AgenticRAGSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global rag_system
    # Startup
    try:
        print("Инициализация RAG системы...")
        rag_system = AgenticRAGSystem(
            qdrant_path="./qdrant_db",
            collection_name="RAG_ML_HANDBOOK"
        )
        print("RAG система успешно инициализирована")
    except Exception as e:
        print(f"Ошибка при инициализации RAG системы: {e}")
        raise
    
    yield
    
    # Shutdown
    print("Завершение работы приложения...")


app = FastAPI(
    title="RAG ML Handbook API",
    version="1.0.0",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    """Модель запроса"""
    question: str
    user_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    """Модель ответа"""
    answer: str

@app.get("/")
async def root():
    """Проверка работоспособности API"""
    return {"status": "ok", "message": "RAG ML Handbook API работает"}


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG система не инициализирована")
    return {"status": "healthy", "rag_system": "initialized"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Обработка запроса пользователя
    
    Args:
        request: Запрос с вопросом и опциональным user_id
        
    Returns:
        Ответ от RAG системы
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG система не инициализирована")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    
    try:
        answer = rag_system.query(
            question=request.question,
            user_id=request.user_id or "default"
        )
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

