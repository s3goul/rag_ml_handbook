import yaml
import os
from dotenv import load_dotenv
import httpx
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import json
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
os.environ["NO_PROXY"] = "localhost,127.0.0.1,example.com" #OPTIONAL
GROQ_API_TOKEN = os.getenv("GROQ_API_TOKEN")
PROXY_URL = os.getenv("PROXY_URL")

with open("prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)

ReActSysPrompt = prompts["ReActPrompt"]

embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
)

COLLECTION_NAME = "RAG_ML_HANDBOOK"

client = QdrantClient(url="localhost", port=6338)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedder,
)


http_client = httpx.Client(proxy=PROXY_URL, verify=False)

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_TOKEN"],
    base_url="https://api.groq.com/openai/v1",
    http_client=http_client,
    temperature=0.0,
    max_tokens=2048,
)


class RelevanceEvaluation(BaseModel):
    """Schema for relevance evaluation"""

    score: Literal[1, 2, 3, 4, 5] = Field(description="Score from 1 to 5")
    explanation: str = Field(description="Brief explanation of the score")


@tool(response_format="content")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join((f"Content: {doc.page_content}") for doc in retrieved_docs)
    return serialized


def evaluate_single_query(query: str):
    """Evaluate a single query using LLM as judge"""

    # Step 1: Get retrieved chunks
    retrieved_docs = vector_store.similarity_search(query, k=5)
    chunks = [doc.page_content for doc in retrieved_docs]

    # Step 2: Get agent response
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": f"eval_{hash(query)}"}}

    agent = create_agent(model=llm, tools=[retrieve_context], system_prompt=ReActSysPrompt, checkpointer=checkpointer)

    agent_output = agent.invoke(
        {
            "messages": [{"role": "user", "content": query}],
            "config": config,
        },
        config=config,
    )

    answer = agent_output["messages"][-1].content

    # Step 3: Evaluate chunks relevance using structured output
    chunks_text = "\n\n".join([f"Chunk {i + 1}: {chunk[:300]}..." for i, chunk in enumerate(chunks)])

    # Create structured LLM for evaluation
    structured_llm = llm.with_structured_output(RelevanceEvaluation)

    chunk_eval_prompt = f"""
Оцени релевантность найденных фрагментов для запроса.

ЗАПРОС: {query}

ФРАГМЕНТЫ:
{chunks_text}

Критерии оценки (1-5):
1 - совершенно нерелевантны
2 - слабо связаны с запросом
3 - частично релевантны
4 - в основном релевантны
5 - полностью релевантны и достаточны для ответа
"""

    # Step 4: Evaluate answer relevance
    answer_eval_prompt = f"""
Оцени качество ответа на запрос пользователя.

ЗАПРОС: {query}

ОТВЕТ: {answer}

Критерии оценки (1-5):
1 - не отвечает на вопрос
2 - частично отвечает, много неточностей
3 - отвечает, но неполно или с ошибками
4 - хорошо отвечает, минимальные недочеты
5 - отлично и полно отвечает на вопрос
"""

    try:
        # Evaluate chunks
        chunk_eval: RelevanceEvaluation = structured_llm.invoke(chunk_eval_prompt)

        # Evaluate answer
        answer_eval: RelevanceEvaluation = structured_llm.invoke(answer_eval_prompt)

        return {
            "query": query,
            "chunks": chunks,
            "answer": answer,
            "chunk_score": chunk_eval.score,
            "chunk_explanation": chunk_eval.explanation,
            "answer_score": answer_eval.score,
            "answer_explanation": answer_eval.explanation,
            "overall_score": (chunk_eval.score * 0.4 + answer_eval.score * 0.6),
        }

    except Exception as e:
        return {"query": query, "error": str(e), "chunk_score": 0, "answer_score": 0, "overall_score": 0}


# Test queries
test_queries = [
    "Что такое L2 регуляризация?",
    "Объясни принцип работы случайного леса",
    "Как работает алгоритм k-means?",
    "Что такое градиентный спуск?",
    "Что такое кросс-валидация?",
    "Как работает логистическая регрессия?",
]

if __name__ == "__main__":
    print("Запуск оценки RAG системы...")

    results = []
    for i, query in enumerate(test_queries):
        print(f"\nОценка запроса {i + 1}/{len(test_queries)}: {query}")
        result = evaluate_single_query(query)
        results.append(result)

        if "error" not in result:
            print(f"Релевантность фрагментов: {result['chunk_score']}/5")
            print(f"Качество ответа: {result['answer_score']}/5")
            print(f"Общая оценка: {result['overall_score']:.1f}/5")
        else:
            print(f"Ошибка: {result}")

    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_chunk = sum(r["chunk_score"] for r in valid_results) / len(valid_results)
        avg_answer = sum(r["answer_score"] for r in valid_results) / len(valid_results)
        avg_overall = sum(r["overall_score"] for r in valid_results) / len(valid_results)

        print(f"\n{'=' * 50}")
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print(f"{'=' * 50}")
        print(f"Успешно оценено запросов: {len(valid_results)}/{len(test_queries)}")
        print(f"Средняя релевантность фрагментов: {avg_chunk:.2f}/5")
        print(f"Среднее качество ответов: {avg_answer:.2f}/5")
        print(f"Общая средняя оценка: {avg_overall:.2f}/5")

        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\nРезультаты сохранены в evaluation_results.json")
