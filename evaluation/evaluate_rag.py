import datetime
import json
import os
from typing import Literal

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.rag_system import AgenticRAGSystem
from src.settings import settings

load_dotenv()
os.environ["NO_PROXY"] = "localhost,127.0.0.1,example.com"

rag_system = AgenticRAGSystem(
    qdrant_path="./qdrant_db",
    collection_name="RAG_ML_HANDBOOK",
)

http_client = httpx.Client(proxy=settings.proxy_url, verify=False)
eval_llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=settings.groq_api_token,
    base_url="https://api.groq.com/openai/v1",
    http_client=http_client,
    temperature=0.0,
    max_tokens=4096,
)


class RelevanceEvaluation(BaseModel):
    """Schema for LLM-based relevance evaluation."""

    score: Literal[1, 2, 3, 4, 5] = Field(description="Score from 1 to 5")
    explanation: str = Field(description="Brief explanation of the score")


def evaluate_single_query(query: str, category: str = "basic") -> dict:
    """Evaluate a single query using LLM as judge."""

    user_id = f"eval_{hash(query)}"
    answer = rag_system.query(query, user_id=user_id)

    structured_llm = eval_llm.with_structured_output(RelevanceEvaluation)

    category_specific_criteria = {
        "trick_questions": """
ОСОБЫЕ КРИТЕРИИ ДЛЯ ВОПРОСОВ С ПОДВОХАМИ:
- Правильно ли модель выявила некорректность или ложность утверждения?
- Объяснила ли она, почему утверждение неверно?
- Предоставила ли альтернативы или правильную информацию?
- Избежала ли галлюцинаций и подтверждения неверных предпосылок?
        """,
        "non_existent": """
ОСОБЫЕ КРИТЕРИИ ДЛЯ НЕСУЩЕСТВУЮЩИХ КОНЦЕПЦИЙ:
- Правильно ли модель определила, что концепция не существует?
- Объяснила ли она, что такого алгоритма/метода нет?
- Предложила ли похожие реальные алгоритмы?
- Избежала ли изобретения несуществующих объяснений?
        """,
        "advanced": """
КРИТЕРИИ ДЛЯ ПРОДВИНУТЫХ ВОПРОСОВ:
- Демонстрирует ли ответ глубокое понимание темы?
- Рассматривает ли различные аспекты и нюансы?
- Приводит ли примеры применения?
- Объясняет ли ограничения и компромиссы?
        """,
        "comparative": """
КРИТЕРИИ ДЛЯ СРАВНИТЕЛЬНЫХ ВОПРОСОВ:
- Проводится ли честное сравнение с учетом плюсов и минусов?
- Указываются ли условия применимости каждого метода?
- Приводятся ли конкретные примеры использования?
        """,
    }

    special_criteria = category_specific_criteria.get(category, "")

    answer_eval_prompt = f"""
Оцени качество ответа на запрос пользователя.
Категория вопроса: {category}

ЗАПРОС: {query}

ОТВЕТ: {answer}

{special_criteria}

Общие критерии оценки (1-5):
1 - не отвечает на вопрос
2 - частично отвечает, много неточностей
3 - отвечает, но неполно или с ошибками
4 - хорошо отвечает, минимальные недочеты
5 - отлично и полно отвечает на вопрос
"""

    try:
        answer_eval: RelevanceEvaluation = structured_llm.invoke(answer_eval_prompt)

        return {
            "query": query,
            "category": category,
            "answer": answer,
            "answer_score": answer_eval.score,
            "answer_explanation": answer_eval.explanation,
            "overall_score": answer_eval.score,
        }

    except Exception as e:
        return {"query": query, "category": category, "error": str(e), "answer_score": 0, "overall_score": 0}


test_queries_categories = {
    "basic": [
        "Что такое L2 регуляризация?",
        "Объясни принцип работы случайного леса",
        "Как работает алгоритм k-means?",
        "Что такое градиентный спуск?",
        "Что такое кросс-валидация?",
        "Как работает логистическая регрессия?",
    ],
    "advanced": [
        "Сравни эффективность L1 и L2 регуляризации при наличии мультиколлинеарности в данных",
        "Когда использовать Random Forest вместо Gradient Boosting и почему?",
        "Как выбрать оптимальное количество кластеров в k-means для высокомерных данных?",
        # "Объясни различия между SGD, Adam и RMSprop оптимизаторами",
        "Как стратифицированная кросс-валидация влияет на оценку моделей с несбалансированными классами?",
        "В каких случаях логистическая регрессия может переобучаться и как это предотвратить?",
    ],
    "comparative": [
        "В чем принципиальное различие между случайным лесом и градиентным бустингом?",
        "Когда лучше использовать SVM вместо нейронных сетей?",
        "Сравни DBSCAN и k-means для кластеризации аномальных данных",
        # "Что лучше для текстовой классификации: TF-IDF + SVM или BERT?",
        # "Сравни эффективность ансамблей и одиночных моделей",
    ],
    "contextual": [
        # "Как применить машинное обучение для прогнозирования оттока клиентов в банке?",
        "Какие алгоритмы лучше всего подходят для рекомендательных систем e-commerce?",
        "Как построить систему детекции мошенничества в реальном времени?",
        "Какие методы использовать для анализа настроений в социальных сетях?",
        # "Как создать систему компьютерного зрения для медицинской диагностики?",
    ],
    "trick_questions": [
        "Почему нейронные сети всегда лучше линейных моделей?",
        # "Правда ли, что больше данных всегда означает лучшую модель?",
        # "Можно ли использовать точность (accuracy) как единственную метрику для всех задач?",
        "Всегда ли более сложная модель дает лучшие результаты?",
        # "Правда ли, что deep learning решает все проблемы машинного обучения?",
        "Можно ли применить k-means для кластеризации текстовых данных напрямую?",
        "Правда ли, что корреляция всегда означает причинно-следственную связь?",
    ],
    "edge_cases": [
        "Как работает машинное обучение при малом количестве данных (few-shot learning)?",
        "Что делать, если в данных 99% одного класса и 1% другого?",
        # "Как обучить модель, если данные постоянно изменяются (concept drift)?",
        "Можно ли применить машинное обучение к задачам, где нет исторических данных?",
        # "Как интерпретировать результаты черного ящика модели для медицинской диагностики?",
    ],
    "non_existent": [
        "Что такое алгоритм квантового k-means?",
        # "Как работает нейронная сеть Шредингера?",
        "Объясни принцип работы голографической регрессии",
        "Что такое эмоциональная кластеризация данных?",
        "Как применить алгоритм временной дефрагментации к машинному обучению?",
    ],
}


# Utility functions
def get_all_test_queries() -> list:
    test_queries = []
    for category, queries in test_queries_categories.items():
        for query in queries:
            test_queries.append({"query": query, "category": category})
    return test_queries


def run_full_evaluation() -> None:
    test_queries = get_all_test_queries()
    print(f"Оценка {len(test_queries)} вопросов из {len(test_queries_categories)} категорий...")

    results = []
    for i, query_data in enumerate(test_queries, 1):
        query_text = query_data["query"]
        category = query_data["category"]

        if i % 5 == 1 or i == len(test_queries):
            print(f"\rПрогресс: {i}/{len(test_queries)}", end="", flush=True)

        result = evaluate_single_query(query_text, category)
        results.append(result)

    _print_evaluation_summary(results)
    _save_results(results)


def _print_evaluation_summary(results):
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print("Нет валидных результатов для анализа")
        return

    avg_answer = sum(r["answer_score"] for r in valid_results) / len(valid_results)
    avg_overall = sum(r["overall_score"] for r in valid_results) / len(valid_results)

    print(f"\n{'=' * 60}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'=' * 60}")
    print(f"Успешно оценено запросов: {len(valid_results)}/{len(results)}")
    print(f"Среднее качество ответов: {avg_answer:.2f}/5")
    print(f"Общая средняя оценка: {avg_overall:.2f}/5")


def _save_results(results):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {filename}")


def evaluate_category(category_name: str) -> list:
    if category_name not in test_queries_categories:
        print(f"Категория '{category_name}' не найдена. Доступные: {list(test_queries_categories.keys())}")
        return

    queries = test_queries_categories[category_name]
    print(f"Оценка категории '{category_name}': {len(queries)} вопросов")

    results = []
    for i, query in enumerate(queries, 1):
        print(f"\rПрогресс: {i}/{len(queries)}", end="", flush=True)
        result = evaluate_single_query(query, category_name)
        results.append(result)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{category_name}_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в {filename}")

    return results


if __name__ == "__main__":
    evaluate_category("non_existent")
    # run_full_evaluation()
