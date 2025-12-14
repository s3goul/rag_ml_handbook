"""
Модуль для инициализации и работы с RAG системой
"""

import os
import yaml
import httpx
import time
from typing import Dict, List, Tuple, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langsmith import traceable
from .settings import settings

import tiktoken
_ENC = tiktoken.get_encoding("cl100k_base")
def _tok(text: str) -> int:
    return len(_ENC.encode(text or ""))


class AgenticRAGSystem:
    """Класс для управления RAG системой"""

    def __init__(self, qdrant_path: str = "./qdrant_db", collection_name: str = "RAG_ML_HANDBOOK"):
        """
        Инициализация RAG системы

        Args:
            qdrant_path: Путь к базе данных Qdrant
            collection_name: Имя коллекции в Qdrant
        """

        # Инициализация Embedder
        self.embedder = HuggingFaceEmbeddings(
            model_name=settings.embedder_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Инициализация PROXY
        http_client = httpx.Client(proxy=settings.proxy_url, verify=False)

        # Инициализация LLM клиента
        self.llm = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.groq_api_token,
            base_url="https://api.groq.com/openai/v1",
            http_client=http_client,
            temperature=0.0,
            max_tokens=4096,
        )

        # Инициализация Qdrant
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

        # Создание векторного хранилища
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embedder,
        )

        # Единожды загружаем промпты
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f) or {}
        self.system_prompt = self.prompts.get("ReActPrompt", "")
        if not self.system_prompt:
            raise ValueError("Промпт не загрузился!")
        self.summary_prompt = self.prompts.get("DialogSummaryPrompt", "")
        if not self.summary_prompt:
            raise ValueError("Промпт суммаризации не загрузился!")
        # Жёсткий лимит на длину отправляемого контекста (токены)
        self.max_context_len_tokens = 6000
        self.max_summary_tokens = 1000
        self.per_chunk_tokens = 700
        self.total_retrieval_tokens = 3500

        # Создание инструмента для поиска
        self.retrieve_tool = self._create_retrieve_tool()

        # Создание агента
        self.checkpointer = InMemorySaver()
        self.agent = self._create_agent()

        # Состояние диалогов в памяти: накопительная summary по пользователям
        # Формат: {user_id: {"summary": str}}
        self.dialog_state: Dict[str, Dict[str, object]] = {}
        # Трек последнего запроса для простого rate-limit (секунды)
        # Формат: {user_id: last_ts}
        self.last_request_ts: Dict[str, float] = {}

    def _approx_tokens(self, text: str) -> int:
        """Подсчёт токенов (точно через tiktoken, иначе грубо)."""
        return _tok(text)

    def _trim_to_tokens(self, text: str, limit: int) -> str:
        """Обрезает текст до указанного лимита токенов (по tiktoken/грубой оценке)."""
        if self._approx_tokens(text) <= limit:
            return text
        # грубый binary-like trim по символам с запасом
        low, high = 0, len(text)
        target_chars = max(1, limit * 4)  # стартовая оценка
        high = min(high, target_chars)
        if high <= low:
            return text[:max(1, target_chars)]
        while low < high:
            mid = (low + high) // 2
            if self._approx_tokens(text[:mid]) > limit:
                high = mid - 1
            else:
                low = mid + 1
        trimmed = text[:high]
        if self._approx_tokens(trimmed) > limit and len(trimmed) > 1:
            trimmed = trimmed[:-1]
        return trimmed

    def _create_retrieve_tool(self):
        """Создание инструмента для поиска в векторной базе"""
        vector_store = self.vector_store  # Сохраняем ссылку для замыкания

        @tool(response_format="content")
        @traceable
        def retrieve_context(query: str) -> str:
            """Используй этот инструмент для поиска информации в учебнике по машинному обучению.

            Args:
                query: Поисковый запрос на русском или английском языке

            Returns:
                Найденные релевантные фрагменты из учебника
            """
            # Сбрасываем счётчик токенов ретривала
            self.last_retrieve_tokens = 0
            retrieved_docs = vector_store.similarity_search(query, k=5)
            chunks: List[str] = []
            total_limit = self.total_retrieval_tokens
            per_chunk_limit = self.per_chunk_tokens
            total_used = 0
            for doc in retrieved_docs:
                text = doc.page_content or ""
                trimmed = self._trim_to_tokens(text, per_chunk_limit)
                tks = self._approx_tokens(trimmed)
                if total_used + tks > total_limit:
                    # если даже текущий trimmed переполняет общий лимит — пропускаем
                    break
                total_used += tks
                self.last_retrieve_tokens += tks
                chunks.append(f"Content: {trimmed}")
                if total_used >= total_limit:
                    break
            serialized = "\n\n".join(chunks)
            return serialized

        return retrieve_context

    def _create_agent(self):
        """Создание агента с инструментами"""
        agent = create_agent(
            model=self.llm,
            tools=[self.retrieve_tool],
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
        )
        return agent

    def _get_state(self, user_id: str) -> Dict[str, object]:
        """Возвращает состояние диалога пользователя"""
        if user_id not in self.dialog_state:
            self.dialog_state[user_id] = {"summary": ""}
        return self.dialog_state[user_id]

    def _save_state(self, user_id: str, state: Dict[str, object]):
        """Сохраняет состояние диалога пользователя"""
        self.dialog_state[user_id] = state

    def _update_summary(self, state: Dict[str, object], question: str, answer: str) -> Dict[str, object]:
        """
        Обновляет накопительную summary, добавляя новый обмен вопрос-ответ.
        Если summary перерастает лимит, она ужимается и обрезается.
        """
        prev_summary: str = state.get("summary", "")

        # Делаем отдельную компактную свёртку только текущего шага (Q/A)
        to_summarize = "\n".join(
            [
                f"user: {question}",
                f"assistant: {answer}",
            ]
        )

        summary_message = self.llm.invoke(
            [
                ("system", self.summary_prompt),
                ("user", to_summarize),
            ]
        )
        step_summary = summary_message.content if hasattr(summary_message, "content") else ""

        # Ограничиваем свёртку шага до ~200 токенов
        step_summary = self._trim_to_tokens(step_summary, 200)

        # Конкатенируем к предыдущей summary и обрезаем по лимиту (старое выкидываем)
        combined = "\n".join(s for s in [prev_summary, step_summary] if s)
        if self._approx_tokens(combined) > self.max_summary_tokens:
            combined = self._trim_to_tokens(combined, self.max_summary_tokens)

        state["summary"] = combined
        return state

    def _build_messages(self, user_id: str, question: str) -> Tuple[List[Tuple[str, str]], Dict[str, object]]:
        """Формирует список сообщений для агента с учетом summary"""
        state = self._get_state(user_id)
        summary = state.get("summary", "")

        messages: List[Tuple[str, str]] = []
        if summary:
            messages.append(("system", f"Краткое резюме диалога пользователя: {summary}"))
        messages.append(("user", question))
        return messages, state

    @traceable
    def query(self, question: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Обработка запроса пользователя

        Args:
            question: Вопрос пользователя
            user_id: ID пользователя для сохранения контекста

        Returns:
            Ответ агента
        """
        try:
            now = time.time()
            # Создаем конфигурацию: новый thread_id на каждый вызов,
            # но прокидываем user_id для трейсинга/метаданных
            thread_id = f"{user_id}_{int(time.time() * 1000)}"
            config = {
                "configurable": {"thread_id": thread_id},
                "metadata": {"user_id": user_id},
            }

            # Формируем сообщения с учетом summary
            messages, state = self._build_messages(user_id, question)
            # Подсчёт токенов отдельных частей
            system_tokens = self._approx_tokens(self.system_prompt)
            summary_tokens = self._approx_tokens(state.get("summary", ""))
            question_tokens = self._approx_tokens(question)

            # Простой прогноз общего объёма (учитываем потолок ретривала)
            estimated_total = system_tokens + summary_tokens + question_tokens + self.total_retrieval_tokens
            if estimated_total > self.max_context_len_tokens:
                self.last_request_ts[user_id] = now
                return {"answer": "Запрос слишком длинный. Укоротите запрос и повторите."}

            # Вызываем агента
            result = self.agent.invoke({"messages": messages}, config=config)

            # Извлекаем ответ из последнего сообщения
            if result and "messages" in result:
                # Последнее сообщение от ассистента
                llm_message = result["messages"][-1]
                if hasattr(llm_message, "content"):
                    answer = llm_message.content
                else:
                    self.last_request_ts[user_id] = now
                    return {"answer": "Извините, не удалось получить ответ."}

                # Обновляем summary в состоянии пользователя
                new_state = self._update_summary(state, question, answer)
                self._save_state(user_id, new_state)
                # Фиксируем время последнего успешного запроса
                self.last_request_ts[user_id] = now

                return {"answer": answer}

            self.last_request_ts[user_id] = now
            return {"answer": "Извините, не удалось получить ответ."}

        except Exception as e:
            msg = str(e)
            # Обработка превышения лимита токенов/TPM (413 / rate_limit_exceeded)
            if "rate_limit_exceeded" in msg or "Error code: 413" in msg:
                now_err = time.time()
                last_ts = self.last_request_ts.get(user_id)
                self.last_request_ts[user_id] = now_err
                if last_ts is not None:
                    remaining = max(1, int(70 - (now_err - last_ts)))
                else:
                    remaining = 70
                return {
                    "answer": (
                        f"Превышен лимит токенов/скорости модели. "
                        f"Подождите {remaining} сек и попробуйте задать вопрос короче."
                    )
                }
            return {"answer": f"Произошла ошибка при обработке запроса: {msg}"}