"""
Модуль для инициализации и работы с RAG системой
"""
import os
import yaml
import httpx
import time
from typing import Dict, List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI


class AgenticRAGSystem:
    """Класс для управления RAG системой"""
    
    def __init__(
        self,
        qdrant_path: str = "./qdrant_db",
        collection_name: str = "RAG_ML_HANDBOOK"
    ):
        """
        Инициализация RAG системы
        
        Args:
            qdrant_path: Путь к базе данных Qdrant
            collection_name: Имя коллекции в Qdrant
        """
        # Проверка переменных окружения
        required_vars = {
            "HUGGINGFACEHUB_API_TOKEN": "Hugging Face API токен",
            "GROQ_API_TOKEN": "GROQ API токен",
            "PROXY_URL": "URL прокси",
            "MODEL_NAME": "Имя модели",
            "EMBEDDER_NAME": "Имя embedder модели"
        }
        
        for var_name, description in required_vars.items():
            if not os.getenv(var_name):
                raise ValueError(f"{var_name} ({description}) должен быть установлен в переменных окружения")

        # Инициализация Embedder
        self.embedder = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDER_NAME"),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Инициализация PROXY
        http_client = httpx.Client(
            proxy=os.getenv("PROXY_URL"),
            verify=False
        )

        # Инициализация LLM клиента
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME"),
            api_key=os.environ["GROQ_API_TOKEN"],
            base_url="https://api.groq.com/openai/v1",
            http_client=http_client,
            temperature=0.3,
            max_tokens=1024
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
        
        # Загрузка промпта
        self.system_prompt = self._load_prompt()
        self.summary_prompt = self._load_summary_prompt()
        # Жёсткий лимит на длину отправляемого контекста (токены, консервативная оценка)
        self.max_context_len_tokens = 2200

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
        """Консервативная оценка количества токенов (≈2 символа на токен для RU/EN)."""
        if not text:
            return 0
        return max(1, len(text) // 2)
    
    def _load_prompt(self) -> str:
        """Загрузка промпта из prompts.yaml"""
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        
        ReActPrompt = prompts.get("ReActPrompt", "")
        if not ReActPrompt:
            raise ValueError("Промпт не загрузился!")
        
        return ReActPrompt

    def _load_summary_prompt(self) -> str:
        """Загрузка промпта для суммаризации диалога"""
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        summary_prompt = prompts.get("DialogSummaryPrompt", "")
        if not summary_prompt:
            raise ValueError("Промпт суммаризации не загрузился!")
        return summary_prompt
    def _create_retrieve_tool(self):
        """Создание инструмента для поиска в векторной базе"""
        vector_store = self.vector_store  # Сохраняем ссылку для замыкания
        
        @tool(response_format="content")
        def retrieve_context(query: str) -> str:
            """Используй этот инструмент для поиска информации в учебнике по машинному обучению.
            
            Args:
                query: Поисковый запрос на русском или английском языке
                
            Returns:
                Найденные релевантные фрагменты из учебника
            """
            retrieved_docs = vector_store.similarity_search(query, k=5)
            serialized = "\n\n".join(
                (f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized
        
        return retrieve_context
    
    def _create_agent(self):
        """Создание агента с инструментами"""
        agent = create_agent(
            model=self.llm,
            tools=[self.retrieve_tool],
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer
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
        Если summary перерастает лимит, она ужимается через LLM и жестко обрезается.
        """
        prev_summary: str = state.get("summary", "")

        # Собираем текст для сжатия: прошлое summary + новая пара Q/A
        parts = []
        if prev_summary:
            parts.append(f"Текущее резюме: {prev_summary}")
        parts.append(f"user: {question}")
        parts.append(f"assistant: {answer}")
        to_summarize = "\n".join(parts)

        summary_message = self.llm.invoke(
            [
                ("system", self.summary_prompt),
                ("user", to_summarize),
            ]
        )
        new_summary = summary_message.content if hasattr(summary_message, "content") else prev_summary

        # Жесткое ограничение длины summary по токенам (консервативно)
        max_summary_tokens = 1500
        if self._approx_tokens(new_summary) > max_summary_tokens:
            # Обрезаем по символам пропорционально, оставляя небольшой запас
            max_chars = max_summary_tokens * 2
            new_summary = new_summary[:max_chars]

        state["summary"] = new_summary
        return state

    def _shrink_messages_if_needed(
        self,
        messages: List[Tuple[str, str]],
        state: Dict[str, object],
        user_id: str
    ) -> Tuple[List[Tuple[str, str]], Dict[str, object]]:
        """
        Контролирует размер контекста перед вызовом LLM.
        Теперь в контексте только summary и текущий вопрос:
          - Если длина > max_context_len_tokens, summary жёстко укорачивается.
          - Вопрос при необходимости также укорачивается.
        """
        def total_tokens(msgs: List[Tuple[str, str]]) -> int:
            return sum(self._approx_tokens(m[1]) for m in msgs)

        if total_tokens(messages) <= self.max_context_len_tokens:
            return messages, state

        # Усечение summary и вопроса (консервативные лимиты)
        max_summary_tokens = 1500
        max_question_tokens = 400

        summary = state.get("summary", "")
        question_role, question_content = messages[-1]

        if summary and self._approx_tokens(summary) > max_summary_tokens:
            summary = summary[: max_summary_tokens * 2]  # грубо по символам

        if self._approx_tokens(question_content) > max_question_tokens:
            question_content = question_content[: max_question_tokens * 2]  # грубо по символам

        rebuilt: List[Tuple[str, str]] = []
        if summary:
            rebuilt.append(("system", f"Краткое резюме диалога пользователя: {summary}"))
        rebuilt.append((question_role, question_content))

        return rebuilt, state

    def _build_messages(self, user_id: str, question: str) -> Tuple[List[Tuple[str, str]], Dict[str, object]]:
        """Формирует список сообщений для агента с учетом summary"""
        state = self._get_state(user_id)
        summary = state.get("summary", "")

        messages: List[Tuple[str, str]] = []
        if summary:
            messages.append(
                (
                    "system",
                    f"Краткое резюме диалога пользователя: {summary}"
                )
            )
        messages.append(("user", question))
        return messages, state
    
    def query(self, question: str, user_id: str = "default") -> str:
        """
        Обработка запроса пользователя
        
        Args:
            question: Вопрос пользователя
            user_id: ID пользователя для сохранения контекста
            
        Returns:
            Ответ агента
        """
        try:
            # Простой анти-спам и TPM-защита: минимум 70 секунд между запросами пользователя
            now = time.time()
            last_ts = self.last_request_ts.get(user_id)
            if last_ts is not None:
                delta = now - last_ts
                cooldown = 70 - delta
                if cooldown > 0:
                    return f"Подождите {int(cooldown)} сек, чтобы не превысить лимиты модели."

            # Создаем конфигурацию с user_id для сохранения истории
            # Создаем конфигурацию с user_id для сохранения истории
            config = {"configurable": {"thread_id": user_id}}

            # Формируем сообщения с учетом summary
            messages, state = self._build_messages(user_id, question)
            # Контроль длины контекста
            messages, state = self._shrink_messages_if_needed(messages, state, user_id)

            # Вызываем агента
            result = self.agent.invoke(
                {"messages": messages},
                config=config
            )

            # Извлекаем ответ из последнего сообщения
            if result and "messages" in result:

                # Последнее сообщение от ассистента
                llm_message = result["messages"][-1]
                if hasattr(llm_message, "content"):
                    answer = llm_message.content
                else:
                    return "Извините, не удалось получить ответ."

                # Обновляем summary в состоянии пользователя
                new_state = self._update_summary(state, question, answer)
                self._save_state(user_id, new_state)

                # Обновляем время последнего запроса (даже при успехе)
                self.last_request_ts[user_id] = now

                return answer

            return "Извините, не удалось получить ответ."
            
        except Exception as e:
            msg = str(e)
            # Обработка превышения лимита токенов/TPM (413 / rate_limit_exceeded)
            if "rate_limit_exceeded" in msg or "Error code: 413" in msg:
                # Пытаемся сообщить, сколько осталось до следующей попытки
                last_ts = self.last_request_ts.get(user_id)
                now_err = time.time()
                # фиксируем даже неуспешный вызов, чтобы пользователь не спамил
                self.last_request_ts[user_id] = now_err
                if last_ts is not None:
                    remaining = max(1, int(70 - (now_err - last_ts)))
                else:
                    remaining = 70
                return (
                    f"Превышен лимит токенов/скорости модели (413). "
                    f"Подождите {remaining} сек и попробуйте задать вопрос короче."
                )
            return f"Произошла ошибка при обработке запроса: {msg}"
