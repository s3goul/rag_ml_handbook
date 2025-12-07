"""
Модуль для инициализации и работы с RAG системой
"""
import os
import yaml
import httpx
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
        
        # Создание инструмента для поиска
        self.retrieve_tool = self._create_retrieve_tool()
        
        # Создание агента
        self.checkpointer = InMemorySaver()
        self.agent = self._create_agent()
    
    def _load_prompt(self) -> str:
        """Загрузка промпта из prompts.yaml"""
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        
        ReActPrompt = prompts.get("ReActPrompt", "")
        if not ReActPrompt:
            raise ValueError("Промпт не загрузился!")
        
        return ReActPrompt
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
            # Создаем конфигурацию с user_id для сохранения истории
            config = {"configurable": {"thread_id": user_id}}
            
            # Вызываем агента
            result = self.agent.invoke(
                {"messages": [("user", question)]},
                config=config
            )
            
            # Извлекаем ответ из последнего сообщения
            if result and "messages" in result:

                # Последнее сообщение от ассистента
                llm_message = result["messages"][-1]
                if hasattr(llm_message, "content"):
                    return llm_message.content
                else:
                    return "Извините, не удалось получить ответ."

            return "Извините, не удалось получить ответ."
            
        except Exception as e:
            return f"Произошла ошибка при обработке запроса: {str(e)}"
