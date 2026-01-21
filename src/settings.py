from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()


class Settings(BaseSettings):
    """Настройки приложения с валидацией через Pydantic"""

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=True)

    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    api_url: str = Field(..., alias="API_URL")
    groq_api_token: str = Field(..., alias="GROQ_API_TOKEN")
    proxy_url: str = Field(..., alias="PROXY_URL")
    model_name: str = Field(..., alias="MODEL_NAME")
    embedder_name: str = Field(..., alias="EMBEDDER_NAME")
    huggingfacehub_api_token: Optional[str] = Field(..., alias="HUGGINGFACEHUB_API_TOKEN")

    # LangSmith настройки
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: Optional[str] = Field(default=None, alias="LANGSMITH_PROJECT")
    langsmith_tracing: Optional[str] = Field(default=None, alias="LANGSMITH_TRACING")


settings = Settings()
