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
    api_url: str = Field(default="http://localhost:8000", alias="API_URL")
    groq_api_token: str = Field(..., alias="GROQ_API_TOKEN")
    proxy_url: str = Field(..., alias="PROXY_URL")
    model_name: str = Field(default="qwen/qwen3-32b", alias="MODEL_NAME")
    embedder_name: str = Field(default="BAAI/bge-m3", alias="EMBEDDER_NAME")
    huggingfacehub_api_token: Optional[str] = Field(default=None, alias="HUGGINGFACEHUB_API_TOKEN")


settings = Settings()
