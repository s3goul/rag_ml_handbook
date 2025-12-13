"""
Телеграм бот для взаимодействия с RAG системой
"""
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import httpx
from .settings import settings

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Константы
API_QUERY_ENDPOINT = f"{settings.api_url}/query"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_message = (
        "👋 Привет! Я бот-ассистент по машинному обучению.\n\n"
        "Я могу помочь вам с вопросами по ML и Data Science, используя "
        "базу знаний из учебника Яндекса по машинному обучению.\n\n"
        "Просто отправьте мне свой вопрос, и я постараюсь на него ответить!"
    )
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_message = (
        "📚 Доступные команды:\n\n"
        "/start - Начать работу с ботом\n"
        "/help - Показать это сообщение\n\n"
        "💡 Просто отправьте мне вопрос по машинному обучению, "
        "и я найду ответ в учебнике!"
    )
    await update.message.reply_text(help_message)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user_id = str(update.effective_user.id)
    question = update.message.text
    
    if not question or not question.strip():
        await update.message.reply_text("Пожалуйста, отправьте вопрос.")
        return
    
    # Отправляем сообщение о том, что обрабатываем запрос
    processing_message = await update.message.reply_text("🤔 Думаю над ответом...")
    
    try:
        # Отправляем запрос в API
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                API_QUERY_ENDPOINT,
                json={
                    "question": question,
                    "user_id": user_id
                }
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("answer", "Не удалось получить ответ.")
        
        # Удаляем сообщение "Думаю над ответом"
        await processing_message.delete()
        
        # Отправляем ответ пользователю
        # Разбиваем длинные ответы на части (Telegram имеет лимит 4096 символов)
        max_length = 4000
        if len(answer) > max_length:
            # Разбиваем на части
            parts = [answer[i:i+max_length] for i in range(0, len(answer), max_length)]
            for i, part in enumerate(parts):
                if i == 0:
                    await update.message.reply_text(part)
                else:
                    await update.message.reply_text(part)
        else:
            await update.message.reply_text(answer)
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP ошибка: {e}")
        await processing_message.delete()
        await update.message.reply_text(
            f"❌ Ошибка при обращении к серверу: {e.response.status_code}\n"
            "Попробуйте позже."
        )
    except httpx.TimeoutException:
        logger.error("Таймаут при запросе к API")
        await processing_message.delete()
        await update.message.reply_text(
            "⏱️ Запрос занял слишком много времени. Попробуйте переформулировать вопрос."
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}", exc_info=True)
        await processing_message.delete()
        await update.message.reply_text(
            f"❌ Произошла ошибка: {str(e)}\nПопробуйте позже."
        )


def main():
    """Запуск телеграм бота"""
    logger.info("Запуск телеграм бота...")
    
    # Создаем приложение
    application = Application.builder().token(settings.telegram_bot_token).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    logger.info("Бот запущен и готов к работе!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

