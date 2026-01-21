#!/bin/bash

# Скрипт для запуска с автоматической загрузкой переменных окружения

# Определяем директорию проекта
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Определяем путь к Python в виртуальном окружении
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
echo "✅ Используется Python: $PYTHON_BIN"

# Загружаем переменные окружения
echo "Загрузка переменных окружения из .env.vars..."
set -a
source "$PROJECT_DIR/.env.vars"
set +a

echo ""
echo "Запуск FastAPI сервера..."
$PYTHON_BIN -m src.api &
API_PID=$!

sleep 3

echo "Запуск телеграм бота..."
$PYTHON_BIN -m src.bot &
BOT_PID=$!

echo ""
echo "✅ API запущен с PID: $API_PID"
echo "✅ Бот запущен с PID: $BOT_PID"
echo ""
echo "Для остановки используйте: kill $API_PID $BOT_PID"
echo "Или нажмите Ctrl+C"

# Обработка сигнала прерывания
trap "echo 'Остановка сервисов...'; kill $API_PID $BOT_PID 2>/dev/null; exit" INT TERM

wait