import csv
import tempfile
import logging
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    CallbackQueryHandler,
    filters
)

from groq import Groq
import pandas as pd
import re
from dotenv import load_dotenv
import os
import time
import asyncio

load_dotenv()

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')  # Получить у @BotFather
GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Получить на groq.com
GROQ_MODEL = "llama3-70b-8192"

# Состояния пользователя
USER_STATE = {}

# Инициализация Groq клиента
groq_client = Groq(api_key=GROQ_API_KEY)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    USER_STATE[user_id] = {
        'dataset': None,
        'queries': None,
        'schema': None
    }

    keyboard = [
        [
            InlineKeyboardButton("Загрузить датасет", callback_data='dataset'),
            InlineKeyboardButton("Загрузить SQL-запросы", callback_data='queries'),
            InlineKeyboardButton("Загрузить схему БД", callback_data='schema')
        ],
        [InlineKeyboardButton("Обработать SQL", callback_data='process')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Добро пожаловать в SQL Optimizer Bot!\n"
        "Пожалуйста, загрузите необходимые файлы:",
        reply_markup=reply_markup
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик нажатий кнопок"""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    data = query.data

    if data in ['dataset', 'queries', 'schema']:
        USER_STATE[user_id]['awaiting'] = data
        await query.edit_message_text(f"Пожалуйста, загрузите файл ({data})")
    elif data == 'process':
        if not USER_STATE[user_id]['queries']:
            await query.edit_message_text("❌ Сначала загрузите файл с SQL-запросами!")
            return

        await query.edit_message_text("⏳ Обрабатываю SQL-запросы...")
        await process_sql_queries(user_id, context, query.message)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик загрузки документов"""
    user_id = update.effective_user.id

    if user_id not in USER_STATE or 'awaiting' not in USER_STATE[user_id]:
        await update.message.reply_text("❌ Сначала выберите тип файла!")
        return

    file_type = USER_STATE[user_id]['awaiting']
    document = update.message.document

    # Создаем временный файл
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file = await context.bot.get_file(document)
        await file.download_to_drive(temp_file.name)
        USER_STATE[user_id][file_type] = temp_file.name

    del USER_STATE[user_id]['awaiting']
    await update.message.reply_text(f"✅ Файл {file_type} успешно загружен!")

    # Обновляем меню
    keyboard = [
        [
            InlineKeyboardButton("Загрузить датасет", callback_data='dataset'),
            InlineKeyboardButton("Загрузить SQL-запросы", callback_data='queries'),
            InlineKeyboardButton("Загрузить схему БД", callback_data='schema')
        ],
        [InlineKeyboardButton("Обработать SQL", callback_data='process')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Выберите следующее действие:",
        reply_markup=reply_markup
    )


async def process_sql_queries(user_id: int, context: ContextTypes.DEFAULT_TYPE, message) -> None:
    """Обработка SQL-запросов с помощью Groq API"""
    try:
        # Чтение SQL-запросов из CSV
        queries_path = USER_STATE[user_id]['queries']
        queries = []
        with open(queries_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Пропускаем пустые строки
                    queries.append(row[0])

        # Ограничиваем количество запросов для одного запроса
        BATCH_SIZE = 10
        results = []

        for i in range(0, len(queries), BATCH_SIZE):
            batch = queries[i:i + BATCH_SIZE]
            batch_result = await process_batch(user_id, batch)
            results.extend(batch_result)

        # Сохраняем результаты в CSV
        output_file = f"results_{user_id}.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Исходный запрос', 'Исправленный запрос', 'Объяснение'])
            for res in results:
                writer.writerow([res['original'], res['fixed'], res['explanation']])

        # Отправляем результат пользователю
        await context.bot.send_document(
            chat_id=user_id,
            document=output_file,
            caption="✅ Результаты обработки SQL-запросов"
        )

        # Удаляем временные файлы
        os.unlink(output_file)
        os.unlink(queries_path)

        if USER_STATE[user_id]['dataset']:
            os.unlink(USER_STATE[user_id]['dataset'])
        if USER_STATE[user_id]['schema']:
            os.unlink(USER_STATE[user_id]['schema'])

        del USER_STATE[user_id]

    except Exception as e:
        logger.error(f"Error processing SQL queries: {e}")
        await context.bot.send_message(
            chat_id=user_id,
            text=f"❌ Ошибка при обработке запросов: {str(e)}"
        )


async def process_batch(user_id: int, queries: list) -> list:
    """Обработка пачки SQL-запросов"""
    # Формируем промпт
    prompt = f"""
Ты эксперт по SQL. Проанализируй следующие SQL-запросы, исправь ошибки и оптимизируй их.
Для каждого запроса верни ответ в строгом формате:

Исходный запрос: [оригинальный запрос]
Исправленный запрос: [исправленная версия или оригинал если не требуется исправлений]
Объяснение: [краткое объяснение ошибки или "Запрос не требует исправлений и оптимизации"]

Контекст:
"""
    # Добавляем схему БД если есть
    if USER_STATE[user_id]['schema']:
        with open(USER_STATE[user_id]['schema'], 'r', encoding='utf-8') as f:
            prompt += f"\nСхема БД:\n{f.read()}\n"

    # Добавляем SQL-запросы
    prompt += "\nSQL-запросы:\n"
    for i, query in enumerate(queries, 1):
        prompt += f"{i}. {query}\n"

    # Отправляем запрос в Groq API с задержкой и повтором при ошибке
    max_retries = 5
    delay = 5  # секунды между успешными запросами
    retry_delay = 30  # секунд при ошибке
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Ты SQL-эксперт, который исправляет и оптимизирует SQL-запросы."},
                    {"role": "user", "content": prompt}
                ],
                model=GROQ_MODEL,
                temperature=0.1,
                max_tokens=4000
            )
            await asyncio.sleep(delay)
            # Парсим ответ
            return parse_groq_response(response.choices[0].message.content, queries)
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise e


def parse_groq_response(response: str, original_queries: list) -> list:
    """Парсинг ответа от Groq API"""
    results = []
    pattern = r"Исходный запрос:\s*(.+?)\s*Исправленный запрос:\s*(.+?)\s*Объяснение:\s*(.+?)(?=\n\n|\Z)"

    matches = re.findall(pattern, response, re.DOTALL)

    for i, match in enumerate(matches):
        original, fixed, explanation = match
        # Убираем лишние пробелы и переносы строк
        original = original.strip()
        fixed = fixed.strip()
        explanation = explanation.strip()

        # Для безопасности, если не нашли все запросы
        if i < len(original_queries):
            results.append({
                'original': original_queries[i],
                'fixed': fixed,
                'explanation': explanation
            })
        else:
            results.append({
                'original': f"Query {i + 1}",
                'fixed': fixed,
                'explanation': explanation
            })

    # Если нашли меньше результатов чем запросов
    if len(results) < len(original_queries):
        for i in range(len(results), len(original_queries)):
            results.append({
                'original': original_queries[i],
                'fixed': original_queries[i],
                'explanation': "Не обработан"
            })

    return results


def main() -> None:
    """Запуск приложения"""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Запуск бота
    application.run_polling()


if __name__ == '__main__':
    main()