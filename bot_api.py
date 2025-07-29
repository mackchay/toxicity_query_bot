import csv
import tempfile
import logging
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, CallbackQueryHandler, filters
)
import pandas as pd
import re
from dotenv import load_dotenv
import os
import asyncio
from groq import Groq

# === Загрузка переменных окружения и настройка логирования ===
load_dotenv()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# === Конфигурация ===
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN_LLM_API')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = "llama3-70b-8192"
USER_STATE = {}

groq_client = Groq(api_key=GROQ_API_KEY)

# === Telegram Bot Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    USER_STATE[user_id] = {'dataset': None, 'queries': None, 'schema': None}
    await update.message.reply_text(
        "Добро пожаловать в SQL Optimizer Bot!\nПожалуйста, загрузите необходимые файлы:",
        reply_markup=main_menu_keyboard()
    )

def main_menu_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("Загрузить датасет", callback_data='dataset'),
            InlineKeyboardButton("Загрузить SQL-запросы", callback_data='queries'),
            InlineKeyboardButton("Загрузить схему БД", callback_data='schema')
        ],
        [InlineKeyboardButton("Обработать SQL", callback_data='process')]
    ]
    return InlineKeyboardMarkup(keyboard)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик нажатий кнопок"""
    query = update.callback_query
    logger.info(f"button_handler start: user_id={query.from_user.id}, data={query.data}")
    await query.answer()
    user_id = query.from_user.id
    # Автоматическая инициализация состояния пользователя
    if user_id not in USER_STATE:
        USER_STATE[user_id] = {'dataset': None, 'queries': None, 'schema': None}
    data = query.data
    try:
        if data in ['dataset', 'queries', 'schema']:
            USER_STATE[user_id]['awaiting'] = data
            await query.edit_message_text(f"Пожалуйста, загрузите файл ({data})")
        elif data == 'process':
            if not USER_STATE[user_id]['queries']:
                await query.edit_message_text("❌ Сначала загрузите файл с SQL-запросами!")
                logger.info("button_handler end (no queries)")
                return
            await query.edit_message_text("⏳ Обрабатываю SQL-запросы...")
            await process_sql_queries(user_id, context, query.message)
        logger.info("button_handler end (success)")
    except Exception as e:
        logger.error(f"button_handler error: {e}")
        await query.edit_message_text(f"❌ Ошибка: {e}")
        logger.info("button_handler end (error)")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик загрузки документов"""
    user_id = update.effective_user.id
    if user_id not in USER_STATE or 'awaiting' not in USER_STATE[user_id]:
        await update.message.reply_text("❌ Сначала выберите тип файла!")
        return
    file_type = USER_STATE[user_id]['awaiting']
    document = update.message.document
    original_filename = document.file_name if document.file_name else ''
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file = await context.bot.get_file(document)
        await file.download_to_drive(temp_file.name)
        USER_STATE[user_id][file_type] = temp_file.name
        USER_STATE[user_id][file_type + '_original_name'] = original_filename
    del USER_STATE[user_id]['awaiting']
    await update.message.reply_text(f"✅ Файл {file_type} успешно загружен!", reply_markup=main_menu_keyboard())


# === Изменения в функции process_sql_queries ===
async def process_sql_queries(user_id: int, context: ContextTypes.DEFAULT_TYPE, message) -> None:
    try:
        queries_path = USER_STATE[user_id]['queries']
        original_name = USER_STATE[user_id].get('queries_original_name', '')
        ext = os.path.splitext(original_name)[1].lower() if original_name else os.path.splitext(queries_path)[1].lower()

        # Получаем ДОПОЛНИТЕЛЬНО список fixables
        queries, reasons, statuses, fixables = load_queries_and_reasons(queries_path, ext)

        if not queries:
            await context.bot.send_message(chat_id=user_id, text='❌ Не удалось загрузить SQL-запросы.')
            return

        BATCH_SIZE = 1
        results = []
        for i in range(0, len(queries), BATCH_SIZE):
            batch = queries[i:i + BATCH_SIZE]
            batch_reasons = reasons[i:i + BATCH_SIZE]
            batch_statuses = statuses[i:i + BATCH_SIZE]
            batch_fixables = fixables[i:i + BATCH_SIZE]  # Новый параметр

            if ext == '.xlsx':
                # Передаем fixables в обработчик
                batch_result = await process_batch_xlsx(
                    batch,
                    batch_reasons,
                    batch_statuses,
                    batch_fixables
                )
            else:
                batch_result = await process_batch(batch, batch_reasons)
            results.extend(batch_result)
        # Выбираем формат сохранения результата
        if ext == '.xlsx':
            output_file = f"results_{user_id}.xlsx"
        else:
            output_file = f"results_{user_id}.csv"
        save_results_to_file(output_file, results, ext)
        await context.bot.send_document(
            chat_id=user_id,
            document=output_file,
            caption="✅ Результаты обработки SQL-запросов"
        )
        cleanup_temp_files(user_id, output_file)
    except Exception as e:
        logger.error(f"Error processing SQL queries: {e}")
        await context.bot.send_message(chat_id=user_id, text=f"❌ Ошибка при обработке запросов: {str(e)}")


# === Изменения в функции load_queries_and_reasons ===
def load_queries_and_reasons(file_path, ext):
    """Загрузка SQL-запросов, причин, статусов и флагов fixable"""
    queries, reasons, statuses, fixables = [], [], [], []
    if ext == '.xlsx':
        df = pd.read_excel(file_path)

        # Проверка обязательных столбцов
        if 'BAD_SQL' not in df.columns or 'REASON' not in df.columns:
            return [], [], [], []

        # Инициализация optional columns
        df['STATUS'] = df.get('STATUS', None)
        df['IS_FIXABLE'] = df.get('IS_FIXABLE', True)  # По умолчанию True

        # Предобработка флагов fixable
        df['IS_FIXABLE'] = df['IS_FIXABLE'].astype(str).str.strip().str.lower()
        df['IS_FIXABLE'] = df['IS_FIXABLE'].map({
            'true': True,
            'false': False,
            '1': True,
            '0': False,
            'yes': True,
            'no': False
        }).fillna(True)  # Неизвестные значения -> True

        # Фильтрация данных
        mask_succes = df['STATUS'].astype(str).str.strip().str.lower() == 'succes'
        mask_other = ~mask_succes

        df_succes = df[mask_succes]
        df_other = df[mask_other]
        df_other = df_other.dropna(subset=['BAD_SQL', 'REASON'])
        df_other = df_other[df_other['BAD_SQL'].astype(str).str.strip().astype(bool)]

        # Объединение данных
        df_final = pd.concat([df_succes, df_other], ignore_index=True)

        # Формирование списков
        queries = df_final['BAD_SQL'].astype(str).tolist()
        reasons = df_final['REASON'].astype(str).tolist()
        statuses = df_final['STATUS'].astype(str).tolist()
        fixables = df_final['IS_FIXABLE'].tolist()

    else:  # CSV обработка
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    queries.append(row[0])
        reasons = [None] * len(queries)
        statuses = [None] * len(queries)
        fixables = [True] * len(queries)  # Для CSV всегда True

    return queries, reasons, statuses, fixables

def parse_llm_response(response: str):
    # Регулярка для поиска блоков
    pattern = r"BAD_SQL:\s*(.*?)\s*GOOD_SQL:\s*(.*?)\s*REASON:\s*(.*?)\s*FIX:\s*(.*)"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        bad_sql, good_sql, reason, fix = match.groups()
        return {
            'BAD_SQL': bad_sql.strip(),
            'GOOD_SQL': good_sql.strip(),
            'REASON': reason.strip(),
            'FIX': fix.strip(),
        }
    # Если не удалось распарсить, вернуть всё в REASON
    return {
        'BAD_SQL': '',
        'GOOD_SQL': '',
        'REASON': response.strip(),
        'FIX': ''
    }

def save_results_to_file(output_file, results, ext):
    headers = ['BAD_SQL', 'REASON', 'GOOD_SQL', 'FIX']
    if ext == '.xlsx':
        df = pd.DataFrame(results)
        # Гарантируем порядок и наличие всех колонок
        for h in headers:
            if h not in df.columns:
                df[h] = ''
        df = df[headers]
        df.to_excel(output_file, index=False)
    else:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for res in results:
                row = [res.get(h, '') for h in headers]
                writer.writerow(row)

def cleanup_temp_files(user_id, output_file):
    os.unlink(output_file)
    for key in ['queries', 'dataset', 'schema']:
        path = USER_STATE[user_id].get(key)
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass
    del USER_STATE[user_id]

# === LLM Batch Processing ===
MAX_PROMPT_LEN = 2000  # Максимальная длина для каждой части промпта

def safe_prompt(text):
    if text is None:
        return ''
    return str(text)[:MAX_PROMPT_LEN]

def build_prompt_xlsx(bad_sql, reason):
    bad_sql = safe_prompt(bad_sql)
    reason = safe_prompt(reason)
    return (
        "You are an expert SQL assistant.\n"
        "Your task is to analyze each SQL query, detect if it is invalid or contains an error, and correct it.\n"
        "Then, return the result in the following structured format:\n\n"
        "BAD_SQL: <original query>\n"
        "GOOD_SQL: <corrected query>\n"
        "REASON: <why the original query was incorrect>\n"
        "FIX: <what exactly was changed to fix the error>\n\n"
        "You must follow this output format strictly.\n\n"
        "Now, process the following SQL query:\n\n"
        f"BAD_SQL:\n{bad_sql}\n"
        f"REASON:\n{reason}\n"
    )

def build_prompt_csv(bad_sql):
    bad_sql = safe_prompt(bad_sql)
    return (
        "You are an expert SQL assistant.\n"
        "Your task is to analyze each SQL query, detect if it is invalid or contains an error, and correct it.\n"
        "Then, return the result in the following structured format:\n\n"
        "BAD_SQL: <original query>\n"
        "GOOD_SQL: <corrected query>\n"
        "REASON: <why the original query was incorrect>\n"
        "FIX: <what exactly was changed to fix the error>\n\n"
        "You must follow this output format strictly.\n\n"
        "Now, process the following SQL query:\n\n"
        f"BAD_SQL:\n{bad_sql}\n"
    )

async def llm_fix_and_optimize_async(prompt):
    import asyncio
    # Первая попытка — задержка 3 секунды
    await asyncio.sleep(3)
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            # Синхронный groq_client не поддерживает async, поэтому используем run_in_executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert SQL assistant. Always answer strictly in the format: BAD_SQL: ... GOOD_SQL: ... REASON: ... FIX: ..."},
                        {"role": "user", "content": prompt}
                    ],
                    model=GROQ_MODEL,
                    temperature=0.1,
                    max_tokens=1000
                )
            )
            result = response.choices[0].message.content.strip()
            # Удаляем ''' и """ и лишние пробелы/переводы строк вокруг
            for bad in ["'''", '"""']:
                if result.startswith(bad) and result.endswith(bad):
                    result = result[len(bad):-len(bad)]
            result = result.strip()
            return result
        except Exception as e:
            err_str = str(e).lower()
            if 'too many requests' in err_str or '429' in err_str:
                if attempt < max_attempts - 1:
                    logger.warning('Too many requests, waiting 25 seconds before retry...')
                    await asyncio.sleep(25)
                    continue
            raise


# === Изменения в функции process_batch_xlsx ===
async def process_batch_xlsx(
        queries: list,
        reasons: list,
        statuses: list,
        fixables: list  # Новый параметр
) -> list:
    results = []
    for bad_sql, reason, status, fixable in zip(queries, reasons, statuses, fixables):
        # 1. Проверка IS_FIXABLE == False
        if not fixable:
            results.append({
                'BAD_SQL': bad_sql,
                'GOOD_SQL': bad_sql,  # Копируем оригинальный запрос
                'REASON': reason,
                'FIX': 'Error cannot be fixed without database analysis'
            })
            continue

        # 2. Обработка статуса SUCCES (существующая логика)
        if status is not None and str(status).strip().lower() == 'succes':
            results.append({
                'BAD_SQL': bad_sql,
                'GOOD_SQL': bad_sql,
                'REASON': '',
                'FIX': 'The query does not need to be fixed.'
            })
            continue

        # 3. Обработка через LLM
        if reason is None or str(reason).strip().lower() in ('', 'nan', 'none'):
            results.append({
                'BAD_SQL': bad_sql,
                'GOOD_SQL': bad_sql,
                'REASON': reason or '',
                'FIX': ''
            })
        else:
            prompt = build_prompt_xlsx(bad_sql, reason)
            max_attempts = 3
            for attempt in range(max_attempts):
                llm_response = await llm_fix_and_optimize_async(prompt)
                parsed = parse_llm_response(llm_response)
                if parsed['BAD_SQL']:
                    results.append(parsed)
                    break
                elif attempt == max_attempts - 1:
                    results.append({
                        'BAD_SQL': bad_sql,
                        'GOOD_SQL': '',
                        'REASON': llm_response,
                        'FIX': ''
                    })
                else:
                    await asyncio.sleep(3)
    return results

async def process_batch(queries: list, reasons: list = None) -> list:
    """Обработка пачки SQL-запросов из csv (только BAD_SQL, reason может быть None)"""
    results = []
    if reasons is None:
        reasons = [''] * len(queries)
    for bad_sql, reason in zip(queries, reasons):
        prompt = build_prompt_csv(bad_sql)
        llm_response = await llm_fix_and_optimize_async(prompt)
        parsed = parse_llm_response(llm_response)
        results.append(parsed)
    return results

# === Telegram Bot Main ===
def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.run_polling()

if __name__ == '__main__':
    main()