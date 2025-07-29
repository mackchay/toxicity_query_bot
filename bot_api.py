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
import sqlparse

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
    query = update.callback_query
    logger.info(f"button_handler start: user_id={query.from_user.id}, data={query.data}")
    await query.answer()
    user_id = query.from_user.id
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

async def process_sql_queries(user_id: int, context: ContextTypes.DEFAULT_TYPE, message) -> None:
    try:
        queries_path = USER_STATE[user_id]['queries']
        original_name = USER_STATE[user_id].get('queries_original_name', '')
        ext = os.path.splitext(original_name)[1].lower() if original_name else os.path.splitext(queries_path)[1].lower()

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
            batch_fixables = fixables[i:i + BATCH_SIZE]

            if ext == '.xlsx':
                batch_result = await process_batch_xlsx(batch, batch_reasons, batch_statuses, batch_fixables)
            else:
                batch_result = await process_batch(batch, batch_reasons)
            results.extend(batch_result)

        output_file = f"results_{user_id}.xlsx" if ext == '.xlsx' else f"results_{user_id}.csv"
        save_results_to_file(output_file, results, ext)
        await context.bot.send_document(chat_id=user_id, document=output_file, caption="✅ Результаты обработки SQL-запросов")
        cleanup_temp_files(user_id, output_file)
    except Exception as e:
        logger.error(f"Error processing SQL queries: {e}")
        await context.bot.send_message(chat_id=user_id, text=f"❌ Ошибка при обработке запросов: {str(e)}")

def load_queries_and_reasons(file_path, ext):
    queries, reasons, statuses, fixables = [], [], [], []
    if ext == '.xlsx':
        df = pd.read_excel(file_path)
        if 'BAD_SQL' not in df.columns or 'REASON' not in df.columns:
            return [], [], [], []
        df['STATUS'] = df.get('STATUS', None)
        df['IS_FIXABLE'] = df.get('IS_FIXABLE', True)
        df['IS_FIXABLE'] = df['IS_FIXABLE'].astype(str).str.strip().str.lower()
        df['IS_FIXABLE'] = df['IS_FIXABLE'].map({
            'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False
        }).fillna(True)
        mask_succes = df['STATUS'].astype(str).str.strip().str.lower() == 'succes'
        mask_other = ~mask_succes
        df_final = pd.concat([df[mask_succes], df[mask_other].dropna(subset=['BAD_SQL', 'REASON'])], ignore_index=True)
        queries = df_final['BAD_SQL'].astype(str).tolist()
        reasons = df_final['REASON'].astype(str).tolist()
        statuses = df_final['STATUS'].astype(str).tolist()
        fixables = df_final['IS_FIXABLE'].tolist()
    else:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    queries.append(row[0])
        reasons = [None] * len(queries)
        statuses = [None] * len(queries)
        fixables = [True] * len(queries)
    return queries, reasons, statuses, fixables

def save_results_to_file(output_file, results, ext):
    headers = ['BAD_SQL', 'REASON', 'GOOD_SQL', 'FIX']
    if ext == '.xlsx':
        df = pd.DataFrame(results)
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

MAX_PROMPT_LEN = 2000

def safe_prompt(text):
    return str(text)[:MAX_PROMPT_LEN] if text else ''

def build_prompt_xlsx(bad_sql, reason):
    return (
        "You are a strict SQL syntax corrector. You must:\n"
        "1. Analyze the BAD_SQL query.\n"
        "2. Return the corrected version in GOOD_SQL.\n"
        "3. Clearly explain what was wrong in REASON.\n"
        "4. Describe how you fixed it in FIX.\n\n"
        "IMPORTANT: Always return the answer strictly in the following format, without any extra text:\n\n"
        f"BAD_SQL:\n{safe_prompt(bad_sql)}\n\nREASON:\n{safe_prompt(reason)}"
    )

def build_prompt_csv(bad_sql):
    return (
        "You are an expert SQL assistant.\n"
        "Analyze each query and correct if needed.\n"
        "Respond strictly in format:\n"
        "BAD_SQL: <original>\n"
        "GOOD_SQL: <corrected>\n"
        "REASON: <what was wrong>\n"
        "FIX: <what was fixed>\n\n"
        f"BAD_SQL:\n{safe_prompt(bad_sql)}\n"
    )

async def llm_fix_and_optimize_async(prompt):
    await asyncio.sleep(3)
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
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
                    max_tokens=5000
                )
            )
            result = response.choices[0].message.content.strip()
            for bad in ["'''", '"""']:
                if result.startswith(bad) and result.endswith(bad):
                    result = result[len(bad):-len(bad)]
            return result.strip()
        except Exception as e:
            if 'too many requests' in str(e).lower() and attempt < max_attempts - 1:
                logger.warning('Too many requests, waiting 25 seconds before retry...')
                await asyncio.sleep(25)
            else:
                raise

async def parse_llm_response_with_retries(prompt: str, max_attempts: int = 3):
    from asyncio import sleep
    for attempt in range(max_attempts):
        response = await llm_fix_and_optimize_async(prompt)
        pattern = r"BAD_SQL:\s*(.*?)\s*GOOD_SQL:\s*(.*?)\s*REASON:\s*(.*?)\s*FIX:\s*(.*)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            bad_sql, good_sql, reason, fix = match.groups()
            parsed = sqlparse.parse(good_sql.strip())
            if parsed and any(str(stmt).strip().lower().startswith(('select', 'insert', 'update', 'delete', 'create', 'with')) for stmt in parsed):
                return {
                    'BAD_SQL': bad_sql.strip(),
                    'GOOD_SQL': good_sql.strip(),
                    'REASON': reason.strip(),
                    'FIX': fix.strip(),
                }
        await sleep(3)
    return None

async def process_batch_xlsx(queries, reasons, statuses, fixables):
    results = []
    for bad_sql, reason, status, fixable in zip(queries, reasons, statuses, fixables):
        if not fixable:
            results.append({
                'BAD_SQL': bad_sql,
                'GOOD_SQL': bad_sql,
                'REASON': reason,
                'FIX': 'Error cannot be fixed without database analysis'
            })
            continue
        if status and str(status).strip().lower() == 'succes':
            results.append({
                'BAD_SQL': bad_sql,
                'GOOD_SQL': bad_sql,
                'REASON': '',
                'FIX': 'The query does not need to be fixed.'
            })
            continue
        if not reason or str(reason).strip().lower() in ('', 'nan', 'none'):
            results.append({
                'BAD_SQL': bad_sql,
                'GOOD_SQL': bad_sql,
                'REASON': reason or '',
                'FIX': ''
            })
            continue
        prompt = build_prompt_xlsx(bad_sql, reason)
        parsed = await parse_llm_response_with_retries(prompt, max_attempts=3)
        if parsed:
            results.append(parsed)
        else:
            logger.warning(f"❌ Failed to parse response for BAD_SQL: {bad_sql[:100]}")
    return results

async def process_batch(queries, reasons=None):
    results = []
    if reasons is None:
        reasons = [''] * len(queries)
    for bad_sql, reason in zip(queries, reasons):
        prompt = build_prompt_csv(bad_sql)
        parsed = await parse_llm_response_with_retries(prompt)
        if parsed:
            results.append(parsed)
        else:
            logger.warning(f"❌ Failed to parse response for BAD_SQL: {bad_sql[:100]}")
    return results

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.run_polling()

if __name__ == '__main__':
    main()
