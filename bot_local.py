import logging
import csv
import os
from aiogram import Bot, Dispatcher
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, FSInputFile
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command
from aiogram import Router
from aiogram.types import Message
import asyncio
from dotenv import load_dotenv

from model_manager import generate_llm_response

load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == 'None':
    raise RuntimeError('TELEGRAM_TOKEN environment variable is not set or is invalid!')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_responses.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()

def get_file_kb():
    kb = [
        [KeyboardButton(text='Загрузить датасет')],
        [KeyboardButton(text='Загрузить SQL-запросы')],
        [KeyboardButton(text='Загрузить схему БД')]
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_llm_kb():
    kb = [
        [KeyboardButton(text='CodeLlama-7b-hf (8bit)'), KeyboardButton(text='CodeLlama-7b-hf (4bit)')],
        [KeyboardButton(text='CodeLlama-7b-Instruct-hf'), KeyboardButton(text='CodeLlama-7b-Instruct-GGUF')],
        [KeyboardButton(text='sqlcoder-7b-2 (8bit)'), KeyboardButton(text='sqlcoder-7b-2 (4bit)')],
        [KeyboardButton(text='sqlcoder-7B-GGUF')],
        [KeyboardButton(text='CodeLlama-13B-GGUF')]
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

user_files = {}

@router.message(Command('start'))
async def send_welcome(message: Message):
    await message.answer("Привет! Выберите, какой файл загрузить:", reply_markup=get_file_kb())

@router.message(lambda message: message.text in ['Загрузить датасет', 'Загрузить SQL-запросы', 'Загрузить схему БД'])
async def ask_file(message: Message):
    user_files[message.from_user.id] = {'file_type': message.text}
    await message.answer("Пожалуйста, отправьте файл.")

@router.message(lambda message: message.document is not None)
async def handle_file(message: Message):
    user_id = message.from_user.id
    file_type = user_files.get(user_id, {}).get('file_type')
    if not file_type:
        await message.answer("Сначала выберите тип файла.", reply_markup=get_file_kb())
        return
    file_info = await bot.get_file(message.document.file_id)
    file_path = f"{user_id}_{message.document.file_name}"
    await bot.download_file(file_info.file_path, file_path)
    user_files[user_id][file_type] = file_path
    await message.answer(f"Файл '{file_type}' успешно загружен.")
    if file_type == 'Загрузить SQL-запросы':
        await message.answer("Выберите LLM для обработки:", reply_markup=get_llm_kb())
    else:
        await message.answer("Выберите следующий файл или загрузите SQL-запросы.", reply_markup=get_file_kb())

@router.message(lambda message: message.text in [
    'CodeLlama-7b-hf (8bit)', 'CodeLlama-7b-hf (4bit)',
    'CodeLlama-7b-Instruct-hf', 'CodeLlama-7b-Instruct-GGUF',
    'sqlcoder-7b-2 (8bit)', 'sqlcoder-7b-2 (4bit)',
    'sqlcoder-7B-GGUF', 'CodeLlama-13B-GGUF'])
async def handle_llm_choice(message: Message):
    user_id = message.from_user.id
    llm_map = {
        'CodeLlama-7b-hf (8bit)': ('codellama/CodeLlama-7b-hf', '8bit'),
        'CodeLlama-7b-hf (4bit)': ('codellama/CodeLlama-7b-hf', '4bit'),
        'CodeLlama-7b-Instruct-hf': ('codellama/CodeLlama-7b-Instruct-hf', None),
        'CodeLlama-7b-Instruct-GGUF': ('TheBloke/CodeLlama-7B-Instruct-GGUF', None),
        'sqlcoder-7b-2 (8bit)': ('defog/sqlcoder-7b-2', '8bit'),
        'sqlcoder-7b-2 (4bit)': ('defog/sqlcoder-7b-2', '4bit'),
        'sqlcoder-7B-GGUF': ('TheBloke/sqlcoder-7B-GGUF', None),
        'CodeLlama-13B-GGUF': ('TheBloke/CodeLlama-13B-GGUF', None)
    }
    llm, quant = llm_map[message.text]
    sql_file = user_files.get(user_id, {}).get('Загрузить SQL-запросы')
    if not sql_file:
        await message.answer("Сначала загрузите файл с SQL-запросами.", reply_markup=get_file_kb())
        return
    await message.answer(f"Model {llm} loading started. Please wait...")
    result_csv = await process_sql_with_llm(sql_file, llm, message, quantization=quant)
    await message.answer(f"Model {llm} loaded and processing finished.")
    with open(result_csv, 'rb') as f:
        await message.answer_document(FSInputFile(result_csv), caption="Результаты обработки SQL-запросов")
    await message.answer("Готово! Можете загрузить новые файлы.", reply_markup=get_file_kb())

def read_sql_queries_from_csv(file_path, limit=10):
    queries = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if row:
                queries.append(row[0])
            if len(queries) >= limit:
                break
    return queries

def get_sqlcoder_prompt(original_query: str, user_question: str, table_metadata_string_DDL_statements: str) -> str:
    return (
        "### Task: Write a SQL query to solve the following problem\n"
        f"### Database Schema:\n{table_metadata_string_DDL_statements}\n\n"
        f"### Question: {user_question}\n"
        f"### Original query: {original_query}\n"
        "### Response: Let me analyze and improve this SQL query.\n"
    )

def get_codellama_base_prompt(original_query: str, user_question: str, table_metadata_string_DDL_statements: str) -> str:
    return (
        "You are an expert SQL developer. Given the following database schema and query, help improve and optimize it.\n\n"
        f"Database Schema:\n{table_metadata_string_DDL_statements}\n\n"
        f"Original Query: {original_query}\n\n"
        f"Task: {user_question}\n\n"
        "Improved query:"
    )

def get_codellama_instruct_prompt(original_query: str, user_question: str, table_metadata_string_DDL_statements: str) -> str:
    return (
        "### Task\n"
        f"Generate a SQL query to answer [QUESTION]{user_question}[/QUESTION]\n\n"
        "### Database Schema\n"
        "The query will run on a database with the following schema:\n"
        f"{table_metadata_string_DDL_statements}\n\n"
        "### Context\n"
        f"Original query to improve: {original_query}\n\n"
        "### Answer\n"
        "Here is the improved SQL query with explanation:\n"
    )

async def process_sql_with_llm(sql_file, llm_name, message=None, quantization=None):
    queries = read_sql_queries_from_csv(sql_file)
    results = []

    # Читаем схему БД из файла, если он есть
    table_metadata_string_ddl_statements = ''
    if message:
        user_id = message.from_user.id
        schema_file = user_files.get(user_id, {}).get('Загрузить схему БД')
        if schema_file:
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    table_metadata_string_ddl_statements = f.read().strip()
                logger.info(f"Loaded database schema from {schema_file}")
            except Exception as e:
                logger.error(f"Error reading schema file: {e}")

    user_question = 'Correct and optimize the SQL query, explain the correction.'

    for idx, original_query in enumerate(queries, 1):
        logger.info(f"\nProcessing query {idx}/{len(queries)}:")
        logger.info(f"Original query: {original_query}")

        # Выбор и формирование промпта
        if 'sqlcoder' in llm_name.lower():
            prompt = get_sqlcoder_prompt(original_query, user_question, table_metadata_string_ddl_statements)
        elif 'instruct' in llm_name.lower():
            prompt = get_codellama_instruct_prompt(original_query, user_question, table_metadata_string_ddl_statements)
        else:
            prompt = get_codellama_base_prompt(original_query, user_question, table_metadata_string_ddl_statements)

        logger.info(f"Generated prompt for {llm_name}:")
        logger.info(f"{prompt}\n")

        try:
            # Получаем ответ от модели
            response = generate_llm_response(prompt, llm_name, quantization=quantization)
            logger.info(f"Full LLM response:\n{response}\n")

            # Сохраняем оригинальный запрос и полный ответ модели
            results.append([original_query, response])
            logger.info(f"Saved query and response for query {idx}")

        except Exception as e:
            error_msg = f"Error processing query with {llm_name}: {str(e)}"
            logger.error(error_msg)
            if message:
                await message.answer(error_msg)
            results.append([original_query, f'Error: {str(e)}'])

    # Сохраняем результаты
    result_csv = f"result_{os.path.basename(sql_file)}"
    with open(result_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original sql', 'response'])
        writer.writerows(results)

    logger.info(f"Results saved to {result_csv}")
    return result_csv

async def main():
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
