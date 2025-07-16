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

logging.basicConfig(level=logging.INFO)
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
    'sqlcoder-7b-2 (8bit)', 'sqlcoder-7b-2 (4bit)',
    'sqlcoder-7B-GGUF', 'CodeLlama-13B-GGUF'])
async def handle_llm_choice(message: Message):
    user_id = message.from_user.id
    llm_map = {
        'CodeLlama-7b-hf (8bit)': ('meta-llama/CodeLlama-7b-hf', '8bit'),
        'CodeLlama-7b-hf (4bit)': ('meta-llama/CodeLlama-7b-hf', '4bit'),
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
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                queries.append(row[0])
            if len(queries) >= limit:
                break
    return queries

async def process_sql_with_llm(sql_file, llm_name, message=None, quantization=None):
    queries = read_sql_queries_from_csv(sql_file)
    results = []
    prompt_template = (
        "Correct and optimize the SQL queries with explanation. "
        "Return the answer in the format: original query, corrected query (if correction is necessary), explanation of the error (if there is one, if not, then 'the query does not require corrections and optimization').\n"
        "SQL queries:\n{queries}"
    )
    prompt = prompt_template.format(queries='\n'.join(queries))
    try:
        response = generate_llm_response(prompt, llm_name, quantization=quantization)
    except Exception as e:
        if message:
            await message.answer(f"Ошибка при загрузке или запуске модели: {llm_name}: {str(e)}")
        raise
    import io
    reader = csv.reader(io.StringIO(response))
    for row in reader:
        if len(row) == 3:
            results.append(row)
        elif len(row) == 2:
            results.append([row[0], row[1], ''])
        elif len(row) == 1:
            results.append([row[0], '', ''])
    if not results:
        for q in queries:
            results.append([q, q, 'the query does not require corrections and optimization'])
    result_csv = f"result_{os.path.basename(sql_file)}"
    with open(result_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original query', 'corrected query', 'explanation'])
        writer.writerows(results)
    return result_csv

async def main():
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
