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

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN_LLM_LOCAL')
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

# === ДОБАВЛЕНО: Flow для дообучения LLM по датасету ===
from train_llm import train_bnb_lora

# Список HuggingFace моделей для дообучения (без GGUF)
HF_FINETUNE_MODELS = [
    'CodeLlama-7b-Instruct-hf',
    'Mistral-7B-Instruct-v0.2',
    # Добавьте другие HF-модели, если нужно
]

def get_file_kb():
    buttons = ['Загрузить датасет', 'Загрузить SQL-запросы', 'Загрузить схему БД']
    kb = [[KeyboardButton(text=btn)] for btn in buttons]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_llm_kb():
    models = [
        ['CodeLlama-7b-Instruct-hf'],
        ['Mistral-7B-Instruct-v0.2']
        # Добавьте другие HF-модели, если нужно
    ]
    kb = [[KeyboardButton(text=btn) for btn in row] for row in models]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_finetune_model_kb():
    kb = [[KeyboardButton(text=btn)] for btn in HF_FINETUNE_MODELS]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

def get_quant_kb():
    kb = [[KeyboardButton(text=btn)] for btn in ['4bit', '8bit', 'fp16']]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

# Сохраняем состояние для дообучения (user_id -> {'dataset': ..., 'model': ...})
finetune_state = {}

# Сохраняем загруженные обычные файлы пользователя (user_id -> {'file_type': ..., ...})
user_files = {}

@router.message(Command('start'))
async def send_welcome(message: Message):
    await message.answer("Привет! Выберите, какой файл загрузить:", reply_markup=get_file_kb())

@router.message(lambda message: message.text in ['Загрузить датасет', 'Загрузить SQL-запросы', 'Загрузить схему БД'])
async def ask_file(message: Message):
    if not message.from_user or not message.from_user.id:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return
    user_files[message.from_user.id] = {'file_type': message.text}
    await message.answer("Пожалуйста, отправьте файл.")

@router.message(lambda message: message.document is not None)
async def handle_file(message: Message):
    if not message.from_user or not message.from_user.id:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return
    user_id = message.from_user.id
    file_type = user_files.get(user_id, {}).get('file_type')
    if not file_type:
        await message.answer("Сначала выберите тип файла.", reply_markup=get_file_kb())
        return
    if not message.document or not message.document.file_id:
        await message.answer("Ошибка: не удалось получить файл.")
        return
    file_info = await bot.get_file(message.document.file_id)
    file_path = f"{user_id}_{message.document.file_name}"
    if not file_info.file_path:
        await message.answer("Ошибка: не удалось получить путь к файлу.")
        return
    await bot.download_file(file_info.file_path, file_path)
    user_files[user_id][file_type] = file_path
    if file_type == 'Загрузить датасет':
        # Сохраняем путь к датасету для дообучения
        finetune_state[user_id] = {'dataset': file_path}
        await message.answer("Файл датасета успешно загружен!\nВыберите модель для дообучения:", reply_markup=get_finetune_model_kb())
    else:
        await message.answer(f"Файл '{file_type}' успешно загружен.")
        if file_type == 'Загрузить SQL-запросы':
            await message.answer("Выберите LLM для обработки:", reply_markup=get_llm_kb())
        else:
            await message.answer("Выберите следующий файл или загрузите SQL-запросы.", reply_markup=get_file_kb())

@router.message(lambda message: message.text in HF_FINETUNE_MODELS and finetune_state.get(message.from_user.id) and finetune_state[message.from_user.id].get('dataset'))
async def handle_finetune_model_choice(message: Message):
    user_id = message.from_user.id
    finetune_state[user_id]['model'] = message.text
    await message.answer("Выберите тип квантования:", reply_markup=get_quant_kb())

@router.message(lambda message: message.text in ['4bit', '8bit', 'fp16'] and finetune_state.get(message.from_user.id) and finetune_state[message.from_user.id].get('model'))
async def handle_finetune_quant_choice(message: Message):
    import asyncio
    user_id = message.from_user.id
    quant = message.text
    dataset_path = finetune_state[user_id]['dataset']
    model_name = finetune_state[user_id]['model']
    # Маппинг на huggingface repo
    hf_map = {
        'CodeLlama-7b-Instruct-hf': 'codellama/CodeLlama-7b-Instruct-hf',
        'Mistral-7B-Instruct-v0.2': 'mistralai/Mistral-7B-Instruct-v0.2',
        # Добавьте другие соответствия, если нужно
    }
    base_model = hf_map.get(model_name, model_name)
    await message.answer(f"Запускается дообучение модели {model_name} с квантованием {quant}. Это может занять много времени...")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: train_bnb_lora(
                dataset_path,
                base_model=base_model,
                output_dir=f"finetuned_{user_id}_{model_name}_{quant}",
                quantization_type=quant
            )
        )
        await message.answer(f"Дообучение завершено! LoRA-адаптер сохранён в папке finetuned_{user_id}_{model_name}_{quant}")
    except Exception as e:
        await message.answer(f"Ошибка при дообучении: {str(e)}")
    # Очистка состояния
    finetune_state.pop(user_id, None)

@router.message(lambda message: message.text in [
    'CodeLlama-7b-Instruct-hf', 'Mistral-7B-Instruct-v0.2'])
async def handle_llm_choice(message: Message):
    if not message.from_user or not message.from_user.id:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return
    user_id = message.from_user.id
    llm_map = {
        'CodeLlama-7b-Instruct-hf': ('codellama/CodeLlama-7b-Instruct-hf', None),
        'Mistral-7B-Instruct-v0.2': ('mistralai/Mistral-7B-Instruct-v0.2', None),
    }
    if not message.text:
        await message.answer("Неизвестная модель")
        return
    model_info = llm_map.get(message.text)
    if not model_info:
        await message.answer("Неизвестная модель")
        return
    model_name, quantization = model_info
    sql_file = user_files.get(user_id, {}).get('Загрузить SQL-запросы')
    if not sql_file:
        await message.answer("Сначала загрузите файл с SQL-запросами.", reply_markup=get_file_kb())
        return
    await message.answer(f"Загрузка модели {model_name} начата. Пожалуйста, подождите...")
    try:
        result_csv = await process_sql_with_llm(sql_file, model_name, message, quantization=quantization)
        await message.answer(f"Модель {model_name} загружена и обработка завершена.")
        with open(result_csv, 'rb') as f:
            await message.answer_document(
                FSInputFile(result_csv),
                caption="Результаты обработки SQL-запросов"
            )
    except Exception as e:
        await message.answer(f"Произошла ошибка при обработке: {str(e)}")
        return
    await message.answer("Готово! Можете загрузить новые файлы.", reply_markup=get_file_kb())

def read_sql_queries_from_csv(file_path):
    queries = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if row:
                queries.append(row[0])
    return queries

def get_sqlcoder_prompt(original_query: str, user_question: str, table_metadata_string_DDL_statements: str) -> str:
    return (
        "### System: You are a SQL expert. Return only the fixed and optimized SQL query without any explanations or comments.\n"
        f"### Schema:\n{table_metadata_string_DDL_statements}\n"
        f"### Query to optimize:\n{original_query}\n"
        "### Response (SQL only):\n"
    )

def get_codellama_base_prompt(original_query: str, user_question: str, table_metadata_string_DDL_statements: str) -> str:
    return (
        f"Schema:\n{table_metadata_string_DDL_statements}\n\n"
        f"Original SQL:\n{original_query}\n\n"
        "Return optimized SQL query only, no comments or explanations:\n"
    )

def get_codellama_instruct_prompt(original_query: str, user_question: str, table_metadata_string_DDL_statements: str) -> str:
    return (
        "### Instruction\n"
        "Analyze and optimize the SQL query. Return only the optimized SQL without any explanations.\n\n"
        f"### Schema\n{table_metadata_string_DDL_statements}\n\n"
        f"### Input Query\n{original_query}\n\n"
        "### Output SQL\n"
    )

def get_mistral_prompt(original_query: str, table_metadata_string_DDL_statements: str) -> str:
    return (
        "<s>[INST] You are an expert SQL assistant. Analyze and optimize this SQL query. "
        "Return only the corrected SQL without any additional explanations.\n\n"
        f"Database Schema:\n{table_metadata_string_DDL_statements}\n\n"
        f"Original SQL Query:\n{original_query}\n\n"
        "Optimized SQL Query: [/INST]"
    )

async def process_sql_with_llm(sql_file, llm_name, message=None, quantization=None):
    try:
        queries = read_sql_queries_from_csv(sql_file)
        if not queries:
            raise ValueError("SQL файл пуст или имеет неверный формат")
    except Exception as e:
        logger.error(f"Ошибка при чтении SQL файла {sql_file}: {str(e)}")
        raise ValueError(f"Не удалось прочитать SQL запросы: {str(e)}")
    results = []
    # Читаем схему БД из файла
    table_metadata_string_ddl_statements = ''
    if message:
        user_id = message.from_user.id
        schema_file = user_files.get(user_id, {}).get('Загрузить схему БД')
        if schema_file:
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    table_metadata_string_ddl_statements = f.read().strip()
                if not table_metadata_string_ddl_statements:
                    logger.warning("Файл схемы БД пуст")
                    await message.answer("Предупреждение: файл схемы БД пуст")
                else:
                    logger.info(f"Загружена схема БД из {schema_file}")
            except Exception as e:
                logger.error(f"Ошибка при чтении файла схемы БД: {e}")
                await message.answer(f"Предупреждение: не удалось прочитать схему БД: {str(e)}")
    total_queries = len(queries)
    for idx, original_query in enumerate(queries, 1):
        if not original_query.strip():
            logger.warning(f"Пропущен пустой запрос #{idx}")
            continue
        logger.info(f"\nОбработка запроса {idx}/{total_queries}:")
        logger.info(f"Исходный запрос: {original_query}")
        if message:
            await message.answer(f"Обработка запроса {idx} из {total_queries}...")
        try:
            if 'mistral' in llm_name.lower():
                prompt = get_mistral_prompt(original_query, table_metadata_string_ddl_statements)
            elif 'instruct' in llm_name.lower():
                prompt = get_codellama_instruct_prompt(original_query, '', table_metadata_string_ddl_statements)
            else:
                prompt = get_codellama_base_prompt(original_query, '', table_metadata_string_ddl_statements)
            logger.info(f"Сгенерирован промпт для {llm_name}:")
            logger.info(f"{prompt}\n")
            if not quantization:
                quantization = '8bit'
            response = generate_llm_response(prompt, llm_name, quantization=quantization)
            logger.info(f"Ответ модели:\n{response}\n")
            if not response.strip():
                logger.warning(f"Получен пустой ответ для запроса #{idx}")
                response = "ERROR: Модель вернула пустой ответ"
            results.append([original_query, response])
            logger.info(f"Сохранен результат для запроса #{idx}")
        except Exception as e:
            error_msg = f"Ошибка при обработке запроса #{idx}: {str(e)}"
            logger.error(error_msg)
            if message:
                await message.answer(error_msg)
            results.append([original_query, f"ERROR: {str(e)}"])
    if not results:
        raise ValueError("Не удалось обработать ни один запрос")
    try:
        result_csv = f"result_{os.path.basename(sql_file)}"
        with open(result_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['original sql', 'response'])
            writer.writerows(results)
        logger.info(f"Результаты сохранены в {result_csv}")
        return result_csv
    except Exception as e:
        error_msg = f"Ошибка при сохранении результатов: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

async def main():
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
