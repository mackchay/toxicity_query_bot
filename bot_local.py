import logging
import os
from aiogram import Bot, Dispatcher
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, FSInputFile
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command
from aiogram import Router
from aiogram.types import Message
import asyncio
from dotenv import load_dotenv

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

# === ИМПОРТЫ ДЛЯ ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ ===
from train_llm import train_bnb_lora
from test_llm import test_model

# Список HuggingFace моделей для дообучения (без GGUF)
HF_FINETUNE_MODELS = [
    'CodeLlama-7b-Instruct-hf',
    'Mistral-7B-Instruct-v0.2',
    # Добавьте другие HF-модели, если нужно
]

# Список доступных базовых моделей для тестирования
BASE_MODELS = [
    'CodeLlama-7b-Instruct-hf',
    'Mistral-7B-Instruct-v0.2'
]


def get_main_kb():
    """Главное меню с выбором действия"""
    buttons = ['Обучить LLM', 'Протестировать LLM', 'Исправить SQL-запросы']  # ← здесь заменено
    kb = [[KeyboardButton(text=btn)] for btn in buttons]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

@router.message(lambda message: message.text == 'Исправить SQL-запрос')
async def fix_sql_request(message: Message):
    """Обработка запроса на исправление SQL"""
    if not message.from_user:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return

    set_user_state(message.from_user.id, "fix_sql")
    await message.answer(
        "Пожалуйста, отправьте SQL-запрос, который вы хотите исправить.\n"
        "Я постараюсь сделать его корректным и более эффективным."
    )

@router.message(lambda message: get_user_state(message.from_user.id) == "fix_sql")
async def handle_sql_fix_request(message: Message):
    """Получение и исправление SQL-запроса"""
    if not message.from_user or not message.text:
        await message.answer("Ошибка: запрос пустой.")
        return

    raw_sql = message.text

    await message.answer("Обрабатываю запрос...")

    try:
        # Пример простого вызова модели или функции
        # Здесь ты можешь вызвать свой LLM, локально или через API
        # Ниже — заглушка
        fixed_sql = "SELECT id, name FROM users WHERE active = TRUE;"  # замените на вызов LLM

        await message.answer(
            f"🔧 Исправленный SQL-запрос:\n```\n{fixed_sql}\n```",
            parse_mode="Markdown"
        )

    except Exception as e:
        await message.answer(f"Ошибка при обработке запроса: {str(e)}")

    set_user_state(message.from_user.id, STATE_MAIN)
    await message.answer("Возврат в главное меню.", reply_markup=get_main_kb())


def get_finetune_model_kb():
    """Клавиатура для выбора модели для дообучения"""
    kb = [[KeyboardButton(text=btn)] for btn in HF_FINETUNE_MODELS]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_test_model_kb():
    """Клавиатура для выбора модели для тестирования"""
    buttons = ['Базовая модель', 'Дообученная модель']
    kb = [[KeyboardButton(text=btn)] for btn in buttons]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_base_model_kb():
    """Клавиатура для выбора базовой модели"""
    kb = [[KeyboardButton(text=btn)] for btn in BASE_MODELS]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_quant_kb():
    """Клавиатура для выбора типа квантования"""
    kb = [[KeyboardButton(text=btn)] for btn in ['4bit', '8bit', 'fp16']]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


# Состояния пользователей
user_states = {}

# Возможные состояния
STATE_MAIN = "main"
STATE_TRAIN_UPLOAD_DATASET = "train_upload_dataset"
STATE_TRAIN_MODEL_CHOICE = "train_model_choice"
STATE_TRAIN_QUANT_CHOICE = "train_quant_choice"
STATE_TEST_MODEL_TYPE = "test_model_type"
STATE_TEST_BASE_MODEL_CHOICE = "test_base_model_choice"
STATE_TEST_FINETUNED_MODEL_CHOICE = "test_finetuned_model_choice"
STATE_TEST_UPLOAD_DATASET = "test_upload_dataset"
STATE_SCHEMA_UPLOAD = "schema_upload"


def set_user_state(user_id, state, **kwargs):
    """Устанавливает состояние пользователя"""
    if user_id not in user_states:
        user_states[user_id] = {}
    user_states[user_id]['state'] = state
    user_states[user_id].update(kwargs)


def get_user_state(user_id):
    """Получает состояние пользователя"""
    return user_states.get(user_id, {}).get('state', STATE_MAIN)


def get_user_data(user_id, key, default=None):
    """Получает данные пользователя"""
    return user_states.get(user_id, {}).get(key, default)


def get_finetuned_models():
    """Получает список дообученных моделей из папок"""
    finetuned_models = []
    current_dir = os.getcwd()

    # Ищем папки, начинающиеся с "finetuned_"
    for item in os.listdir(current_dir):
        if os.path.isdir(item) and item.startswith("finetuned_"):
            finetuned_models.append(item)

    # Также ищем стандартные папки из train_llm.py
    for item in os.listdir(current_dir):
        if os.path.isdir(item) and "bnb-lora-finetuned" in item:
            finetuned_models.append(item)

    return finetuned_models


def get_finetuned_model_kb():
    """Клавиатура для выбора дообученной модели"""
    models = get_finetuned_models()
    if not models:
        kb = [[KeyboardButton(text="Нет доступных дообученных моделей")]]
    else:
        kb = [[KeyboardButton(text=model)] for model in models]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


@router.message(Command('start'))
async def send_welcome(message: Message):
    """Команда /start - показываем главное меню"""
    if not message.from_user:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return

    set_user_state(message.from_user.id, STATE_MAIN)
    await message.answer(
        "Привет! Выберите действие:",
        reply_markup=get_main_kb()
    )


@router.message(lambda message: message.text == 'Обучить LLM')
async def start_training(message: Message):
    """Начало процесса обучения LLM"""
    if not message.from_user:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return

    set_user_state(message.from_user.id, STATE_TRAIN_UPLOAD_DATASET)
    await message.answer(
        "Для обучения LLM загрузите датасет в формате Excel (.xlsx).\n\n"
        "Файл должен содержать следующие столбцы:\n"
        "- BAD_SQL: неправильный SQL-запрос\n"
        "- GOOD_SQL: исправленный SQL-запрос\n"
        "- REASON: причина ошибки\n"
        "- FIX: описание исправления\n\n"
        "Пожалуйста, отправьте файл датасета:"
    )


@router.message(lambda message: message.text == 'Протестировать LLM')
async def start_testing(message: Message):
    """Начало процесса тестирования LLM"""
    if not message.from_user:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return

    set_user_state(message.from_user.id, STATE_TEST_MODEL_TYPE)
    await message.answer(
        "Выберите тип модели для тестирования:",
        reply_markup=get_test_model_kb()
    )


@router.message(lambda message: message.text == 'Загрузить схему БД')
async def start_schema_upload(message: Message):
    """Загрузка схемы БД"""
    if not message.from_user:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return

    set_user_state(message.from_user.id, STATE_SCHEMA_UPLOAD)
    await message.answer(
        "Загрузите файл со схемой базы данных (DDL statements).\n"
        "Файл должен содержать SQL-команды CREATE TABLE и другие DDL-команды.\n\n"
        "Пожалуйста, отправьте файл:"
    )


@router.message(
    lambda message: message.text == 'Базовая модель' and get_user_state(message.from_user.id) == STATE_TEST_MODEL_TYPE)
async def choose_base_model_for_test(message: Message):
    """Выбор базовой модели для тестирования"""
    if not message.from_user:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return

    set_user_state(message.from_user.id, STATE_TEST_BASE_MODEL_CHOICE)
    await message.answer(
        "Выберите базовую модель для тестирования:",
        reply_markup=get_base_model_kb()
    )


@router.message(lambda message: message.text == 'Дообученная модель' and get_user_state(
    message.from_user.id) == STATE_TEST_MODEL_TYPE)
async def choose_finetuned_model_for_test(message: Message):
    """Выбор дообученной модели для тестирования"""
    if not message.from_user:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return

    finetuned_models = get_finetuned_models()
    if not finetuned_models:
        await message.answer(
            "Дообученные модели не найдены. Сначала обучите модель.",
            reply_markup=get_main_kb()
        )
        set_user_state(message.from_user.id, STATE_MAIN)
        return

    set_user_state(message.from_user.id, STATE_TEST_FINETUNED_MODEL_CHOICE)
    await message.answer(
        "Выберите дообученную модель для тестирования:",
        reply_markup=get_finetuned_model_kb()
    )


@router.message(lambda message: message.text in BASE_MODELS and get_user_state(
    message.from_user.id) == STATE_TEST_BASE_MODEL_CHOICE)
async def base_model_chosen_for_test(message: Message):
    """Базовая модель выбрана для тестирования"""
    if not message.from_user:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return

    # Маппинг на huggingface repo
    hf_map = {
        'CodeLlama-7b-Instruct-hf': 'codellama/CodeLlama-7b-Instruct-hf',
        'Mistral-7B-Instruct-v0.2': 'mistralai/Mistral-7B-Instruct-v0.2',
    }
    model_path = hf_map.get(message.text, message.text)

    set_user_state(message.from_user.id, STATE_TEST_UPLOAD_DATASET,
                   model_path=model_path, model_type='base')
    await message.answer(
        f"Выбрана базовая модель: {message.text}\n\n"
        "Теперь загрузите тестовый датасет в формате Excel (.xlsx).\n\n"
        "Файл должен содержать следующие столбцы:\n"
        "- bad_sql: неправильный SQL-запрос\n"
        "- good_sql: ожидаемый исправленный SQL-запрос\n"
        "- reason: причина по которой запрос неправильный\n"
        "- fix: ожидаемое описание исправления\n\n"
        "Пожалуйста, отправьте файл тестового датасета:"
    )


@router.message(lambda message: get_user_state(message.from_user.id) == STATE_TEST_FINETUNED_MODEL_CHOICE)
async def finetuned_model_chosen_for_test(message: Message):
    """Дообученная модель выбрана для тестирования"""
    if not message.from_user or not message.text:
        await message.answer("Ошибка: не удалось определить пользователя или модель.")
        return

    finetuned_models = get_finetuned_models()
    if message.text not in finetuned_models:
        await message.answer(
            "Выбранная модель не найдена. Выберите из доступных:",
            reply_markup=get_finetuned_model_kb()
        )
        return

    set_user_state(message.from_user.id, STATE_TEST_UPLOAD_DATASET,
                   model_path=message.text, model_type='finetuned')
    await message.answer(
        f"Выбрана дообученная модель: {message.text}\n\n"
        "Теперь загрузите тестовый датасет в формате Excel (.xlsx).\n\n"
        "Файл должен содержать следующие столбцы:\n"
        "- bad_sql: неправильный SQL-запрос\n"
        "- good_sql: ожидаемый исправленный SQL-запрос\n"
        "- fix: ожидаемое описание исправления\n\n"
        "Пожалуйста, отправьте файл тестового датасета:"
    )


@router.message(lambda message: message.document is not None)
async def handle_file(message: Message):
    """Обработка загруженных файлов"""
    if not message.from_user or not message.document:
        await message.answer("Ошибка: не удалось получить файл.")
        return

    user_id = message.from_user.id
    state = get_user_state(user_id)

    # Загружаем файл
    file_info = await bot.get_file(message.document.file_id)
    file_path = f"{user_id}_{message.document.file_name}"

    if not file_info.file_path:
        await message.answer("Ошибка: не удалось получить путь к файлу.")
        return

    await bot.download_file(file_info.file_path, file_path)

    if state == STATE_TRAIN_UPLOAD_DATASET:
        # Датасет для обучения загружен
        set_user_state(user_id, STATE_TRAIN_MODEL_CHOICE, dataset_path=file_path)
        await message.answer(
            "Файл датасета для обучения успешно загружен!\n"
            "Выберите модель для дообучения:",
            reply_markup=get_finetune_model_kb()
        )

    elif state == STATE_TEST_UPLOAD_DATASET:
        # Тестовый датасет загружен
        model_path = get_user_data(user_id, 'model_path')
        model_type = get_user_data(user_id, 'model_type')

        await message.answer("Запускается тестирование модели. Это может занять некоторое время...")

        try:
            # Запускаем тестирование в отдельном потоке
            loop = asyncio.get_event_loop()
            result_path = await loop.run_in_executor(
                None,
                lambda: test_model(file_path, model_path, model_type)
            )

            # Отправляем результаты
            with open(result_path, 'rb') as f:
                await message.answer_document(
                    FSInputFile(result_path),
                    caption=f"Результаты тестирования модели: {model_path}"
                )

            await message.answer(
                "Тестирование завершено! Результаты сохранены в файле.",
                reply_markup=get_main_kb()
            )

        except Exception as e:
            await message.answer(f"Ошибка при тестировании: {str(e)}")

        set_user_state(user_id, STATE_MAIN)

    elif state == STATE_SCHEMA_UPLOAD:
        # Схема БД загружена
        set_user_state(user_id, STATE_MAIN, schema_path=file_path)
        await message.answer(
            "Файл схемы БД успешно загружен!",
            reply_markup=get_main_kb()
        )


@router.message(lambda message: message.text in HF_FINETUNE_MODELS and get_user_state(
    message.from_user.id) == STATE_TRAIN_MODEL_CHOICE)
async def handle_finetune_model_choice(message: Message):
    """Выбор модели для дообучения"""
    if not message.from_user:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return

    user_id = message.from_user.id
    set_user_state(user_id, STATE_TRAIN_QUANT_CHOICE, model_name=message.text)
    await message.answer("Выберите тип квантования:", reply_markup=get_quant_kb())


@router.message(lambda message: message.text in ['4bit', '8bit', 'fp16'] and get_user_state(
    message.from_user.id) == STATE_TRAIN_QUANT_CHOICE)
async def handle_finetune_quant_choice(message: Message):
    """Выбор типа квантования и запуск обучения"""
    if not message.from_user:
        await message.answer("Ошибка: не удалось определить пользователя.")
        return

    user_id = message.from_user.id
    quant = message.text
    dataset_path = get_user_data(user_id, 'dataset_path')
    model_name = get_user_data(user_id, 'model_name')

    if not dataset_path or not model_name:
        await message.answer("Ошибка: отсутствуют данные для обучения.")
        set_user_state(user_id, STATE_MAIN)
        return

    # Маппинг на huggingface repo
    hf_map = {
        'CodeLlama-7b-Instruct-hf': 'codellama/CodeLlama-7b-Instruct-hf',
        'Mistral-7B-Instruct-v0.2': 'mistralai/Mistral-7B-Instruct-v0.2',
    }
    base_model = hf_map.get(model_name, model_name)
    output_dir = f"finetuned_{user_id}_{model_name}_{quant}"

    await message.answer(
        f"Запускается дообучение модели {model_name} с квантованием {quant}.\n"
        "Это может занять много времени..."
    )

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: train_bnb_lora(
                dataset_path,
                base_model=base_model,
                output_dir=output_dir,
                quantization_type=quant
            )
        )
        await message.answer(
            f"Дообучение завершено! LoRA-адаптер сохранён в папке {output_dir}",
            reply_markup=get_main_kb()
        )
    except Exception as e:
        await message.answer(f"Ошибка при дообучении: {str(e)}")

    set_user_state(user_id, STATE_MAIN)


@router.message()
async def handle_other_messages(message: Message):
    """Обработка прочих сообщений"""
    if not message.from_user:
        return

    state = get_user_state(message.from_user.id)

    if state == STATE_MAIN:
        await message.answer(
            "Выберите действие из меню:",
            reply_markup=get_main_kb()
        )
    else:
        await message.answer("Пожалуйста, следуйте инструкциям или вернитесь в главное меню с помощью /start")


async def main():
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())