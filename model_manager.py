import logging
import os
import re
from typing import Optional, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from transformers.pipelines import Pipeline
import torch

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("llm_model_loader")

# Константы
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_QUANTIZATION = '4bit'  # теперь по умолчанию 4bit через bitsandbytes
MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.3

# Кэширование моделей
_loaded_pipelines: Dict[str, Pipeline] = {}


def log_cuda_memory():
    """Логирует использование CUDA памяти."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        logger.info(f"CUDA memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")


def get_quantization_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    """Создает конфигурацию квантования для bitsandbytes."""
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True
        )
    elif quantization == "none" or quantization is None:
        return None
    else:
        logger.warning(f"Неизвестный тип квантования: {quantization}. Используется без квантования.")
        return None


def load_llm_pipeline(model_name: str, quantization: str = DEFAULT_QUANTIZATION) -> Pipeline:
    """Загружает pipeline с учётом bitsandbytes квантования."""
    cache_key = f"{model_name}_{quantization}"
    if cache_key in _loaded_pipelines:
        logger.info(f"Модель {model_name} уже загружена из кэша")
        return _loaded_pipelines[cache_key]

    logger.info(f"Загрузка модели: {model_name} с типом квантования: {quantization}")

    try:
        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN,
            use_fast=True,
            trust_remote_code=True
        )

        # Устанавливаем pad_token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Получаем конфигурацию квантования
        quantization_config = get_quantization_config(quantization)

        # Загружаем модель
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "token": HF_TOKEN
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Создаем pipeline
        device = 0 if torch.cuda.is_available() and quantization == "none" else -1
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        log_cuda_memory()

        _loaded_pipelines[cache_key] = pipe
        logger.info(f"Модель {model_name} успешно загружена")
        return pipe

    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {model_name}: {e}")
        raise


def clean_model_response(response: str, original_prompt: str = "") -> str:
    """Очищает ответ модели от артефактов и исходного промпта."""
    if not response:
        return ""

    # Удаляем исходный промпт из ответа, если он там есть
    if original_prompt and response.startswith(original_prompt):
        response = response[len(original_prompt):].strip()

    # Очищаем различные артефакты
    response = re.sub(r'\[UNK_BYTE_[^\]]*\]', '', response)
    response = response.replace("]", " ")
    response = re.sub(r'\s+', ' ', response)
    response = response.replace('```', '').strip('`')

    # Добавляем пробелы между словами там, где нужно
    response = re.sub(r'([a-z])([A-Z])', r'\1 \2', response)
    response = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', response)
    response = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', response)

    return response.strip()


def generate_llm_response(
        prompt: str,
        model_name: str,
        quantization: str = DEFAULT_QUANTIZATION,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE
) -> str:
    """Генерирует ответ от LLM модели."""
    try:
        return _generate_transformers_response(
            prompt, model_name, quantization, max_new_tokens, temperature
        )
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return f"ERROR: {str(e)}"


def _generate_transformers_response(
        prompt: str,
        model_name: str,
        quantization: str,
        max_new_tokens: int,
        temperature: float
) -> str:
    """Внутренняя функция для генерации ответа через transformers."""
    llm_pipeline = load_llm_pipeline(model_name, quantization=quantization)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 50,
        "num_return_sequences": 1,
        "do_sample": True if temperature > 0 else False,
        "truncation": True,
        "pad_token_id": llm_pipeline.tokenizer.eos_token_id,
        "return_full_text": False  # Возвращаем только новый текст
    }

    logger.debug(f"Генерация с параметрами: {generation_kwargs}")

    result = llm_pipeline(prompt, **generation_kwargs)

    raw_response = result[0]["generated_text"] if result else ""
    logger.debug(f"Raw pipeline response:\n{raw_response}")

    cleaned_response = clean_model_response(raw_response, prompt)

    if cleaned_response:
        logger.info(f"Cleaned pipeline response:\n{cleaned_response}")

    return cleaned_response


def clear_model_cache():
    """Очищает кэш загруженных моделей."""
    global _loaded_pipelines
    _loaded_pipelines.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Кэш моделей очищен")


def get_supported_models():
    """Возвращает список популярных моделей для тестирования."""
    return [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "NousResearch/Llama-2-7b-chat-hf",
        "huggingface/CodeBERTa-small-v1"
    ]