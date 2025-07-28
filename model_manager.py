import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

logger = logging.getLogger(__name__)

# Кэш для загруженных моделей
_model_cache = {}


def get_bnb_config(quantization="4bit"):
    """Создает конфигурацию для bitsandbytes квантования"""
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        return None


def load_model_and_tokenizer(model_name, quantization="4bit"):
    """
    Загружает модель и токенизатор с кэшированием

    Args:
        model_name: Название модели (HuggingFace ID или локальный путь)
        quantization: Тип квантования ("4bit", "8bit", "fp16")

    Returns:
        tuple: (model, tokenizer)
    """
    cache_key = f"{model_name}_{quantization}"

    if cache_key in _model_cache:
        logger.info(f"Используем кэшированную модель: {model_name}")
        return _model_cache[cache_key]

    logger.info(f"Загружаем модель: {model_name} с квантованием {quantization}")

    # Определяем, является ли это дообученной моделью (локальный путь с адаптерами)
    is_finetuned = os.path.exists(model_name) and os.path.exists(os.path.join(model_name, "adapter_config.json"))

    if is_finetuned:
        # Загрузка дообученной модели с LoRA
        import json
        with open(os.path.join(model_name, "adapter_config.json")) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "")

        if not base_model_name:
            # Пытаемся определить по названию папки
            if "CodeLlama" in model_name:
                base_model_name = "codellama/CodeLlama-7b-Instruct-hf"
            elif "Mistral" in model_name:
                base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            else:
                raise ValueError(f"Не удалось определить базовую модель для {model_name}")

        logger.info(f"Загружаем базовую модель: {base_model_name}")

        # Токенизатор
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Базовая модель
        bnb_config = get_bnb_config(quantization)
        if bnb_config:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

        # Применяем LoRA адаптер
        model = PeftModel.from_pretrained(base_model, model_name)

    else:
        # Загрузка базовой модели
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        bnb_config = get_bnb_config(quantization)
        if bnb_config:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

    model.eval()

    # Кэшируем модель
    _model_cache[cache_key] = (model, tokenizer)

    logger.info(f"Модель {model_name} успешно загружена")
    return model, tokenizer


def generate_llm_response(prompt, model_name, quantization="4bit", max_new_tokens=256):
    """
    Генерирует ответ модели на промпт

    Args:
        prompt: Входной промпт
        model_name: Название модели
        quantization: Тип квантования
        max_new_tokens: Максимальное количество новых токенов

    Returns:
        str: Сгенерированный ответ
    """
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, quantization)
        device = next(model.parameters()).device

        # Токенизируем промпт
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Генерируем ответ
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Декодируем ответ
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Убираем исходный промпт из ответа
        response = generated_text[len(prompt):].strip()

        return response

    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {str(e)}")
        raise


def clear_model_cache():
    """Очищает кэш моделей для освобождения памяти"""
    global _model_cache
    logger.info("Очистка кэша моделей...")

    for cache_key in _model_cache:
        del _model_cache[cache_key]

    _model_cache = {}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Кэш моделей очищен")


def get_available_models():
    """Возвращает список доступных моделей"""
    base_models = [
        'codellama/CodeLlama-7b-Instruct-hf',
        'mistralai/Mistral-7B-Instruct-v0.2'
    ]

    # Добавляем дообученные модели
    finetuned_models = []
    current_dir = os.getcwd()

    for item in os.listdir(current_dir):
        if os.path.isdir(item) and (item.startswith("finetuned_") or "bnb-lora-finetuned" in item):
            adapter_config_path = os.path.join(item, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                finetuned_models.append(item)

    return {
        "base_models": base_models,
        "finetuned_models": finetuned_models
    }