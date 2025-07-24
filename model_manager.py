import logging
import os
import re
from typing import Optional, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines import Pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig
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
DEFAULT_QUANTIZATION = '8bit'
MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.3

# Кэширование моделей
_loaded_models: Dict[str, Any] = {}
_loaded_tokenizers: Dict[str, Any] = {}
_loaded_pipelines: Dict[str, Pipeline] = {}

def load_llm_pipeline(model_name: str, quantization: str = DEFAULT_QUANTIZATION) -> Pipeline:
    """Загружает pipeline для модели с поддержкой квантования."""
    if model_name in _loaded_pipelines:
        return _loaded_pipelines[model_name]

    model_kwargs = {"token": HF_TOKEN}
    quant_config = None

    if quantization in ("8bit", "4bit") and torch.cuda.is_available():
        try:
            import bitsandbytes
            if quantization == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "4bit":
                quant_config = BitsAndBytesConfig(load_in_4bit=True)
        except ImportError:
            logger.warning("bitsandbytes не установлен, используется обычная загрузка")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    _loaded_pipelines[model_name] = pipe
    return pipe

def clean_model_response(response: str) -> str:
    """Очищает ответ модели от специальных символов и артефактов."""
    if not response:
        return ""

    # Удаление специальных последовательностей
    response = re.sub(r'\[UNK_BYTE_[^\]]*\]', '', response)
    response = response.replace("]", " ")
    response = re.sub(r'\s+', ' ', response)
    response = response.replace('```', '').strip('`')

    # Разделение склеенных слов
    response = re.sub(r'([a-z])([A-Z])', r'\1 \2', response)
    response = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', response)
    response = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', response)

    return response.strip()

def generate_llm_response(prompt: str, model_name: str, quantization: str = DEFAULT_QUANTIZATION) -> str:
    """Генерирует ответ от выбранной LLM с обработкой ошибок."""
    try:
        return _generate_transformers_response(prompt, model_name, quantization)
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return f"ERROR: {str(e)}"

def _generate_transformers_response(prompt: str, model_name: str, quantization: str) -> str:
    """Генерация ответа для transformers моделей."""
    llm_pipeline = load_llm_pipeline(model_name, quantization=quantization)
    
    result = llm_pipeline(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.1,
        top_k=10,
        num_return_sequences=1,
        do_sample=True,
        truncation=True,
        pad_token_id=llm_pipeline.tokenizer.eos_token_id
    )
    raw_response = result[0]["generated_text"] if result else ""
    logger.debug(f"Raw pipeline response:\n{raw_response}")

    cleaned_response = clean_model_response(raw_response)
    if cleaned_response:
        logger.info(f"Cleaned pipeline response:\n{cleaned_response}")

    return cleaned_response
