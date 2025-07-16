import logging
import os
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("llm_model_loader")

# Кэш для уже загруженных моделей и пайплайнов
_loaded_models = {}
_loaded_tokenizers = {}
_loaded_pipelines = {}
# Кэш для GGUF моделей
_loaded_llama_cpp_models = {}

HF_TOKEN = os.getenv("HF_TOKEN")

GGUF_MODEL_MAP = {
    'TheBloke/sqlcoder-7B-GGUF': 'local_models/sqlcoder-7b.Q4_K_M.gguf',  # путь к файлу GGUF
}

def load_llm_pipeline(model_name: str, quantization: str = None):
    """
    Загружает и возвращает pipeline для указанной модели с оптимизацией памяти (8bit/4bit quantization).
    quantization: None | '8bit' | '4bit'
    """
    if model_name in _loaded_pipelines:
        return _loaded_pipelines[model_name]

    model_path = model_name  # Можно добавить маппинг, если нужно
    logger.info(f"Загрузка модели {model_name} с Hugging Face с оптимизацией памяти...")
    try:
        import bitsandbytes
        model_kwargs = {"device_map": "auto", "token": HF_TOKEN}
        if quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
        elif quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    except ImportError:
        logger.warning("bitsandbytes не установлен, загружаем модель в обычном режиме (RAM usage будет выше)")
        model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    _loaded_models[model_name] = model
    _loaded_tokenizers[model_name] = tokenizer
    _loaded_pipelines[model_name] = pipe
    return pipe

def load_llama_cpp_model(model_name: str) -> Optional[object]:
    """
    Загружает GGUF модель через llama-cpp-python.
    """
    if model_name in _loaded_llama_cpp_models:
        return _loaded_llama_cpp_models[model_name]
    model_path = GGUF_MODEL_MAP.get(model_name)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"GGUF файл для {model_name} не найден: {model_path}")
    llm = Llama(model_path=model_path, n_ctx=2048)
    _loaded_llama_cpp_models[model_name] = llm
    return llm

def generate_llm_response(prompt: str, model_name: str, quantization: str = None) -> str:
    """
    Универсальная функция для генерации ответа от выбранной LLM.
    model_name: например, 'defog/sqlcoder-7b-2', 'meta-llama/CodeLlama-7b-hf', 'TheBloke/sqlcoder-7B-GGUF' и др.
    quantization: None | '8bit' | '4bit'
    """
    if model_name == 'TheBloke/sqlcoder-7B-GGUF':
        llm = load_llama_cpp_model(model_name)
        output = llm
        return output["choices"][0]["text"].strip()
    else:
        llm_pipeline = load_llm_pipeline(model_name, quantization=quantization)
        result = llm_pipeline(prompt, return_full_text=False)
        return result[0]["generated_text"] if result else ""
