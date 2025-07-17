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
    'TheBloke/sqlcoder-7B-GGUF': os.getenv('GGUF_SQLCODER_PATH', 'local_models/sqlcoder-7b.Q4_K_M.gguf'),
    'TheBloke/CodeLlama-13B-GGUF': os.getenv('GGUF_CODELLAMA13B_PATH', 'local_models/codellama-13b.Q4_K_M.gguf'),
    'TheBloke/CodeLlama-7B-Instruct-GGUF': os.getenv('GGUF_CODELLAMA7B_INSTRUCT_PATH', 'local_models/codellama-7b-instruct.Q4_K_M.gguf'),
    'TheBloke/sqlcoder-GGUF': os.getenv('GGUF_SQLCODER_Q4_PATH', 'local_models/sqlcoder.Q4_K_M.gguf'),
}

def download_gguf_from_hf(model_name: str):
    """
    Скачивает GGUF-файл с Hugging Face, если его нет локально, с прогресс-баром.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import tqdm
    if model_name == 'TheBloke/sqlcoder-7B-GGUF':
        repo_id = 'TheBloke/sqlcoder-7B-GGUF'
        filename = 'sqlcoder-7b.Q4_K_M.gguf'
        local_dir = 'local_models'
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        if not os.path.exists(local_path):
            print(f"Скачивание {filename} из {repo_id}...")
            # huggingface_hub уже использует tqdm, но можно явно включить прогресс
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        return local_path
    if model_name == 'TheBloke/CodeLlama-13B-GGUF':
        repo_id = 'TheBloke/CodeLlama-13B-GGUF'
        filename = 'codellama-13b.Q4_K_M.gguf'
        local_dir = 'local_models'
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        if not os.path.exists(local_path):
            print(f"Скачивание {filename} из {repo_id}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        return local_path
    if model_name == 'TheBloke/CodeLlama-7B-Instruct-GGUF':
        repo_id = 'TheBloke/CodeLlama-7B-Instruct-GGUF'
        filename = 'codellama-7b-instruct.Q4_K_M.gguf'
        local_dir = 'local_models'
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        if not os.path.exists(local_path):
            print(f"Скачивание {filename} из {repo_id}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        return local_path
    if model_name == 'TheBloke/sqlcoder-GGUF':
        repo_id = 'TheBloke/sqlcoder-GGUF'
        filename = 'sqlcoder.Q4_K_M.gguf'
        local_dir = 'local_models'
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        if not os.path.exists(local_path):
            print(f"Скачивание {filename} из {repo_id}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        return local_path
    raise ValueError(f"Неизвестная GGUF модель: {model_name}")

def load_llm_pipeline(model_name: str, quantization: str = None):
    """
    Загружает и возвращает pipeline для указанной модели с оптимизацией памяти (8bit/4bit quantization).
    quantization: None | '8bit' | '4bit'
    Квантизация поддерживается только на GPU. На CPU загружается обычная модель.
    """
    if model_name in _loaded_pipelines:
        return _loaded_pipelines[model_name]

    model_path = model_name  # Можно добавить маппинг, если нужно
    logger.info(f"Загрузка модели {model_name} с Hugging Face...")
    if quantization in ("8bit", "4bit") and torch.cuda.is_available():
        try:
            import bitsandbytes
            from transformers import BitsAndBytesConfig
            model_kwargs = {"token": HF_TOKEN}
            quant_config = None
            if quantization == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "4bit":
                quant_config = BitsAndBytesConfig(load_in_4bit=True)
            if quant_config:
                model_kwargs["quantization_config"] = quant_config
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        except ImportError:
            logger.warning("bitsandbytes не установлен, загружаем модель в обычном режиме (RAM usage будет выше)")
            model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN)
            tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    else:
        if quantization in ("8bit", "4bit"):
            logger.warning("Квантизация 8bit/4bit поддерживается только на GPU. Модель будет загружена в обычном режиме на CPU.")
        model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    _loaded_models[model_name] = model
    _loaded_tokenizers[model_name] = tokenizer
    _loaded_pipelines[model_name] = pipe
    return pipe

def load_llama_cpp_model(model_name: str) -> Optional[object]:
    """
    Загружает GGUF модель через llama-cpp-python. Если файла нет, скачивает с Hugging Face.
    """
    if model_name in _loaded_llama_cpp_models:
        return _loaded_llama_cpp_models[model_name]
    model_path = GGUF_MODEL_MAP.get(model_name)
    if not model_path or not os.path.exists(model_path):
        model_path = download_gguf_from_hf(model_name)
    llm = Llama(model_path=model_path, n_ctx=2048)
    _loaded_llama_cpp_models[model_name] = llm
    return llm

def generate_llm_response(prompt: str, model_name: str, quantization: str = None) -> str:
    """
    Универсальная функция для генерации ответа от выбранной LLM.
    model_name: например, 'defog/sqlcoder-7b-2', 'meta-llama/CodeLlama-7b-hf', 'TheBloke/sqlcoder-7B-GGUF', 'TheBloke/CodeLlama-13B-GGUF' и др.
    quantization: None | '8bit' | '4bit'
    """
    if model_name in ['TheBloke/sqlcoder-7B-GGUF', 'TheBloke/CodeLlama-13B-GGUF',
                      'TheBloke/CodeLlama-7B-Instruct-GGUF', 'TheBloke/sqlcoder-GGUF']:
        llm = load_llama_cpp_model(model_name)
        output = llm(prompt, max_tokens=512, stop=["\n"], echo=False)
        return output["choices"][0]["text"].strip()
    else:
        llm_pipeline = load_llm_pipeline(model_name, quantization=quantization)
        result = llm_pipeline(prompt, return_full_text=False)
        return result[0]["generated_text"] if result else ""
