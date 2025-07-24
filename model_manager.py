import logging
import os
import re
from typing import Optional, Dict, Any
from collections.abc import Iterator

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines import Pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch
from llama_cpp import Llama

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
_loaded_llama_cpp_models: Dict[str, Llama] = {}

# Конфигурация GGUF моделей
GGUF_MODEL_MAP = {
    'TheBloke/sqlcoder-7B-GGUF': os.getenv('GGUF_SQLCODER_PATH', 'local_models/sqlcoder-7b.Q4_K_M.gguf'),
    'TheBloke/CodeLlama-13B-GGUF': os.getenv('GGUF_CODELLAMA13B_PATH', 'local_models/codellama-13b.Q4_K_M.gguf'),
    'TheBloke/CodeLlama-7B-Instruct-GGUF': os.getenv('GGUF_CODELLAMA7B_INSTRUCT_PATH', 'local_models/codellama-7b-instruct.Q4_K_M.gguf'),
    'TheBloke/sqlcoder-GGUF': os.getenv('GGUF_SQLCODER_Q4_PATH', 'local_models/sqlcoder.Q4_K_M.gguf'),
    'MaziyarPanahi/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF': os.getenv('GGUF_MAZIYARPANAHI_SQLCODER_PATH', 'local_models/sqlcoder-7b-mistral-instruct.Q4_K_M.gguf'),
    'TheBloke/Amethyst-13B-Mistral-GGUF': os.getenv('GGUF_AMETHYST13B_PATH', 'local_models/amethyst-13b-mistral.Q4_K_M.gguf'),
}

def download_gguf_from_hf(model_name: str) -> str:
    """Скачивает GGUF-файл с Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    
    model_config = {
        'TheBloke/sqlcoder-7B-GGUF': ('sqlcoder-7b.Q4_K_M.gguf', 'TheBloke/sqlcoder-7B-GGUF'),
        'TheBloke/CodeLlama-13B-GGUF': ('codellama-13b.Q4_K_M.gguf', 'TheBloke/CodeLlama-13B-GGUF'),
        'TheBloke/CodeLlama-7B-Instruct-GGUF': ('codellama-7b-instruct.Q4_K_M.gguf', 'TheBloke/CodeLlama-7B-Instruct-GGUF'),
        'TheBloke/sqlcoder-GGUF': ('sqlcoder.Q4_K_M.gguf', 'TheBloke/sqlcoder-GGUF'),
        'MaziyarPanahi/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF': ('sqlcoder-7b-mistral-instruct.Q4_K_M.gguf', 'MaziyarPanahi/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF'),
        'TheBloke/Amethyst-13B-Mistral-GGUF': ('amethyst-13b-mistral.Q4_K_M.gguf', 'TheBloke/Amethyst-13B-Mistral-GGUF'),
    }
    
    filename, repo_id = model_config[model_name]
    local_dir = 'local_models'
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    
    if not os.path.exists(local_path):
        logger.info(f"Скачивание {filename} из {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    return local_path

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

def load_llama_cpp_model(model_name: str) -> Optional[Llama]:
    """Загружает GGUF модель через llama-cpp-python."""
    if model_name in _loaded_llama_cpp_models:
        return _loaded_llama_cpp_models[model_name]

    model_path = GGUF_MODEL_MAP.get(model_name)
    if not model_path or not os.path.exists(model_path):
        model_path = download_gguf_from_hf(model_name)

    n_ctx = 2048
    n_threads = max(1, (os.cpu_count() or 2) - 1)
    n_gpu_layers = 1 if torch.cuda.is_available() else 0

    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            use_mmap=True,
            verbose=False
        )
        _loaded_llama_cpp_models[model_name] = llm
        return llm
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {model_name}: {e}")
        return None

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
        if model_name in GGUF_MODEL_MAP:
            return _generate_gguf_response(prompt, model_name)
        return _generate_transformers_response(prompt, model_name, quantization)
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        return f"ERROR: {str(e)}"

def _generate_gguf_response(prompt: str, model_name: str) -> str:
    """Генерация ответа для GGUF моделей."""
    llm = load_llama_cpp_model(model_name)
    if not llm:
        raise RuntimeError(f"Не удалось загрузить модель {model_name}")
    
    params = {
        'max_tokens': MAX_NEW_TOKENS,
        'temperature': 0.1,
        'top_p': 0.05,
        'top_k': 5,
        'repeat_penalty': 1.5,
        'presence_penalty': 0.5,
        'frequency_penalty': 0.5,
        'stop': ["```", "###", "--", "*/", ";\\n"],
        'echo': False,
        'grammar': None,
        'mirostat_mode': 2,
        'mirostat_tau': 3.0,
        'mirostat_eta': 0.1,
    }

    if 'sqlcoder' not in model_name.lower():
        params.update({
            'temperature': 0.5,
            'top_p': 0.1,
            'top_k': 10,
            'repeat_penalty': 1.2,
        })

    output = llm.create_completion(prompt=prompt, **params)
    if isinstance(output, Iterator):
        first = next(output)
    else:
        first = output
    raw_response = first["choices"][0]["text"].strip()

    if 'sqlcoder' in model_name.lower():
        raw_response = raw_response.replace('[/SQL]', '')
        raw_response = raw_response.replace('[SQL]', '')
        raw_response = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw_response)
        raw_response = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', raw_response)
        raw_response = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', raw_response)

    logger.debug(f"Raw response from {model_name}:\n{raw_response}")
    cleaned_response = clean_model_response(raw_response)
    logger.info(f"Cleaned response:\n{cleaned_response}")
    return cleaned_response

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
        max_length=1024,
        truncation=True,
        pad_token_id=llm_pipeline.tokenizer.eos_token_id
    )
    raw_response = result[0]["generated_text"] if result else ""
    logger.debug(f"Raw pipeline response:\n{raw_response}")

    cleaned_response = clean_model_response(raw_response)
    if cleaned_response:
        logger.info(f"Cleaned pipeline response:\n{cleaned_response}")

    return cleaned_response
