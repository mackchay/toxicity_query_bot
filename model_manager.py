import logging
import os
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch
from llama_cpp import Llama
from transformers.utils.quantization_config import BitsAndBytesConfig

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
    'MaziyarPanahi/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF': os.getenv('GGUF_MAZIYARPANAHI_SQLCODER_PATH', 'local_models/sqlcoder-7b-mistral-instruct.Q4_K_M.gguf'),
    'TheBloke/Amethyst-13B-Mistral-GGUF': os.getenv('GGUF_AMETHYST13B_PATH', 'local_models/amethyst-13b-mistral.Q4_K_M.gguf'),
}

def download_gguf_from_hf(model_name: str):
    """
    Скачивает GGUF-файл с Hugging Face, если его нет локально, с прогресс-баром.
    """
    from huggingface_hub import hf_hub_download
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
    if model_name == 'MaziyarPanahi/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF':
        repo_id = 'MaziyarPanahi/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF'
        filename = 'sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp.Q4_K_M.gguf'
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
    if model_name == 'TheBloke/Amethyst-13B-Mistral-GGUF':
        repo_id = 'TheBloke/Amethyst-13B-Mistral-GGUF'
        filename = 'amethyst-13b-mistral.Q4_K_M.gguf'
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

def load_llm_pipeline(model_name: str, quantization: str = '8bit'):
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
            model_kwargs = {"token": HF_TOKEN}
            quant_config = None
            if quantization == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "4bit":
                quant_config = BitsAndBytesConfig(load_in_4bit=True)
            if quant_config:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quant_config,
                    **model_kwargs
                )
            else:
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
    Использует оптимизации для ускорения загрузки и работы модели.
    """
    if model_name in _loaded_llama_cpp_models:
        return _loaded_llama_cpp_models[model_name]

    model_path = GGUF_MODEL_MAP.get(model_name)
    if not model_path or not os.path.exists(model_path):
        model_path = download_gguf_from_hf(model_name)

    # Определяем оптимальные параметры для модели
    n_ctx = 2048  # Размер контекста
    cpu_count = os.cpu_count() or 2
    n_threads = max(1, cpu_count - 1)  # Используем все ядра процессора кроме одного
    n_gpu_layers = 0  # По умолчанию не используем GPU

    # Проверяем наличие CUDA для возможности использования GPU
    if torch.cuda.is_available():
        try:
            n_gpu_layers = 1  # Можно увеличить если хватает видеопамяти
            logger.info(f"GPU доступен, будет использовано {n_gpu_layers} слоев на GPU")
        except Exception as e:
            logger.warning(f"GPU обнаружен, но не может быть использован: {e}")
            n_gpu_layers = 0

    logger.info(f"Загрузка модели {model_name} с параметрами:")
    logger.info(f"- Количество потоков: {n_threads}")
    logger.info(f"- Размер контекста: {n_ctx}")
    logger.info(f"- GPU слои: {n_gpu_layers}")

    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            use_mmap=True,  # Использовать memory mapping для быстрой загрузки
            use_mlock=False,  # Не блокировать память
            verbose=False  # Отключаем лишний вывод
        )
        _loaded_llama_cpp_models[model_name] = llm
        return llm
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {model_name}: {e}")
        # Пробуем загрузить с минимальными параметрами в случае ошибки
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=1,
            n_gpu_layers=0,
            use_mmap=True
        )
        _loaded_llama_cpp_models[model_name] = llm
        return llm

def clean_model_response(response: str) -> str:
    """
    Очищает ответ модели от специальных символов и проблем с кодировкой
    """
    if not response:
        return ""

    # Удаляем UNK_BYTE и другие специальные последовательности
    response = response.replace("[UNK_BYTE_0x20", " ")
    response = response.replace("[UNK_BYTE_0x0a", "\n")

    # Удаляем все оставшиеся [UNK_BYTE_ последовательности
    import re
    response = re.sub(r'\[UNK_BYTE_[^\]]*\]', '', response)

    # Удаляем разделители ]
    response = response.replace("]", " ")

    # Удаляем повторяющиеся пробелы
    response = re.sub(r'\s+', ' ', response)

    # Удаляем бэктики и другие специальные символы
    response = response.replace('```', '')
    response = response.strip('`')

    # Если в тексте есть длинные последовательности без пробелов, разбиваем их
    words = response.split()
    cleaned_words = []
    for word in words:
        if len(word) > 30:  # Если слово слишком длинное
            # Разбиваем по заглавным буквам
            split_word = re.sub(r'([A-Z][a-z])', r' \1', word)
            # Разбиваем по цифрам
            split_word = re.sub(r'(\d+)', r' \1 ', split_word)
            cleaned_words.extend(split_word.split())
        else:
            cleaned_words.append(word)

    response = ' '.join(cleaned_words)

    # Финальная очистка пробелов и пунктуации
    response = re.sub(r'\s+', ' ', response)
    response = re.sub(r'\s*([,.;])\s*', r'\1 ', response)

    return response.strip()

def generate_llm_response(prompt: str, model_name: str, quantization: str = '8bit') -> str:
    """
    Универсальная функция для генерации ответа от выбранной LLM.
    """
    if model_name in [
        'TheBloke/sqlcoder-7B-GGUF',
        'TheBloke/CodeLlama-13B-GGUF',
        'TheBloke/CodeLlama-7B-Instruct-GGUF',
        'TheBloke/sqlcoder-GGUF',
        'MaziyarPanahi/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF',
        'TheBloke/Amethyst-13B-Mistral-GGUF'
    ]:
        # llama-cpp
        llm_obj = load_llama_cpp_model(model_name)
        if llm_obj is None:
            raise RuntimeError(f"Не удалось загрузить модель {model_name} (llm is None)")
        llm: Llama = llm_obj  # type: ignore
        # Специальные настройки для SQLCoder
        params = {
            'max_tokens': 256,        # Ограничиваем длину ответа
            'temperature': 0.5,       # Делаем генерацию более детерминированной
            'top_p': 0.05,           # Сильно ограничиваем выбор токенов
            'top_k': 5,              # Очень строгий выбор следующего токена
            'repeat_penalty': 1.5,    # Сильный штраф за повторения
            'presence_penalty': 0.5,  # Штраф за повторяющиеся темы
            'frequency_penalty': 0.5, # Штраф за частые токены
            'stop': ["```", "###", "--", "*/", ";\\n"],  # Стоп-токены
            'echo': False,
            'grammar': None,         # Отключаем специальную грамматику
            'mirostat_mode': 2,      # Включаем mirostat для лучшего контроля
            'mirostat_tau': 3.0,     # Целевая перплексия
            'mirostat_eta': 0.1,     # Скорость обучения
        }

        # Для моделей не-SQLCoder используем более мягкие параметры
        if 'sqlcoder' not in model_name.lower():
            params.update({
                'temperature': 0.3,
                'top_p': 0.1,
                'top_k': 10,
                'repeat_penalty': 1.2,
            })

        import collections.abc
        output = llm.create_completion(prompt=prompt, **params)
        if isinstance(output, collections.abc.Iterator):
            first = next(output)
        else:
            first = output
        raw_response = first["choices"][0]["text"].strip()

        # Дополнительная обработка для SQLCoder
        if 'sqlcoder' in model_name.lower():
            # Удаляем специальные токены и маркеры
            raw_response = raw_response.replace('[/SQL]', '')
            raw_response = raw_response.replace('[SQL]', '')
            # Разделяем склеенные слова
            import re
            raw_response = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw_response)
            raw_response = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', raw_response)
            raw_response = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', raw_response)

        logger.debug(f"Raw response from {model_name}:\n{raw_response}")
        cleaned_response = clean_model_response(raw_response)
        logger.info(f"Cleaned response:\n{cleaned_response}")
        return cleaned_response
    else:
        # transformers
        # kanxxyc/Mistral-7B-SQLTuned поддерживается через transformers
        llm_pipeline = load_llm_pipeline(model_name, quantization=quantization or '8bit')
        result = llm_pipeline(
            prompt,
            max_length=256,
            temperature=0.3,
            top_p=0.1,
            top_k=10,
            num_return_sequences=1,
            do_sample=True
        )
        raw_response = result[0]["generated_text"] if result else ""
        logger.debug(f"Raw pipeline response:\n{raw_response}")

        cleaned_response = clean_model_response(raw_response)
        if cleaned_response:
            logger.info(f"Cleaned pipeline response:\n{cleaned_response}")

        return cleaned_response
