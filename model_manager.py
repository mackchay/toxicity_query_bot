import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("llm_model_loader")

# Кэш для уже загруженных моделей и пайплайнов
_loaded_models = {}
_loaded_tokenizers = {}
_loaded_pipelines = {}

def load_llm_pipeline(model_name: str):
    """
    Загружает и возвращает pipeline для указанной модели.
    Кэширует pipeline, чтобы не загружать повторно.
    """
    if model_name in _loaded_pipelines:
        return _loaded_pipelines[model_name]
    logger.info(f"Загрузка токенизатора для {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Токенизатор для {model_name} загружен.")
    logger.info(f"Загрузка модели {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        # device_map="auto"  # убрано для совместимости без accelerate
    )
    logger.info(f"Модель {model_name} загружена.")
    logger.info(f"Создание пайплайна для {model_name}...")
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # всегда CPU
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2
    )
    logger.info(f"Пайплайн для {model_name} готов к работе.")
    _loaded_models[model_name] = model
    _loaded_tokenizers[model_name] = tokenizer
    _loaded_pipelines[model_name] = llm_pipeline
    return llm_pipeline

def generate_llm_response(prompt: str, model_name: str) -> str:
    """
    Универсальная функция для генерации ответа от выбранной LLM.
    model_name: например, 'defog/sqlcoder-7b-2', 'meta-llama/CodeLlama-7b-hf' и др.
    """
    llm_pipeline = load_llm_pipeline(model_name)
    result = llm_pipeline(prompt, return_full_text=False)
    return result[0]["generated_text"] if result else ""
