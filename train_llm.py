import os
import torch
import pandas as pd
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("train_bnb_lora")

def load_dataset(xlsx_path):
    logger.info(f"Загрузка датасета из {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    required_columns = ['BAD_SQL', 'GOOD_SQL', 'REASON']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Отсутствует обязательный столбец: {col}")
    df = df.dropna(subset=['BAD_SQL', 'GOOD_SQL'])

    df['input_text'] = df.apply(lambda row: f"BAD_SQL: {row['BAD_SQL']}\nREASON: {row['REASON']}", axis=1)
    df['target_text'] = df['GOOD_SQL'].astype(str)

    logger.info(f"Загружено {len(df)} примеров для обучения")
    return Dataset.from_pandas(df[['input_text', 'target_text']])

def tokenize_function(example, tokenizer, max_length=512):
    model_inputs = tokenizer(
        example['input_text'],
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example['target_text'],
            max_length=max_length,
            padding='max_length',
            truncation=True
        )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def get_bnb_config(quantization_type="4bit"):
    if quantization_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_type == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Неподдерживаемый тип квантования: {quantization_type}")

def train_bnb_lora(
        xlsx_path,
        base_model="meta-llama/Llama-2-7b-hf",
        output_dir="./bnb-lora-finetuned",
        quantization_type="4bit",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        batch_size=1,
        epochs=3,
        lr=2e-4,
        max_length=512,
        gradient_accumulation_steps=16
):
    logger.info(f"=== НАЧАЛО ДООБУЧЕНИЯ С BITSANDBYTES ===")
    logger.info(f"Базовая модель: {base_model}")
    logger.info(f"Тип квантования: {quantization_type}")
    logger.info(f"Выходная директория: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = get_bnb_config(quantization_type)
    logger.info("Загружаем базовую модель...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    logger.info("Применяем LoRA адаптеры...")
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Тренируемые параметры: {trainable_params:,}")
    logger.info(f"Все параметры: {all_params:,}")
    logger.info(f"% тренируемых параметров: {100 * trainable_params / all_params:.2f}%")

    dataset = load_dataset(xlsx_path)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=False,
        remove_columns=dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    model.config.use_cache = False

    try:
        logger.info("Запускаем процесс дообучения...")
        trainer.train()
        logger.info("Сохраняем дообученную модель...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"=== ДООБУЧЕНИЕ ЗАВЕРШЕНО ===")
        logger.info(f"Модель сохранена в: {output_dir}")
    except Exception as e:
        logger.error(f"Ошибка во время обучения: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_model(model_path, test_prompt="BAD_SQL: SELECT * FROM users WHERE id = 1\nREASON: Использование SELECT *"):
    logger.info(f"Проверка модели из {model_path}")
    try:
        from model_manager import generate_llm_response
        response = generate_llm_response(
            prompt=test_prompt,
            model_name=model_path,
            quantization="4bit"
        )
        logger.info(f"Тестовый промпт: {test_prompt}")
        logger.info(f"Ответ модели: {response}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при проверке модели: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Дообучение LLM с использованием bitsandbytes и LoRA (Text-to-Text)")
    parser.add_argument("xlsx_path", help="Путь к xlsx файлу с данными для обучения")
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf", help="Название базовой модели на HuggingFace")
    parser.add_argument("--output_dir", default="./bnb-lora-finetuned", help="Директория для сохранения результатов")
    parser.add_argument("--quantization_type", choices=["4bit", "8bit"], default="4bit", help="Тип квантования")
    parser.add_argument("--lora_r", type=int, default=16, help="Ранг LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Альфа параметр LoRA")
    parser.add_argument("--batch_size", type=int, default=1, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--validate", action="store_true", help="Проверить модель после обучения")

    args = parser.parse_args()

    train_bnb_lora(
        xlsx_path=args.xlsx_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        quantization_type=args.quantization_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )

    if args.validate:
        validate_model(args.output_dir)
