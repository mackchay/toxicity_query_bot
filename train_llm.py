import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset

def make_prompt(row):
    # Если reason пустой — запрос исправен, иначе нужно исправить
    if not str(row['REASON']).strip():
        return (
            f"BAD_SQL: {row['BAD_SQL']}\n"
            f"GOOD_SQL: {row['BAD_SQL']}\n"
            f"REASON: \n"
            f"FIX: The query does not need to be fixed."
        )
    else:
        return (
            f"BAD_SQL: {row['BAD_SQL']}\n"
            f"GOOD_SQL: {row['GOOD_SQL']}\n"
            f"REASON: {row['REASON']}\n"
            f"FIX: {row['FIX']}"
        )

def load_dataset(xlsx_path):
    df = pd.read_excel(xlsx_path)
    # Проверка на нужные столбцы
    for col in ['BAD_SQL', 'GOOD_SQL', 'REASON', 'FIX']:
        if col not in df.columns:
            raise ValueError(f"В датасете отсутствует столбец: {col}")
    # Фильтруем только строки с BAD_SQL и GOOD_SQL
    df = df.dropna(subset=['BAD_SQL', 'GOOD_SQL'])
    # Формируем промпты
    df['prompt'] = df.apply(make_prompt, axis=1)
    return Dataset.from_pandas(df[['prompt']])

def tokenize_function(example, tokenizer, max_length=1024):
    return tokenizer(
        example["prompt"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

def train_lora(
    xlsx_path,
    base_model="meta-llama/Llama-2-7b-hf",
    output_dir="./lora-llm-finetuned",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    batch_size=2,
    epochs=3,
    lr=2e-4,
    max_length=1024,
    quantization="fp16"  # "fp16", "8bit", "4bit"
):
    # 1. Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    quant_args = {}
    if quantization == "8bit":
        quant_args["load_in_8bit"] = True
    elif quantization == "4bit":
        quant_args["load_in_4bit"] = True
        quant_args["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        quant_args["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        **quant_args
    )
    model = prepare_model_for_kbit_training(model)
    # 2. LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],  # для Llama
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    # 3. Датасет
    dataset = load_dataset(xlsx_path)
    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)
    # 4. Аргументы тренировки
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        output_dir=output_dir,
        save_total_limit=2,
        logging_steps=10,
        save_steps=100,
        fp16=(quantization == "fp16"),
        report_to="none"
    )
    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx_path", help="Путь к xlsx датасету")
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf", help="Имя модели HuggingFace")
    parser.add_argument("--output_dir", default="./lora-llm-finetuned", help="Куда сохранить результат")
    parser.add_argument("--quantization", choices=["fp16", "8bit", "4bit"], default="fp16", help="Тип квантования")
    args = parser.parse_args()
    train_lora(
        xlsx_path=args.xlsx_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        quantization=args.quantization
    )
