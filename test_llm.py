import torch
import pandas as pd
import sqlparse
import os
import logging
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("test_llm")


def levenshtein_similarity(a, b):
    """Вычисляет сходство между двумя строками по алгоритму Левенштейна"""
    return SequenceMatcher(None, a, b).ratio()


def is_valid_sql(sql: str) -> bool:
    """Проверяет, является ли строка валидным SQL-запросом"""
    try:
        parsed = sqlparse.parse(sql)
        return len(parsed) > 0 and len(parsed[0].tokens) > 0
    except Exception:
        return False


def load_model(model_path, model_type="base", quantization="4bit"):
    """
    Загружает модель (базовую или дообученную) с квантованием

    Args:
        model_path: Путь к модели или HuggingFace model ID
        model_type: "base" или "finetuned"
        quantization: Тип квантования ("4bit", "8bit", "fp16")

    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Загрузка модели: {model_path}, тип: {model_type}, квантование: {quantization}")

    # Настройка квантования
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:  # fp16
        bnb_config = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "base":
        # Загрузка базовой модели
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if bnb_config:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

    else:  # finetuned
        # Загрузка дообученной модели с LoRA адаптерами
        # Определяем базовую модель из папки дообученной модели
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Читаем конфигурацию адаптера, чтобы узнать базовую модель
            import json
            with open(os.path.join(model_path, "adapter_config.json")) as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "")

            if not base_model_name:
                # Пытаемся определить по названию папки
                if "CodeLlama" in model_path:
                    base_model_name = "codellama/CodeLlama-7b-Instruct-hf"
                elif "Mistral" in model_path:
                    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
                else:
                    raise ValueError(f"Не удалось определить базовую модель для {model_path}")
        else:
            raise ValueError(f"Файл adapter_config.json не найден в {model_path}")

        logger.info(f"Базовая модель для адаптера: {base_model_name}")

        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Загружаем базовую модель
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
        model = PeftModel.from_pretrained(base_model, model_path)

    model.eval()
    logger.info("Модель успешно загружена")
    return model, tokenizer


def generate_fix_and_sql(model, tokenizer, input_sql, max_new_tokens=256):
    """
    Генерирует исправленный SQL и описание исправления

    Args:
        model: Загруженная модель
        tokenizer: Токенизатор
        input_sql: Исходный SQL-запрос
        max_new_tokens: Максимальное количество новых токенов

    Returns:
        tuple: (corrected_sql, fix_description)
    """
    device = next(model.parameters()).device

    # Формируем промпт в том же формате, что использовался для обучения
    prompt = (
        f"BAD_SQL: {input_sql}\n"
        "GOOD_SQL: "
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

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

    # Извлекаем ответ (убираем исходный промпт)
    response = generated_text[len(prompt):].strip()

    # Парсим ответ для извлечения SQL и описания исправления
    corrected_sql = ""
    fix_description = ""

    lines = response.split('\n')
    current_section = ""

    for line in lines:
        line = line.strip()
        if line.startswith("REASON:"):
            current_section = "reason"
            continue
        elif line.startswith("FIX:"):
            current_section = "fix"
            fix_description = line[4:].strip()
            continue
        elif line.startswith("GOOD_SQL:"):
            current_section = "sql"
            corrected_sql = line[9:].strip()
            continue

        # Добавляем содержимое к соответствующей секции
        if current_section == "sql" and line:
            if corrected_sql:
                corrected_sql += " " + line
            else:
                corrected_sql = line
        elif current_section == "fix" and line:
            if fix_description:
                fix_description += " " + line
            else:
                fix_description = line

    # Если не удалось извлечь структурированный ответ, используем весь response как SQL
    if not corrected_sql:
        # Пытаемся найти SQL в начале ответа
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('REASON:', 'FIX:', 'BAD_SQL:', 'GOOD_SQL:')):
                corrected_sql = line
                break

        if not corrected_sql:
            corrected_sql = response.split('\n')[0].strip()

    if not fix_description:
        fix_description = "No fix description provided"

    return corrected_sql, fix_description


def test_model(test_data_path, model_path, model_type="base", quantization="4bit"):
    """
    Тестирует модель на тестовом датасете

    Args:
        test_data_path: Путь к файлу с тестовыми данными (Excel)
        model_path: Путь к модели или HuggingFace model ID
        model_type: "base" или "finetuned"
        quantization: Тип квантования ("4bit", "8bit", "fp16")

    Returns:
        str: Путь к файлу с результатами
    """
    logger.info(f"Начало тестирования модели: {model_path}")

    # Загружаем тестовые данные
    try:
        df = pd.read_excel(test_data_path)
        logger.info(f"Загружено {len(df)} тестовых примеров")
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке тестовых данных: {str(e)}")

    # Проверяем наличие необходимых столбцов
    required_columns = ['BAD_SQL', 'GOOD_SQL', 'FIX']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют обязательные столбцы: {missing_columns}")

    # Загружаем модель
    try:
        model, tokenizer = load_model(model_path, model_type, quantization)
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке модели: {str(e)}")

    results = []

    logger.info("Начинаем тестирование...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Тестирование"):
        bad_sql = str(row['BAD_SQL']).strip()
        expected_sql = str(row['GOOD_SQL']).strip()
        expected_fix = str(row['FIX']).strip()

        if not bad_sql:
            logger.warning(f"Пропущен пустой запрос в строке {idx}")
            continue

        try:
            predicted_sql, predicted_fix = generate_fix_and_sql(model, tokenizer, bad_sql)

            # Вычисляем метрики
            sql_similarity = levenshtein_similarity(predicted_sql.lower().strip(), expected_sql.lower().strip())
            fix_similarity = levenshtein_similarity(predicted_fix.lower().strip(), expected_fix.lower().strip())

            # Проверяем точное совпадение (с игнорированием регистра и пробелов)
            sql_exact_match = predicted_sql.lower().strip() == expected_sql.lower().strip()
            fix_exact_match = predicted_fix.lower().strip() == expected_fix.lower().strip()

            # Проверяем валидность SQL
            is_predicted_sql_valid = is_valid_sql(predicted_sql)
            is_expected_sql_valid = is_valid_sql(expected_sql)

            results.append({
                "test_id": idx + 1,
                "bad_sql": bad_sql,
                "expected_sql": expected_sql,
                "expected_fix": expected_fix,
                "predicted_sql": predicted_sql,
                "predicted_fix": predicted_fix,
                "sql_levenshtein_similarity": round(sql_similarity, 4),
                "fix_levenshtein_similarity": round(fix_similarity, 4),
                "sql_exact_match": sql_exact_match,
                "fix_exact_match": fix_exact_match,
                "predicted_sql_valid": is_predicted_sql_valid,
                "expected_sql_valid": is_expected_sql_valid,
                "both_sql_valid": is_predicted_sql_valid and is_expected_sql_valid
            })

        except Exception as e:
            logger.error(f"Ошибка при обработке строки {idx}: {str(e)}")
            results.append({
                "test_id": idx + 1,
                "bad_sql": bad_sql,
                "expected_sql": expected_sql,
                "expected_fix": expected_fix,
                "predicted_sql": f"ERROR: {str(e)}",
                "predicted_fix": f"ERROR: {str(e)}",
                "sql_levenshtein_similarity": 0.0,
                "fix_levenshtein_similarity": 0.0,
                "sql_exact_match": False,
                "fix_exact_match": False,
                "predicted_sql_valid": False,
                "expected_sql_valid": is_valid_sql(expected_sql),
                "both_sql_valid": False
            })

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)

    # Вычисляем общие метрики
    if len(results_df) > 0:
        metrics = {
            "total_tests": len(results_df),
            "avg_sql_similarity": results_df['sql_levenshtein_similarity'].mean(),
            "avg_fix_similarity": results_df['fix_levenshtein_similarity'].mean(),
            "sql_exact_match_rate": results_df['sql_exact_match'].mean(),
            "fix_exact_match_rate": results_df['fix_exact_match'].mean(),
            "predicted_sql_valid_rate": results_df['predicted_sql_valid'].mean(),
            "both_sql_valid_rate": results_df['both_sql_valid'].mean()
        }

        # Добавляем метрики в начало таблицы
        metrics_df = pd.DataFrame([metrics])

        # Сохраняем результаты
        model_name = os.path.basename(model_path).replace('/', '_')
        output_path = f"test_results_{model_type}_{model_name}_{quantization}.xlsx"

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Записываем общие метрики
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            # Записываем детальные результаты
            results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)

        logger.info(f"Результаты сохранены в {output_path}")
        logger.info(f"Общие метрики:")
        logger.info(f"  - Средняя схожесть SQL: {metrics['avg_sql_similarity']:.4f}")
        logger.info(f"  - Средняя схожесть FIX: {metrics['avg_fix_similarity']:.4f}")
        logger.info(f"  - Точные совпадения SQL: {metrics['sql_exact_match_rate']:.2%}")
        logger.info(f"  - Точные совпадения FIX: {metrics['fix_exact_match_rate']:.2%}")
        logger.info(f"  - Валидность предсказанного SQL: {metrics['predicted_sql_valid_rate']:.2%}")

        return output_path

    else:
        raise ValueError("Не удалось обработать ни одного тестового примера")


def main():
    """Основная функция для запуска из командной строки"""
    import argparse

    parser = argparse.ArgumentParser(description="Тестирование LLM модели для исправления SQL")
    parser.add_argument("test_data", help="Путь к файлу с тестовыми данными (Excel)")
    parser.add_argument("model_path", help="Путь к модели или HuggingFace model ID")
    parser.add_argument("--model_type", choices=["base", "finetuned"], default="base",
                        help="Тип модели: base или finetuned")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "fp16"], default="4bit",
                        help="Тип квантования")

    args = parser.parse_args()

    try:
        result_path = test_model(
            test_data_path=args.test_data,
            model_path=args.model_path,
            model_type=args.model_type,
            quantization=args.quantization
        )
        print(f"✅ Результаты тестирования сохранены в {os.path.abspath(result_path)}")

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())