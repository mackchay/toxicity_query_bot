import torch
import pandas as pd
import sqlparse
import os
import logging
from difflib import SequenceMatcher
from tqdm import tqdm
from model_manager import load_model_and_tokenizer, generate_llm_response
from prompt_handler import build_sql_correction_prompt, parse_model_response

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("test_llm")

GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1
}


def levenshtein_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def is_valid_sql(sql: str) -> bool:
    try:
        parsed = sqlparse.parse(sql)
        return len(parsed) > 0 and len(parsed[0].tokens) > 0
    except Exception:
        return False


def parse_llm_response_with_retries(model_name, bad_sql, max_attempts=3, quantization="4bit"):
    """
    Генерирует ответ модели с повторными попытками через model_manager

    Args:
        model_name: Название модели
        bad_sql: Исходный SQL запрос
        max_attempts: Максимальное количество попыток
        quantization: Тип квантования

    Returns:
        tuple: (predicted_sql, predicted_fix)
    """
    for attempt in range(max_attempts):
        try:
            # Строим промпт через prompt_handler
            prompt = build_sql_correction_prompt(bad_sql)

            # Генерируем ответ через model_manager
            response = generate_llm_response(
                prompt=prompt,
                model_name=model_name,
                quantization=quantization,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"]
            )

            # Парсим ответ через prompt_handler
            predicted_sql, predicted_fix = parse_model_response(response)

            if predicted_sql and is_valid_sql(predicted_sql):
                return predicted_sql, predicted_fix

            logging.warning(f"Попытка {attempt + 1} не удалась, повтор...")

        except Exception as e:
            logging.error(f"Ошибка в попытке {attempt + 1}: {str(e)}")

    logging.warning(f"Не удалось получить корректный ответ от модели для SQL после {max_attempts} попыток")
    return None, None


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

    # Предварительно загружаем модель через model_manager для проверки
    try:
        model, tokenizer = load_model_and_tokenizer(model_path, quantization)
        logger.info("Модель успешно загружена через model_manager")
        # Очищаем память после проверки - model_manager будет загружать по необходимости
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
            # Используем функцию с повторными попытками через model_manager
            predicted_sql, predicted_fix = parse_llm_response_with_retries(
                model_name=model_path,
                bad_sql=bad_sql,
                quantization=quantization
            )

            if not predicted_sql:
                logger.warning(f"Отброшен невалидный ответ в строке {idx}")
                continue

            sql_similarity = levenshtein_similarity(predicted_sql.lower().strip(), expected_sql.lower().strip())
            fix_similarity = levenshtein_similarity(predicted_fix.lower().strip(), expected_fix.lower().strip())

            sql_exact_match = predicted_sql.lower().strip() == expected_sql.lower().strip()
            fix_exact_match = predicted_fix.lower().strip() == expected_fix.lower().strip()

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