import csv
import sys
from difflib import SequenceMatcher

# === Замените эту функцию на вызов вашей LLM ===
def llm_fix_sql(bad_sql):
    # Здесь должен быть вызов вашей модели, например:
    # return my_llm.fix_sql(bad_sql)
    return bad_sql  # Заглушка: возвращает входной запрос


def levenshtein(a, b):
    # Быстрая оценка через SequenceMatcher (не совсем Levenshtein, но близко)
    matcher = SequenceMatcher(None, a, b)
    return int(max(len(a), len(b)) * (1 - matcher.ratio()))


def main(csv_path):
    total = 0
    correct = 0
    total_lev = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bad_sql = row['bad_sql'].strip()
            good_sql = row['good_sql'].strip()
            pred_sql = llm_fix_sql(bad_sql).strip()
            if pred_sql == good_sql:
                correct += 1
            lev = levenshtein(pred_sql, good_sql)
            total_lev += lev
            total += 1
            print(f'BAD: {bad_sql}\nPRED: {pred_sql}\nGOOD: {good_sql}\nLEV: {lev}\n')
    print(f'Total: {total}')
    print(f'Accuracy: {correct/total:.3f}')
    print(f'Avg Levenshtein: {total_lev/total:.2f}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python test_llm.py dataset.csv')
        sys.exit(1)
    main(sys.argv[1])
