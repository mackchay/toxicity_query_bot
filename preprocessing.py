import pandas as pd
import os
import chardet
import re


def detect_encoding(file_path, sample_size=10000):
    with open(file_path, 'rb') as f:
        rawdata = f.read(sample_size)
    result = chardet.detect(rawdata)
    print(f"Определённая кодировка: {result['encoding']}")
    return result['encoding']


def classify_reason(reason: str) -> str:
    reason = reason.lower()

    # Специфичные паттерны
    if 'value cannot be cast to date' in reason:
        return 'CastError'

    if 'query text length' in reason and 'exceeds the maximum length' in reason:
        return 'QueryTooLargeError'

    if 'destination table' in reason and 'already exists' in reason:
        return 'ObjectAlreadyExistsError'

    if 'function' in reason and 'not registered' in reason:
        return 'FunctionError'

    if 'table' in reason and 'does not exist' in reason:
        return 'MissingObjectError'

    if 'column' in reason and 'cannot be resolved' in reason:
        return 'MissingColumnError'

    if 'schema' in reason and 'does not exist' in reason:
        return 'MissingSchemaError'

    if 'encountered too many errors talking to a worker node' in reason:
        return 'TransientError'

    if 'pvlos' in reason and 'java.net.sockettimeoutexception' in reason:
        return 'TransientError'

    if 'failed connecting to hive metastore' in reason:
        return 'MetastoreError'

    if 'gss authentication failed' in reason:
        return 'AuthError'

    if 'the optimizer exhausted the time limit' in reason:
        return 'OptimizerError'

    if 'logical expression term must evaluate to a boolean' in reason:
        return 'TypeError'

    if 'unexpected parameters' in reason and 'for function' in reason:
        return 'FunctionSignatureError'

    # Общие категории
    if 'too many connections' in reason:
        return 'TooManyConnectionsError'

    if any(x in reason for x in
           ['mismatched input', 'syntax', 'unexpected', 'missing', 'expecting', 'invalid identifier']):
        return 'SyntaxError'

    if any(x in reason for x in
           ['cannot cast', 'value cannot be cast', 'incompatible types', 'must be', 'cannot resolve', 'unknown type']):
        return 'TypeError'

    if any(x in reason for x in ['array subscript', 'index out of bounds']):
        return 'IndexError'

    if any(x in reason for x in ['ambiguous', 'multiple columns', 'column name']):
        return 'AmbiguityError'

    if any(x in reason for x in
           ['timeout', 'query exceeded', 'memory limit', 'exceeded maximum time', 'exceeded per-node memory',
            'exceeded distributed user memory']):
        return 'ResourceError'

    if any(x in reason for x in ['access denied', 'permission', 'not authorized']):
        return 'PermissionError'

    if any(x in reason for x in ['does not exist', 'not found', 'cannot be resolved']):
        return 'MissingObjectError'

    if any(x in reason for x in ['cancelled', 'canceled', 'query was canceled']):
        return 'CanceledError'

    return 'Other'


def is_valid_sql(sql_str):
    """Проверяет, содержит ли строка допустимый SQL-запрос"""
    if not isinstance(sql_str, str):
        return False

    sql_str = sql_str.strip().lower()

    # Проверка на отсутствие SQL
    if len(sql_str) == 0:
        return False

    # Проверка на специальные значения
    if sql_str in ['nan', 'null', 'none', 'n/a', 'na', 'нет', 'отсутствует', 'empty']:
        return False

    # Проверка минимальной длины SQL
    if len(sql_str) < 10:  # Минимальная длина для SQL-запроса
        return False

    # Проверка на наличие SQL-ключевых слов
    sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'alter', 'from', 'where', 'join']
    if any(keyword in sql_str for keyword in sql_keywords):
        return True

    return False


def preprocess(input_xlsx, output_path):
    if not os.path.exists(input_xlsx):
        print(f'Error: Input file "{input_xlsx}" not found.')
        return

    detect_encoding(input_xlsx)

    try:
        df = pd.read_excel(input_xlsx)
    except Exception as e:
        print(f'Ошибка при чтении Excel: {e}')
        return

    print('=== Диагностика после чтения Excel ===')
    print('Все имена столбцов:', list(df.columns))
    print('Типы данных по столбцам:')
    print(df.dtypes)
    print('Количество пропусков по столбцам:')
    print(df.isnull().sum())
    print('Первые 5 строк:')
    print(df.head(5))

    # Создаем копию исходных данных для отладки
    df_original = df.copy()

    # Гибкий поиск столбцов
    reason_col = None
    bad_sql_col = None

    possible_reason_cols = ['REASON', 'REASON_1', 'REASON_2', 'REASON_DETAIL']
    possible_sql_cols = ['BAD_SQL', 'OBJECT_NAME_1', 'SQL_TEXT', 'QUERY_TEXT', 'QUERY']

    for col in possible_reason_cols:
        if col in df.columns:
            reason_col = col
            break

    for col in possible_sql_cols:
        if col in df.columns:
            bad_sql_col = col
            break

    if reason_col is None or bad_sql_col is None:
        print('Не найдены нужные столбцы для REASON и BAD_SQL. Доступные столбцы:')
        print(list(df.columns))
        return

    df = df.rename(columns={
        reason_col: 'REASON',
        bad_sql_col: 'BAD_SQL'
    })

    # Основные столбцы
    df = df[['STATUS', 'REASON', 'BAD_SQL']].copy()

    # Преобразование в строки и очистка
    df['REASON'] = df['REASON'].astype(str).str.replace('\n', ' ', regex=False)
    df['BAD_SQL'] = df['BAD_SQL'].astype(str).str.replace('\n', ' ', regex=False)

    # Фильтрация: пропускаем строки без валидного SQL
    print(f"\nВсего строк до фильтрации: {len(df)}")
    valid_sql_mask = df['BAD_SQL'].apply(is_valid_sql)
    df = df[valid_sql_mask]
    print(f"Строк после фильтрации по наличию SQL: {len(df)}")

    # Фильтр: исключаем строки с CanceledError
    canceled_mask = df['REASON'].str.contains('cancell?ed|query was canceled', case=False, regex=True)
    df = df[~canceled_mask]

    # Сохраняем строки с успешным статусом
    success_mask = df['STATUS'].str.lower().isin(['success', 'succes', 'успех'])
    success_df = df[success_mask].copy()
    success_df['ERROR_CLASS'] = 'Success'
    success_df['IS_FIXABLE'] = True

    # Обрабатываем ошибочные строки
    fail_df = df[~success_mask].copy()
    fail_df = fail_df.dropna(subset=['REASON'])

    # Классифицируем ошибки
    fail_df['ERROR_CLASS'] = fail_df['REASON'].apply(classify_reason)

    # Определяем исправимые ошибки
    fixable_errors = [
        'SyntaxError', 'TypeError', 'FunctionError', 'IndexError',
        'AmbiguityError', 'ResourceError', 'TooManyConnectionsError',
        'QueryTooLargeError', 'ObjectAlreadyExistsError', 'FunctionSignatureError',
        'CastError', 'OptimizerError'
    ]

    non_fixable_errors = [
        'PermissionError', 'MissingObjectError', 'CanceledError',
        'TransientError', 'MetastoreError', 'AuthError', 'MissingColumnError',
        'MissingSchemaError'
    ]

    # Помечаем известные исправимые ошибки
    fail_df['IS_FIXABLE'] = fail_df['ERROR_CLASS'].isin(fixable_errors)

    # Помечаем известные неисправимые ошибки
    fail_df.loc[fail_df['ERROR_CLASS'].isin(non_fixable_errors), 'IS_FIXABLE'] = False

    # Для новых/неизвестных ошибок (Other) по умолчанию считаем неисправимыми
    fail_df.loc[fail_df['ERROR_CLASS'] == 'Other', 'IS_FIXABLE'] = False

    # Объединяем успешные и ошибочные строки
    df_result = pd.concat([success_df, fail_df], ignore_index=True)

    # Удаляем CanceledError
    df_result = df_result[df_result['ERROR_CLASS'] != 'CanceledError']

    # Сохранение результата
    if output_path.lower().endswith('.xlsx'):
        df_result.to_excel(output_path, index=False)
        print(f'\nФайл успешно сохранён (Excel): {os.path.abspath(output_path)}')
    else:
        df_result.to_csv(output_path, index=False, encoding='utf-8')
        print(f'\nФайл успешно сохранён (CSV): {os.path.abspath(output_path)}')

    print('\nЗаголовки столбцов:', df_result.columns.tolist())
    print('Первые 5 строк:')
    print(df_result.head(5))

    # Статистика
    print('\n=== Статистика по типам ошибок ===')
    error_stats = df_result['ERROR_CLASS'].value_counts()
    print(error_stats)

    print('\n=== Статистика по исправимости ===')
    fixable_stats = df_result['IS_FIXABLE'].value_counts()
    print(fixable_stats)

    # Сохраняем лог
    log_path = os.path.splitext(output_path)[0] + '_log.txt'
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("=== Статистика обработки ===\n")
        log_file.write(f"Всего строк в исходном файле: {len(df_original)}\n")
        log_file.write(f"Строк после фильтрации (с SQL): {len(df)}\n")
        log_file.write(f"Строк в результате: {len(df_result)}\n")
        log_file.write(f"Успешных: {len(success_df)}\n")
        log_file.write(f"Ошибочных: {len(fail_df)}\n")
        log_file.write("\nРаспределение ошибок:\n")
        log_file.write(error_stats.to_string())
        log_file.write("\n\nИсправимость ошибок:\n")
        log_file.write(fixable_stats.to_string())

        # Примеры пропущенных строк без SQL
        missing_sql = df_original[~valid_sql_mask]
        if not missing_sql.empty:
            log_file.write("\n\n=== Пропущенные строки (без SQL) ===\n")
            for _, row in missing_sql.head(20).iterrows():
                log_file.write(f"STATUS: {row.get('STATUS', '')}\n")
                log_file.write(f"REASON: {row.get(reason_col, '')[:200]}\n")
                log_file.write(f"SQL: {row.get(bad_sql_col, '')[:200]}\n")
                log_file.write("-" * 50 + "\n")
            log_file.write(f"\nВсего пропущено строк без SQL: {len(missing_sql)}")

    print(f'\nЛог обработки сохранён: {os.path.abspath(log_path)}')
    print(f'Пропущено строк без SQL: {len(df_original) - len(df)}')


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print('Usage: python preprocessing.py input.xlsx output_path(.csv|.xlsx)')
    else:
        preprocess(sys.argv[1], sys.argv[2])