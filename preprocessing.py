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

    # Сначала проверяем специфичные паттерны
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

    if 'pvlos' in reason and 'java.net.SocketTimeoutException' in reason:
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

    # Затем общие категории
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

    # Попробуем разные варианты имен столбцов
    reason_col = None
    bad_sql_col = None

    possible_reason_cols = ['REASON', 'REASON_1', 'REASON_2', 'REASON_DETAIL']
    possible_sql_cols = ['BAD_SQL', 'OBJECT_NAME_1', 'SQL_TEXT', 'QUERY_TEXT']

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

    df = df[['STATUS', 'REASON', 'BAD_SQL']].copy()
    df['REASON'] = df['REASON'].astype(str).str.replace('\n', ' ', regex=False)
    df['BAD_SQL'] = df['BAD_SQL'].astype(str).str.replace('\n', ' ', regex=False)

    # Фильтр: исключаем строки с CanceledError
    canceled_mask = df['REASON'].str.contains('cancell?ed|query was canceled', case=False, regex=True)
    df = df[~canceled_mask]

    # Сохраняем строки с успешным статусом независимо от REASON
    success_mask = df['STATUS'].str.lower().isin(['success', 'succes', 'успех'])
    success_df = df[success_mask].copy()
    success_df['ERROR_CLASS'] = 'Success'
    success_df['IS_FIXABLE'] = True

    # Обрабатываем остальные строки
    fail_df = df[~success_mask].copy()
    fail_df = fail_df.dropna(subset=['REASON', 'BAD_SQL'])
    fail_df = fail_df[fail_df['BAD_SQL'].astype(str).str.strip().astype(bool)]
    fail_df = fail_df[~fail_df['BAD_SQL'].str.lower().isin(['nan', 'null', 'none'])]

    # Классифицируем все ошибки
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

    # Объединяем успешные и все неуспешные строки
    df_result = pd.concat([success_df, fail_df], ignore_index=True)

    # Дополнительная фильтрация: удаляем CanceledError, если они прошли через классификатор
    df_result = df_result[df_result['ERROR_CLASS'] != 'CanceledError']

    # Сохранение результата
    if output_path.lower().endswith('.xlsx'):
        df_result.to_excel(output_path, index=False)
        print(f'Файл успешно сохранён (Excel): {os.path.abspath(output_path)}')
    else:
        df_result.to_csv(output_path, index=False, encoding='utf-8')
        print(f'Файл успешно сохранён (CSV): {os.path.abspath(output_path)}')

    print('Заголовки столбцов:', df_result.columns.tolist())
    print('Первые 5 строк:')
    print(df_result.head(5))

    # Статистика по типам ошибок
    print('\n=== Статистика по типам ошибок ===')
    error_stats = df_result['ERROR_CLASS'].value_counts()
    print(error_stats)

    print('\n=== Статистика по исправимости ===')
    fixable_stats = df_result['IS_FIXABLE'].value_counts()
    print(fixable_stats)

    # Сохраняем лог обработки
    log_path = os.path.splitext(output_path)[0] + '_log.txt'
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("=== Статистика обработки ===\n")
        log_file.write(f"Всего строк: {len(df_result)}\n")
        log_file.write(f"Успешных: {len(success_df)}\n")
        log_file.write(f"Ошибочных: {len(fail_df)}\n")
        log_file.write("\nРаспределение ошибок:\n")
        log_file.write(error_stats.to_string())
        log_file.write("\n\nИсправимость ошибок:\n")
        log_file.write(fixable_stats.to_string())

        # Сохраняем примеры ошибок для каждого класса
        log_file.write("\n\n=== Примеры ошибок ===\n")
        for error_class in df_result['ERROR_CLASS'].unique():
            examples = df_result[df_result['ERROR_CLASS'] == error_class].head(3)
            log_file.write(
                f"\n--- {error_class} (всего: {len(df_result[df_result['ERROR_CLASS'] == error_class])}) ---\n")
            for _, row in examples.iterrows():
                log_file.write(f"REASON: {row['REASON']}\n")
                log_file.write(f"SQL: {row['BAD_SQL'][:200]}...\n")
                log_file.write("-" * 50 + "\n")

    print(f'Лог обработки сохранён: {os.path.abspath(log_path)}')


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print('Usage: python preprocessing.py input.xlsx output_path(.csv|.xlsx)')
    else:
        preprocess(sys.argv[1], sys.argv[2])