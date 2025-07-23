import pandas as pd
import os
import chardet


def detect_encoding(file_path, sample_size=10000):
    with open(file_path, 'rb') as f:
        rawdata = f.read(sample_size)
    result = chardet.detect(rawdata)
    print(f"Определённая кодировка: {result['encoding']}")
    return result['encoding']


def classify_reason(reason: str) -> str:
    reason = reason.lower()

    if 'too many connections' in reason:
        return 'TooManyConnectionsError'

    if any(x in reason for x in ['mismatched input', 'syntax', 'unexpected', 'missing', 'expecting', 'invalid identifier']):
        return 'SyntaxError'

    if any(x in reason for x in ['cannot cast', 'value cannot be cast', 'incompatible types', 'must be', 'cannot resolve']):
        return 'TypeError'

    if any(x in reason for x in ['function', 'not registered', 'invalid function', 'unknown function']):
        return 'FunctionError'

    if any(x in reason for x in ['array subscript', 'index out of bounds']):
        return 'IndexError'

    if any(x in reason for x in ['ambiguous', 'multiple columns', 'column name']):
        return 'AmbiguityError'

    if any(x in reason for x in ['timeout', 'query exceeded', 'memory limit']):
        return 'ResourceError'

    if any(x in reason for x in ['access denied', 'permission', 'not authorized']):
        return 'PermissionError'

    if any(x in reason for x in ['does not exist', 'not found', 'cannot be resolved']):
        return 'MissingObjectError'

    if any(x in reason for x in ['cancelled', 'query was canceled']):
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

    df = df.rename(columns={
        'REASON_1': 'REASON',
        'OBJECT_NAME_1': 'BAD_SQL'
    })

    if 'REASON' not in df.columns or 'BAD_SQL' not in df.columns:
        print('Не найдены нужные столбцы "REASON" и "BAD_SQL".')
        return

    df = df[['STATUS', 'REASON', 'BAD_SQL']].copy()
    df['REASON'] = df['REASON'].astype(str).str.replace('\n', ' ', regex=False)
    df['BAD_SQL'] = df['BAD_SQL'].astype(str).str.replace('\n', ' ', regex=False)

    # Сохраняем строки с успешным статусом независимо от REASON
    succes_df = df[df['STATUS'] == 'SUCCES'].copy()

    # Фильтруем остальные строки по прежней логике
    fail_df = df[df['STATUS'] != 'SUCCES'].copy()
    fail_df = fail_df.dropna(subset=['REASON', 'BAD_SQL'])
    fail_df = fail_df[fail_df['BAD_SQL'].astype(str).str.strip().astype(bool)]
    fail_df = fail_df[~fail_df['BAD_SQL'].str.lower().isin(['nan'])]
    fail_df = fail_df[~fail_df['REASON'].str.contains('Query was canceled', na=False)]

    impossible_patterns = [
        r'does not exist',
        r'cannot be resolved',
        r'access denied',
        r'hive'
    ]
    for pat in impossible_patterns:
        fail_df = fail_df[~fail_df['REASON'].str.lower().str.contains(pat, na=False)]

    fail_df['ERROR_CLASS'] = fail_df['REASON'].apply(classify_reason)
    fixable_errors = ['SyntaxError', 'TypeError', 'FunctionError', 'IndexError', 'AmbiguityError', 'ResourceError', 'TooManyConnectionsError']
    fail_df = fail_df[fail_df['ERROR_CLASS'].isin(fixable_errors)]

    # Объединяем успешные и отфильтрованные неуспешные строки
    df_result = pd.concat([succes_df, fail_df], ignore_index=True)

    if output_path.lower().endswith('.xlsx'):
        df_result.to_excel(output_path, index=False)
        print(f'Файл успешно сохранён (Excel): {os.path.abspath(output_path)}')
    else:
        df_result.to_csv(output_path, index=False, encoding='utf-8')
        print(f'Файл успешно сохранён (CSV): {os.path.abspath(output_path)}')

    print('Заголовки столбцов:', df_result.columns.tolist())
    print('Первые 5 строк:')
    print(df_result.head(5))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python preprocessing.py input.xlsx output_path(.csv|.xlsx)')
    else:
        preprocess(sys.argv[1], sys.argv[2])
