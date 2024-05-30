import pandas as pd
from fuzzywuzzy import process, fuzz
from tqdm.auto import tqdm

# Загрузка данных из файлов
df = pd.read_csv('processed_passports.csv', delimiter=';', dtype=str, encoding='cp1251')
codes_df = pd.read_csv('коды паспартов.csv', delimiter=';', quotechar='"')
df = df.fillna('')
codes_df = codes_df.fillna('')

# Преобразование INN из чисел в строки
df['INN'] = df['INN'].astype(str).replace('\.0', '', regex=True)

# Функция для поиска совпадений с учетом постепенного снижения порога совпадения
def find_best_match(name, choices):
    original_name = name.upper()  # Приведение к верхнему регистру для сравнения
    for threshold in range(100, 88, -1):  # Поиск с 100% до 90% с шагом в 1%
        match = process.extractOne(original_name, choices, scorer=fuzz.WRatio, score_cutoff=threshold)
        if match:
            return match[0], match[1]
    return original_name, None  # Возврат исходного имени в верхнем регистре и None для match_score, если совпадение не найдено


# Сопоставление NAME с CODE
name_to_code = {row['NAME'].upper(): row['CODE'] for index, row in codes_df.iterrows()}


# Применение функции к каждой строке с отображением прогресса и выводом результатов
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
    matched_name, match_score = find_best_match(row['issuing_authority'], codes_df['NAME'].str.upper())
    matched_code = name_to_code.get(matched_name, None)
    df.at[index, 'Matched_NAME'] = matched_name
    df.at[index, 'Matched_CODE'] = matched_code
    df.at[index, 'Match_Score'] = match_score
    if matched_code:  # Проверка, что код действительно найден
        print(f"issuing_authority: {row['issuing_authority']} -> Matched_NAME: {matched_name}, Matched_CODE: {matched_code}")


# Сохранение результатов
df.to_csv('passport_data4.csv', index=False)
df.to_excel('passport_data4.xlsx', index=False, engine='openpyxl')

print("Обработка завершена и данные сохранены.")
