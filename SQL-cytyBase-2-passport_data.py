import sqlite3
import os
from fuzzywuzzy import process
import pandas as pd
from tqdm import tqdm
import re

import re
from datetime import datetime

# функция разбора поля с данными паспорта
def parse_passport_string(s):
    s = str(s) if s is not None else ''
    # Убираем лишние пробелы
    s = re.sub(' +', ' ', s)

    # Находим серию
    seria_match = re.search(r'\b\d{2} \d{2}\b', s)
    seria = seria_match.group(0) if seria_match else None

    # Находим номер
    nomer_match = re.search(r'(?<=\b\d{2} \d{2} )\d{6}', s)
    nomer = nomer_match.group(0) if nomer_match else None

    # Находим дату выдачи
    data_vidachi_match = re.search(r'выдан (\d{2}\.\d{2}\.\d{4})', s, re.IGNORECASE)
    if data_vidachi_match:
        data_vidachi = datetime.strptime(data_vidachi_match.group(1), '%d.%m.%Y').date()
    else:
        data_vidachi = None

    # Находим наименование подразделения
    if data_vidachi_match:
        name_podrazdel = s[data_vidachi_match.end():].strip()
    else:
        name_podrazdel = None

    return {
        'seria': seria,
        'nomer': nomer,
        'data_vidachi': data_vidachi,
        'name_podrazdel': name_podrazdel
    }





# Функция parse_passport_string уже определена выше

def read_csv_to_df(filepath):
    return pd.read_csv(filepath, delimiter=';')


def process_dataframe(df):
    # Добавление новых столбцов с инициализацией пустыми значениями
    df['Seria'] = None
    df['Nomer'] = None
    df['DataVidachi'] = None
    df['NamePodrazdel'] = None

    # Обработка каждой строки DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing", leave=False):
        parsed_data = parse_passport_string(row['DPass'])
        # Явное преобразование в строковый тип данных для серии, номера и наименования подразделения
        df.at[index, 'Seria'] = str(parsed_data['seria']) if parsed_data['seria'] is not None else None
        df.at[index, 'Nomer'] = str(parsed_data['nomer']) if parsed_data['nomer'] is not None else None
        df.at[index, 'NamePodrazdel'] = str(parsed_data['name_podrazdel']) if parsed_data[
                                                                                  'name_podrazdel'] is not None else None

        # Преобразование даты выдачи в строку в формате dd.mm.yyyy
        if parsed_data['data_vidachi'] is not None:
            df.at[index, 'DataVidachi'] = parsed_data['data_vidachi'].strftime('%d.%m.%Y')
        else:
            df.at[index, 'DataVidachi'] = None

        print(
            f"Processed DPass: {row['DPass']} -> Seria: {df.at[index, 'Seria']}, Nomer: {df.at[index, 'Nomer']}, DataVidachi: {df.at[index, 'DataVidachi']}, NamePodrazdel: {df.at[index, 'NamePodrazdel']}")

    return df


# Загрузка данных из CSV файла в DataFrame
filepath = 'passport_data.csv'
df = read_csv_to_df(filepath)

# Обработка DataFrame
processed_df = process_dataframe(df)

# Сохранение обработанного DataFrame в новый файл CSV и Excel
processed_df.to_csv('passport_data2.csv', index=False)
processed_df.to_excel('passport_data2.xlsx', index=False, engine='openpyxl')

print("Data processing complete and saved to 'updated_passport_data.csv' and 'updated_passport_data.xlsx'")
