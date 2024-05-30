import sqlite3
import os
from fuzzywuzzy import process
import pandas as pd
from tqdm import tqdm
import re

def extract_first_three_words(filename):
    # Разделяем имя файла по пробелам и знакам пунктуации, оставляем первые три слова
    words = re.split(r'[\s_,?]+', filename)[:3]
    return ' '.join(words)

a = 'E:\\CityBase\\sqlite.db'
b = 'D:\\данные\\06 Паспорт 1-я страница'

conn = sqlite3.connect(a)
c = conn.cursor()

c.execute("""
SELECT DISTINCT 
    d.id as driver_id,
    c.id as car_id,
    e.id as ekipag_id,
    c.CarID,
    substr(c.CarID, 1, 3) || ' ' || 
    substr(d.DName, 1, instr(d.DName || ' ', ' ') - 1) || ' ' || 
    substr(d.DName, instr(d.DName || ' ', ' ') + 1, instr(substr(d.DName, instr(d.DName || ' ', ' ') + 1) || ' ', ' ') - 1) AS Name_File,
    d.Dname,
    d.DDate, 
    d.DPass,
    d.DHome,
    d.Phone,
    d.Dreg ,
    d.SNILS,
    d.INN,
    d.Dreg,
    d.DLicense ,
    d.DBegLicDate ,
    d.DLicDate ,
    d.Samozan,
    d.Sudim,
    d.Mintrans,
    d.DPhoto_filename
    
FROM Ekipag e
JOIN driver d ON e.id_Driver = d.id
JOIN CarInfo c ON c.id = e.id_CarInfo 
ORDER BY d.DName
""")
rows = c.fetchall()
conn.close()


df = pd.DataFrame(rows, columns=[desc[0] for desc in c.description])

print(df.dtypes)
print(df.columns)

df['DDate'] = pd.to_datetime(df['DDate']).dt.strftime('%d.%m.%Y')
df['DBegLicDate'] = pd.to_datetime(df['DBegLicDate']).dt.strftime('%d.%m.%Y')
df['DLicDate'] = pd.to_datetime(df['DLicDate']).dt.strftime('%d.%m.%Y')

file_names = os.listdir(b)
names = [extract_first_three_words(name) for name in file_names]

# Множество для отслеживания уже использованных файлов
used_files = set()
matched_files = []

for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Обработка"):
    name_file = row['Name_File']
    match = process.extractOne(name_file, names, score_cutoff=96)

    if match and names.index(match[0]) not in used_files:
        matched_file = file_names[names.index(match[0])]
        used_files.add(names.index(match[0]))  # Добавляем индекс использованного файла
        matched_files.append(matched_file)
    else:
        # Повторный поиск без первого слова, если не найдено точное соответствие
        name_file_parts = name_file.split()
        if len(name_file_parts) > 1:
            name_file_modified = ' '.join(name_file_parts[1:])
            match = process.extractOne(name_file_modified, names, score_cutoff=96)
            if match and names.index(match[0]) not in used_files:
                matched_file = file_names[names.index(match[0])]
                used_files.add(names.index(match[0]))
                matched_files.append(matched_file)
            else:
                matched_files.append(None)
        else:
            matched_files.append(None)

df['Matched_File'] = matched_files
# Фильтрация DataFrame, чтобы оставить только те строки, где 'Matched_File' не пустой
filtered_df = df[df['Matched_File'] != '']

# Сохранение отфильтрованного DataFrame в файлы CSV и Excel
filtered_df.to_csv('passport_data.csv', index=False)
filtered_df.to_excel('passport_data.xlsx', index=False, engine='openpyxl')

print("Скрипт завершен, результаты сохранены в 'passport_vs_BY.csv' и 'passport_vs_BY.xlsx'")
