import pandas as pd
import requests
from tqdm.auto import tqdm



# Загрузка данных из файла
df = pd.read_csv('СПИСОК ВОДИТЕЛЕЙ 10-4-2024.csv', delimiter=',', dtype=str)

# Создание нового столбца для сохранения найденных ИНН
df['INN_find'] = pd.NA
df = df.fillna('')

# Преобразование столбца 'Nomer' в строковый тип
df['number_passport'] = df['number_passport'].astype(str)

# Добавление нулей слева, чтобы обеспечить минимальную длину строки в 6 символов
df['number_passport'] = df['number_passport'].str.zfill(6)

print(df.columns)
print(df.astype)
# Сохранение обновленных данных

# Обработка строки с разным количеством компонентов
def process_name(name):
    parts = name.split(' ')
    if len(parts) >= 3:
        return parts[0], parts[1], ' '.join(parts[2:])
    elif len(parts) == 2:
        return parts[0], parts[1], None
    elif len(parts) == 1:
        return parts[0], None, None
    else:
        return None, None, None

# Применение функции обработки к каждой строке
df['Фамилия'], df['Имя'], df['Отчество'] = zip(*df['DName'].apply(process_name))


# Сохранение результатов
df.to_csv('passport_data5.csv', index=False)
df.to_excel('passport_data5.xlsx', index=False, engine='openpyxl')

print("Обработка завершена и данные сохранены.")

