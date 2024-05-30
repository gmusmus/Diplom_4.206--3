import os
import pandas as pd

def analyze_folder(folder_path):
    rows = []  # Создаем пустой список для хранения данных каждой строки

    for root, dirs, files in os.walk(folder_path):
        total_size = sum(os.path.getsize(os.path.join(root, name)) for name in files)
        jpg_files = sum(1 for name in files if name.lower().endswith('.jpg'))
        pdf_files = sum(1 for name in files if name.lower().endswith('.pdf'))
        jpg_size = sum(os.path.getsize(os.path.join(root, name)) for name in files if name.lower().endswith('.jpg'))
        pdf_size = sum(os.path.getsize(os.path.join(root, name)) for name in files if name.lower().endswith('.pdf'))

        # Сохраняем данные текущей итерации в список как словарь
        rows.append({'Папка': root,
                     'Общий объем (байты)': total_size,
                     'Количество файлов': len(files),
                     'Файлы JPG': jpg_files,
                     'Объем JPG (байты)': jpg_size,
                     'Файлы PDF': pdf_files,
                     'Объем PDF (байты)': pdf_size})

    # Преобразуем список словарей в DataFrame
    df = pd.DataFrame(rows)

    return df

folder_path = r'D:\данные'  #  путь к  папке
df = analyze_folder(folder_path)
# Настройка отображения для вывода большого количества строк и столбцов
pd.set_option('display.max_rows', None)  # Показывать все строки
pd.set_option('display.max_columns', None)  # Показывать все столбцы
pd.set_option('display.width', None)  # Автоматически подбирать ширину столбцов
pd.set_option('display.max_colwidth', None)  # Показывать полный текст в ячейках

print(df)

df.to_csv('struct_dir.csv', index=False)
df.to_excel('struct_dir.xlsx', index=False, engine='openpyxl')