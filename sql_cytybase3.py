import pandas as pd
from dadata import Dadata

token = "df2b5d4bc85360fc2862706c5f6483e116caeaf7"
dadata = Dadata(token)

# Загружаем данные из файла
df = pd.read_csv('2коды.csv', delimiter=';', encoding='windows-1251')


# Функция для обновления информации подразделения через API DaData
def update_department_info(row):
    # Пример использования поля кода подразделения, если оно доступно
    query = row['code'] if 'code' in row else row['name']

    # Совершаем запрос к API
    result = dadata.suggest("fms_unit", query)

    # Если API возвращает данные, обновляем информацию
    if result:
        row['name'] = result[0]['value']  # Обновленное полное название подразделения
        row['code'] = result[0]['data']['code']  # Код подразделения
    return row


# Применяем функцию обновления к каждой строке DataFrame
updated_df = df.apply(update_department_info, axis=1)

# Сохраняем обновленные данные обратно в файл CSV
updated_df.to_csv('2коды_updated.csv', index=False)

print("Обновление завершено и данные сохранены в '2коды_updated.csv'.")