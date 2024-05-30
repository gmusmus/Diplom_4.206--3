import pandas as pd
import requests
import time
import re
from enum import Enum
from tqdm.auto import tqdm


class DocumentType(Enum):
    passport_ussr = "01"
    birth_certificate = "03"
    passport_foreign = "10"
    residence_permit = "12"
    residence_permit_temp = "15"
    asylum_certificate_temp = "19"
    passport_russia = "21"
    birth_certificate_foreign = "23"
    residence_permit_foreign = "62"

def clean_spaces(text):
    """Удаление лишних пробелов из строки."""
    # Проверка на NaN и замена на пустую строку перед обработкой
    if pd.isna(text):
        return ''
    text = text.strip()  # Удаление пробелов в начале и конце строки
    text = re.sub(' +', ' ', text)  # Замена множественных пробелов на один
    return text
def check_inn(surname, name, patronymic, birthdate, doctype, docnumber, docdate):
    surname = clean_spaces(surname)
    name = clean_spaces(name)
    patronymic = clean_spaces(patronymic)
    birthdate = clean_spaces(birthdate)
    docnumber = clean_spaces(docnumber)
    docdate = clean_spaces(docdate)

    urlJson = "https://service.nalog.ru/inn-new-proc.json"
    dataDo = {
        "c": "find",
        "captcha": "",
        "captchaToken": "",
        "fam": surname,
        "nam": name,
        "otch": patronymic,
        "bdate": birthdate,
        "doctype": doctype,
        "docno": docnumber,
        "docdt": docdate,
    }

    # Печать URL и данных первого запроса
    print("Отправка первого запроса на URL:", urlJson)
    print("Данные первого запроса:", dataDo)

    response = requests.post(url=urlJson, data=dataDo)
    if response.status_code != 200:
        return {'inn': 'Не найдено из-за ошибки запроса'}

    requestId = response.json().get('requestId', '')
    if not requestId:
        return {'inn': 'Не найдено, requestId отсутствует'}

    urlDo = "https://service.nalog.ru/inn-new-proc.do"
    dataJson = {"c": "get", "requestId": requestId}

    # Пауза для обработки сервером
    time.sleep(1)

    # Печать URL и данных второго запроса
    print("Отправка второго запроса на URL:", urlDo)
    print("Данные второго запроса:", dataJson)

    response = requests.post(url=urlDo, data=dataJson)
    if response.status_code != 200:
        return {'inn': 'Не найдено из-за ошибки запроса'}

    return response.json()


# Загрузка данных из файла
df = pd.read_csv('СПИСОК ВОДИТЕЛЕЙ 10-4-2024.csv', delimiter=';', dtype=str, encoding='utf8')


def extract_sequence(text):
    pattern = r'(\d{2}\s\d{2}\s\d{6})|(\d{4}\s\d{6})'
    matches = re.findall(pattern, text)
    ser1 = ''
    ser2 = ''
    for match in matches:
        if match[0]:
            ser1 = match[0]
            formatted_ser1 = f"{ser1[:2]} {ser1[3:5]} {ser1[6:]}"
            ser2 = re.sub(pattern, '', text, 1).strip()
            return formatted_ser1, ser2
        elif match[1]:
            formatted_ser1 = f"{match[1][:2]} {match[1][2:4]} {match[1][5:]}"
            ser2 = re.sub(pattern, '', text, 1).strip()
            return formatted_ser1, ser2
    return None, text  # Возвращаем None и исходный текст, если совпадений нет


def find_date_and_remainder(text):
    # Регулярное выражение для поиска даты в форматах dd.mm.yyyy и dd mm yyyy
    date_pattern = r'(\d{2}[.\s]\d{2}[.\s]\d{4})'
    match = re.search(date_pattern, text)

    if match:
        # Если дата найдена, извлекаем её
        dat1 = match.group()
        # Преобразуем дату к формату dd.mm.yyyy, если она была найдена с пробелами
        dat1 = dat1.replace(' ', '.')
        # Удаляем найденную дату из текста для получения остатка
        ovd = re.sub(date_pattern, '', text, 1).strip()
    else:
        # Если дата не найдена, возвращаем пустые строки
        dat1 = ''
        ovd = text.strip() if text.strip() else ''

    return dat1, ovd

def split_string(text):
    if not text:  # Проверка на None и пустую строку
        return '', ''

    # Убедимся, что text является строкой
    text = str(text)

    # Первая подстрока содержит первые 5 символов или меньше, если длина text меньше 5
    first_part = text[:5]

    # Вторая подстрока содержит все символы после пятого
    second_part = text[5:]

    return first_part, second_part


df.fillna('', inplace=True)
df['ИД водителя'] = df['ИД водителя'].astype(int)
df.columns = df.columns.str.strip()

split_regex = r'выд[ао]н|дата выдачи:|выда|дат[а ]*выдач[и:]'

df[['Passport_part1', 'Passport_part2']] = df['Паспорт'].str.split(split_regex, n=1, expand=True)
df.columns = df.columns.str.strip().str.replace(' ', '_', regex=True)






df[['DATA_OVD', 'Place_OVD']] = df['Passport_part2'].apply(lambda x: find_date_and_remainder(x) if pd.notnull(x) else ('', '')).apply(pd.Series)

print(df.head())


# Применение функции check_inn к каждой строке датафрейма
inn_new = []
for i, row in tqdm(df.iterrows(), total=df.shape[0]):

    response = check_inn(
        surname=row['Фамилия'],
        name=row['Имя'],
        patronymic=row['Отчество'],
        birthdate=row['День_Рождения'],
        doctype=DocumentType.passport_russia.value,  # Это значение нужно подстраивать под данные
        docnumber=row['Серия_номер_паспорта'],
        docdate=row['DATA_OVD'],
    )

    inn_result = response.get('inn', 'Не найдено')
    inn_new.append(inn_result)

    print(f"Обработано: {row['Фамилия']} {row['Имя']} {row['Отчество']}. Результат поиска ИНН: {inn_result}")

pd.set_option('display.max_columns', None)  # Показывать все столбцы
pd.set_option('display.max_colwidth', None)  # Показывать полное содержимое ячеек

# Добавление нового столбца с ИНН в датафрейм
df['inn_new'] = inn_new

# Сохранение результатов
df.to_csv('passport_INN.csv', index=False, sep=';', encoding='utf-8')
df.to_excel('passport_INN.xlsx', index=False, engine='openpyxl')

print("Обработка завершена и данные сохранены.")
