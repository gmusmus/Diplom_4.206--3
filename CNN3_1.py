import pandas as pd
import os
import torch
import dlib
import cv2
import pytesseract
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import re



# Путь к файлам с предобученными моделями dlib
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
RECOGNITION_MODEL_PATH = 'dlib_face_recognition_resnet_model_v1.dat'

# Инициализация dlib детектора и загрузка моделей
sp = dlib.shape_predictor(PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(RECOGNITION_MODEL_PATH)
detector = dlib.get_frontal_face_detector()

#откроем файл с данными и загрузим только определенные столбцы
file_data = r'D:\данныеМодели\данные для обучения.xlsx'
dir_jpg = r'D:\данные\06 Паспорт 1-я страница'

# уберём двойные пробелы в названии файла
for name_file in os.listdir(dir_jpg):
    if '  ' in name_file:
        new_name = name_file.replace('  ', ' ')
        os.rename(os.path.join(dir_jpg, name_file), os.path.join(dir_jpg, new_name))
        print(f'Переменованный файл, убраны доп пробелы "{i}" to "{new_name}"')

#загрузим датафрейм
df = pd.read_excel(file_data, dtype=str)
df1 = df[['Фамилия',
                       'Имя',
                       'Отчество',
                       'День_Рождения',
                       'seria',
                       'number',
                       'Код_Подразделения',
                       'data_vidachi',
                       'Podrazdelen',
                       'filename']].copy()
df=df1
print('Кол-во строк в датафрейм: ',df.shape[0])

# Удалим строки, где данные нет или не соответствую типу и размерности
df = df[pd.notna(df['Код_Подразделения']) & (df['Код_Подразделения'] != '') & (df['Код_Подразделения'].str.len() == 7)]
df = df[pd.notna(df['Podrazdelen']) & (df['Podrazdelen'] != '')& (df['Podrazdelen'].str.len() > 10)]
df = df[pd.notna(df['День_Рождения']) & (df['День_Рождения'] != '') & (df['День_Рождения'].str.len() == 10)]
df = df[pd.notna(df['seria']) & (df['seria'] != '') & (df['seria'].str.len() == 5)]
df = df[pd.notna(df['number']) & (df['number'] != '') & (df['number'].str.len() == 6)]
df = df[pd.notna(df['Фамилия']) & (df['Фамилия'] != '')]
df = df[pd.notna(df['Имя']) & (df['Имя'] != '')]
#уберём дубликаты
df.drop_duplicates(subset=['Фамилия', 'Имя', 'Отчество', 'День_Рождения'], keep='first', inplace=True)
df.drop_duplicates(subset=['filename'], keep='first', inplace=True)

#посмотрим датафрейм
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.sample(20))
print('Кол-во строк в датафрейм: ',df.shape[0])


########################################################################################################################
########################################################################################################################
########################################################################################################################
def rotate_image_to_rectangle(image_pil):
    image = np.array(image_pil)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)  # Увеличили размытие для снижения детализации
    edges = cv2.Canny(image_blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 0.2 * image.shape[0] * image.shape[1]  # 20% от площади изображения
    rects = [cv2.minAreaRect(cnt) for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not rects:
        return image_pil  # Возвращаем исходное изображение, если подходящие контуры не найдены

    # Определение углов для поворота, которые максимизируют горизонтальное расположение прямоугольников
    angles = [rect[2] if rect[2] <= -45 else rect[2] - 90 for rect in rects]  # Корректировка углов

    # Выбираем угол, который чаще всего встречается, как оптимальный для поворота
    optimal_angle = max(set(angles), key=angles.count)

    print('Оптимальный угол поворота:', optimal_angle)

    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, optimal_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(rotated)


# Путь к изображению для обработки
sample_row = df.sample(1).iloc[0]
print(sample_row)
image_filename = sample_row['filename']
image_path = os.path.join(dir_jpg, image_filename)

# Загрузка изображения через Pillow и конвертация в формат, который OpenCV может обработать
image = Image.open(image_path)
image.show()

aligned_face = rotate_image_to_rectangle(image)
if aligned_face:
    aligned_face.show()
else:
    print("Лицо не обнаружено.")

