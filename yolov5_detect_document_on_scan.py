import os
import torch
from PIL import Image
from tqdm import tqdm

# Пути
dir_photo = r'D:\данные\06 Паспорт 1-я страница'
dir_outphoto = r'D:\данные\06 Паспорт 1-я страница\out'
dir_model = r'D:\данные\Model'
file_model_yolov5 = 'yolov5_passport_50epoch_last.pt'

import sys
sys.path.append(r'D:\PycharmProjects\4-206М-3_Diplom\yolov5')

from models.experimental import attempt_load

# Загрузка модели YOLOv5
model_path = os.path.join(dir_model, file_model_yolov5)
model = attempt_load(model_path)  # Загрузка на CPU по умолчанию

# Проверка и создание выходной директории
if not os.path.exists(dir_outphoto):
    os.makedirs(dir_outphoto)

# Получение списка JPG файлов
files = [f for f in os.listdir(dir_photo) if f.lower().endswith('.jpg')]

# Цикл по файлам с индикатором прогресса
for filename in tqdm(files, desc='Обрабатываю изображения', unit='file'):
    img_path = os.path.join(dir_photo, filename)
    img = Image.open(img_path)  # Загрузка изображения

    # Применение модели
    results = model(img)

    # Сохранение результатов
    save_path = os.path.join(dir_outphoto, filename)
    results.save(save_path)  # Сохранение результатов обработки

print("Обработка завершена. Результаты сохранены в:", dir_outphoto)