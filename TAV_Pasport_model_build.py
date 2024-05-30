import cv2
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np


def preprocess_image(image_path):
    # Чтение изображения как байтового массива
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
        image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Cannot open image")

    # Преобразование в градации серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение гауссова размытия
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    return blurred_image

def read_and_resize_image(path, screen_width, screen_height):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    scale_width = screen_width / width / 2
    scale_height = screen_height / height
    scale = min(scale_width, scale_height)
    width = int(width * scale)
    height = int(height * scale)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

def select_image(initial_dir):
    root = tk.Tk()
    root.withdraw()  # Закрываем основное окно tkinter
    file_path = filedialog.askopenfilename(initialdir=initial_dir)
    return file_path


def save_preprocessed_image(image, original_path):
    # Получение директории и имени файла без расширения
    directory, filename_with_extension = os.path.split(original_path)
    filename, extension = os.path.splitext(filename_with_extension)

    # Создание нового имени файла с окончанием _REZ
    new_filename = f"{filename}_REZ{extension}"
    save_path = os.path.join(directory, new_filename)

    # Сохранение обработанного изображения
    cv2.imwrite(save_path, image)

    return save_path


# Предполагаемый размер экрана пользователя
screen_width, screen_height = 1920, 1080

# Получение текущего рабочего каталога
current_directory = os.getcwd()

# Выбор изображения пользователем из текущего каталога
image_path = select_image(current_directory)
if image_path:
    print(image_path)
    preprocessed_image = preprocess_image(image_path)  # Теперь функция определена
    preprocessed_image_path = save_preprocessed_image(preprocessed_image, image_path)

    # Чтение и масштабирование изображений
    original_resized = read_and_resize_image(image_path, screen_width, screen_height)
    preprocessed_resized = read_and_resize_image(preprocessed_image_path, screen_width, screen_height)

    # Объединение изображений для совместного отображения
    combined = cv2.hconcat([original_resized, preprocessed_resized])

    # Отображение объединенного изображения
    cv2.imshow("Before and After Preprocessing", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
