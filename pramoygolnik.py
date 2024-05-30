import cv2
import numpy as np
import os
import shutil
import sys


def is_size_similar(w, h, expected_w, expected_h, tolerance=0.1):
    """
    Проверяет, находятся ли размеры в пределах допуска от ожидаемых размеров.
    """
    lower_w = expected_w * (1 - tolerance)
    upper_w = expected_w * (1 + tolerance)
    lower_h = expected_h * (1 - tolerance)
    upper_h = expected_h * (1 + tolerance)

    return lower_w <= w <= upper_w and lower_h <= h <= upper_h


def is_angle_acceptable(angle, max_angle=15):
    """
    Проверяет, находится ли угол поворота прямоугольника в допустимых пределах.
    """
    angle = abs(angle) % 180
    if angle > 90:
        angle = abs(angle - 180)
    return angle <= max_angle


def update_progress(progress):
    """
    Выводит или обновляет индикатор прогресса в консоль.
    """
    bar_length = 50  # Измените, чтобы сделать индикатор длиннее или короче
    block = int(round(bar_length * progress))
    text = "\rПрогресс: [{0}] {1}%".format("#" * block + "-" * (bar_length - block), round(progress * 100, 2))
    sys.stdout.write(text)
    sys.stdout.flush()


def find_rect_and_move(source_dir, target_dir, rect_width, rect_height, tolerance=0.1, max_angle=15):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(files)
    files_processed = 0

    for filename in files:
        filepath = os.path.join(source_dir, filename)
        image = cv2.imread(filepath)

        # Проверка на успешное чтение изображения
        if image is None:
            print(f"Не удалось прочитать файл {filename}. Проверьте путь и целостность файла. {source_dir}   {filepath}" )
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        rect_found = False

        for c in contours:
            rect = cv2.minAreaRect(c)
            (x, y), (w, h), angle = rect

            if is_size_similar(w, h, rect_width, rect_height, tolerance) or is_size_similar(h, w, rect_width,
                                                                                            rect_height, tolerance):
                if is_angle_acceptable(angle, max_angle):
                    rect_found = True
                    break

        if rect_found:
            shutil.move(filepath, os.path.join(target_dir, filename))
            print(f"\nФайл {filename} перемещен.")

        files_processed += 1
        update_progress(files_processed / total_files)

    print("\nОбработка завершена.")


# Пример использования
source_dir = r"D:\данные\06 Паспорт"
target_dir = r"D:\данные\07 ВУ\TMP"
rect_width = int(85 * (300 / 25.4))  # Примерное преобразование из мм в пиксели для 300 DPI
rect_height = int(53 * (300 / 25.4))
find_rect_and_move(source_dir, target_dir, rect_width, rect_height)