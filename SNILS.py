from PIL import Image
import numpy as np
import os
import shutil
import sys

def is_color_similar(color1, color2, tolerance=0.2):
    """
    Проверяет, находится ли color2 в пределах tolerance от color1.
    """
    color1 = np.array(color1)
    color2 = np.array(color2)
    lower_bound = color1 * (1 - tolerance)
    upper_bound = color1 * (1 + tolerance)
    return np.all(color2 >= lower_bound) and np.all(color2 <= upper_bound)

def update_progress(progress):
    """
    Выводит или обновляет индикатор прогресса в консоль.
    """
    bar_length = 50  # Можно изменить для длиннее или короче индикатора
    block = int(round(bar_length * progress))
    text = "\rПрогресс: [{0}] {1}%".format("#" * block + "-" * (bar_length - block), round(progress * 100, 2))
    sys.stdout.write(text)
    sys.stdout.flush()

def move_similar_colored_images(source_dir, target_dir, base_colors, tolerance=0.2):
    """
    Перемещает изображения, основные цвета которых подходят под base_colors, из source_dir в target_dir.
    """
    base_colors = [tuple(int(color[i:i + 2], 16) for i in (0, 2, 4)) for color in base_colors]
    white_color = (255, 255, 255)  # Белый цвет в RGB
    base_colors.append(white_color)  # Добавляем белый цвет к базовым цветам

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(".jpg")]
    total_files = len(files)
    processed_files = 0

    for filename in files:
        file_path = os.path.join(source_dir, filename)
        try:
            with Image.open(file_path) as img:
                colors = img.getcolors(maxcolors=1024 * 1024)
                dominant_color = max(colors, key=lambda item: item[0])[1]

                if any(is_color_similar(dominant_color, base_color, tolerance) for base_color in base_colors):
                    target_path = os.path.join(target_dir, filename)
                    shutil.move(file_path, target_path)
                    print(f"\nФайл {filename} перемещен.")
        except Exception as e:
            print(f"\nОшибка при обработке файла {filename}: {e}")

        processed_files += 1
        update_progress(processed_files / total_files)

    print("\nОбработка завершена.")

# Пример использования
source_dir = "D://данные//06 Паспорт"
target_dir = "D://данные//08 СНИЛС//TMP"
base_colors = ["c8cf9c"]  # Основные цвета в hex
move_similar_colored_images(source_dir, target_dir, base_colors)
