import os
import shutil
import random
from tqdm import tqdm
import csv


def ensure_dir(directory):
    """Проверяет наличие каталога и создает его, если отсутствует."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except Exception as e:
            print(f"Ошибка при создании каталога {directory}: {e}")


def copy_file(source, target):
    """Копирует файл, пропуская ошибки."""
    try:
        shutil.copy2(source, target)
    except Exception as e:
        print(f"Ошибка при копировании {source} в {target}: {e}")


def find_min_files_count(folders):
    """Находит минимальное количество файлов среди заданных папок."""
    min_files_count = float('inf')
    for folder in folders:
        try:
            num_files = len(os.listdir(folder))
            if num_files < min_files_count:
                min_files_count = num_files
        except FileNotFoundError:
            continue  # Пропускаем папки, которые не найдены
    return min_files_count if min_files_count != float('inf') else 0


def prepare_data(base_dir, folders_to_process, exceptions, target_base_dir):
    folders = [os.path.join(base_dir, f) for f in folders_to_process if f not in exceptions]
    min_files_count = find_min_files_count(folders)
    print(f"Минимальное количество файлов в директории: {min_files_count}. Сколько файлов копировать?")
    files_to_copy_count = int(input("Введите количество файлов для копирования: "))

    ensure_dir(target_base_dir)
    classes_file_path = os.path.join(base_dir, "classes.csv")
    with open(classes_file_path, mode="w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Class Number", "Class Name"])
        for i, folder_path in enumerate(folders):
            folder_name = os.path.basename(folder_path)
            writer.writerow([i, folder_name])
            target_dir = os.path.join(target_base_dir, folder_name)
            ensure_dir(target_dir)
            copy_files(folder_path, target_dir, files_to_copy_count)


def copy_files(source_dir, target_dir, files_to_copy_count):
    if not os.path.exists(source_dir):
        print(f"Исходный каталог {source_dir} не найден.")
        return
    files = os.listdir(source_dir)
    if files_to_copy_count > len(files):
        print(f"В директории {source_dir} меньше файлов, чем запрошено для копирования.")
        files_to_copy = files  # Копируем все файлы, если их меньше, чем запрошено
    else:
        files_to_copy = random.sample(files, files_to_copy_count)

    # Создаем прогресс-бар с начальным описанием
    pbar = tqdm(total=len(files_to_copy), desc=f"Копирование файлов", leave=True)
    for file in files_to_copy:
        source_file = os.path.join(source_dir, file)
        target_file = os.path.join(target_dir, file)
        copy_file(source_file, target_file)
        # Обновляем описание с текущим прогрессом
        pbar.set_description(f"Копирование {pbar.n}/{len(files_to_copy)}")
        pbar.update(1)
    pbar.close()
    print(f"\nСкопировано {len(files_to_copy)} файлов из {source_dir} в {target_dir}")



base_dir = r"D:\куда_сохраняем"
target_base_dir = r"d:\куда_сохраняем_512"
folders_to_process = [
    "01 ПТС 1страница",
    "01 ПТС 2страница",
    "01 ПТС новые",
    "02 СТС 1страница",
    "02 СТС 2страница",
    "03 ОСАГО 1страница",
    "03 ОСАГО 2страница",
    "04 Разрешение на авто новое",
    "04 Разрешение старое",
    "05 Техосмотр",
    "06 Паспорт 1-я страница",
    "06 Паспорт прописка",
    "07 ВУ 1 страница",
    "07 ВУ 2 страница",
    "07 ВУ старые права",
    "08 СНИЛС 1 страница",
    "08 СНИЛС 2 страница",
    "09 Судимость",
    "10 ИНН",
]
exceptions = ["00 Не определено", "Модели", "Хлам"]

prepare_data(base_dir, folders_to_process, exceptions, target_base_dir)
