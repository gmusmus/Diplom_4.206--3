import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
import logging
from PIL import Image
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

def pil_loader(path, bad_folder="D://данные//BAD"):  # Функция для загрузки изображений с обработкой исключений
    try:
        with open(path, 'rb') as f:  # Открываем файл в режиме чтения байтов
            img = Image.open(f)  # Пытаемся открыть изображение
            img.verify()  # Проверяем изображение на целостность
            # После вызова verify() изображение нужно открыть заново, так как verify() выгружает данные из памяти
            img = Image.open(f)  # Снова открываем изображение для дальнейшей работы
            return img.convert('RGB')  # Конвертируем изображение в формат RGB и возвращаем его
    except (IOError, Image.UnidentifiedImageError) as e:
        # В случае возникновения исключений, связанных с ошибкой ввода-вывода или неопознанным форматом изображения
        logging.warning(f"Поврежденный файл изображения {path} будет перемещен. Ошибка: {e}")  # Логгируем предупреждение
        os.makedirs(bad_folder, exist_ok=True)  # Создаем директорию для поврежденных файлов, если она еще не существует
        shutil.move(path, os.path.join(bad_folder, os.path.basename(path)))  # Перемещаем поврежденный файл в папку BAD
        return None  # Возвращаем None, так как загрузить изображение не удалось


# Пользовательский класс датасета, который пропускает повреждённые изображения
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        if sample is None:
            return torch.zeros(3, 224, 224), target  # Возвращаем заглушку для поврежденного файла
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


# Функция создания модели
def create_model(num_classes):
    #model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  #ResNet18
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  #ResNet50
    #model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT) #ResNet152
    # Получаем количество признаков на последнем полносвязном слое
    num_ftrs = model.fc.in_features
    # Заменяем последний полносвязный слой на новый с количеством выходов, равным num_classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model  # Возвращаем модифицированную модель

# Функция аугментации данных из одной фото можно получить до 64 фотографий
def augmented_transforms():
    #Создание пайплайна аугментации изображений с использованием библиотеки torchvision.transforms

    return transforms.Compose([
        transforms.RandomHorizontalFlip(),        #Случайное горизонтальное отражение изображения с вероятностью 0.5

        transforms.RandomVerticalFlip(),        #Случайное вертикальное отражение изображения с вероятностью 0.5

        transforms.RandomApply([
            transforms.RandomRotation(degrees=(-180, 180))
        ], p=0.5),        #Случайный поворот изображения на произвольный угол от -180 до 180 градусов с вероятностью 0.5

        transforms.RandomChoice([
            transforms.RandomRotation(degrees=(-45, -45)),
            transforms.RandomRotation(degrees=(45, 45)),
            transforms.RandomRotation(degrees=(-30, -30)),
            transforms.RandomRotation(degrees=(30, 30))
        ]),        #Случайный выбор одного из поворотов на -45, 45, -30 или 30 градусов

        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),        #Случайный обрез размером 224x224 пикселя со случайным масштабированием от 0.5 до 1.0

        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2), shear=(20, 20, 20, 20)),        #Случайные аффинные преобразования: масштабирование от 0.8 до 1.2 и сдвиг в пределах [-20, 20] по осям X и Y

        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),        #Изменение цветовых характеристик изображения: яркость, контраст, насыщенность и оттенок

        transforms.RandomApply([
            transforms.Resize((224, int(224 * 0.8))),  #Сжатие изображения по горизонтали
            transforms.Resize((int(224 * 0.8), 224))  #Сжатие изображения по вертикали
        ], p=0.5),        #Случайный выбор одного из вариантов изменения размера с вероятностью 0.5

        transforms.RandomGrayscale(p=0.1),        #Случайное преобразование изображения в оттенки серого с вероятностью 0.1

        transforms.Resize((224, 224)),        #Изменение размера изображения до 224x224 пикселя

        transforms.ToTensor(),        #Преобразование изображения в тензор PyTorch

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),        #Нормализация значений тензора с использованием предварительно вычисленных средних и стандартных отклонений
    ])


# Функция загрузки данных
def load_data(data_dir, transform):
    # Создаем датасет изображений с применением предобработки
    dataset = CustomImageFolder(root=data_dir, transform=transform)

    # Рассчитываем размеры обучающей и тестовой выборок (80% на обучение, 20% на тест)
    train_size = int(0.8 * len(dataset))  # 80% от общего числа образцов для обучения
    test_size = len(dataset) - train_size  # Оставшиеся образцы для тестирования

    # Разбиваем датасет на обучающую и тестовую выборки случайным образом
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Создаем загрузчики данных для обучающей выборки (с перемешиванием данных)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # И для тестовой выборки (без перемешивания)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Возвращаем загрузчики
    return train_loader, test_loader

# Функция валидации модели
def validate_model(model, test_loader, criterion):
    model.eval()  # Переключаем модель в режим оценки (выключаем dropout и т.д.)
    test_loss = 0.0  # Инициализируем суммарную потерю на тестовых данных
    correct = 0  # Счётчик количества правильно предсказанных меток
    total = 0  # Общее количество обработанных примеров

    with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
        for inputs, labels in test_loader:  # Перебираем пакеты (batches) данных
            outputs = model(inputs)  # Вычисляем предсказания модели
            loss = criterion(outputs, labels)  # Вычисляем потерю для пакета
            test_loss += loss.item()  # Добавляем потерю пакета к общей потере
            _, predicted = torch.max(outputs.data, 1)  # Получаем предсказанные классы
            total += labels.size(0)  # Увеличиваем общее кол-во обработанных примеров
            correct += (predicted == labels).sum().item()  # Считаем кол-во правильных предсказаний

    test_loss /= len(test_loader)  # Вычисляем среднюю потерю
    accuracy = 100 * correct / total  # Вычисляем точность как процент правильных ответов

    return test_loss, accuracy  # Возвращаем среднюю потерю и точность



def plot_training_graphs(train_losses, test_losses, accuracy_list, save_path):
    epochs = range(1, len(train_losses) + 1)  # Генерируем список эпох от 1 до количества эпох включительно

    plt.figure(figsize=(12, 6))  # Создаем фигуру для графиков с размерами 12x6 дюймов
    plt.subplot(1, 2, 1)  # Добавляем первый подграфик (1 ряд, 2 колонки, первый график)
    plt.plot(epochs, train_losses, label='Train Loss')  # Строим график потерь на обучающей выборке
    plt.plot(epochs, test_losses, label='Test Loss')  # Строим график потерь на тестовой выборке
    plt.title('Loss during training')  # Заголовок графика
    plt.xlabel('Epoch')  # Подпись оси X
    plt.ylabel('Loss')  # Подпись оси Y
    plt.legend()  # Отображаем легенду графика

    plt.subplot(1, 2, 2)  # Добавляем второй подграфик (1 ряд, 2 колонки, второй график)
    plt.plot(epochs, accuracy_list, label='Accuracy')  # Строим график точности
    plt.title('Accuracy during training')  # Заголовок графика
    plt.xlabel('Epoch')  # Подпись оси X
    plt.ylabel('Accuracy (%)')  # Подпись оси Y, указываем, что точность в процентах
    plt.legend()  # Отображаем легенду графика

    plt.tight_layout()  # Автоматически корректируем расстояния между подграфиками
    plt.savefig(os.path.join(save_path, 'training_graphs.png')) # Сохраняем созданные графики в файл
    plt.show()  # Отображаем графики


# Функция обучения модели
def train_model(model, train_loader, test_loader, epochs=20):
    train_losses = []  # Список для хранения потерь на обучении
    test_losses = []  # Список для хранения потерь на валидации
    accuracy_list = []  # Список для хранения значений точности

    criterion = nn.CrossEntropyLoss()  # Функция потерь (кросс-энтропия)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Оптимизатор (Adam) с шагом обучения 0.001
    save_dir = "D:\\данные\\Модели"  # Директория для сохранения моделей
    os.makedirs(save_dir, exist_ok=True)  # Создаем директорию, если она не существует

    for epoch in range(epochs):  # Цикл по эпохам
        model.train()  # Переключаем модель в режим обучения
        running_loss = 0.0  # Суммарная потеря за эпоху
        # Инициализируем progress bar для визуализации процесса обучения
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")

        for i, (inputs, labels) in progress_bar:  # Цикл по данным обучения
            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(inputs)  # Прямой проход
            loss = criterion(outputs, labels)  # Вычисляем потерю
            loss.backward()  # Обратный проход (вычисляем градиенты)
            optimizer.step()  # Шаг оптимизатора (обновляем веса)
            running_loss += loss.item()  # Добавляем потерю от текущего пакета к суммарной потере
            # Обновляем значение потери в progress bar
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        train_loss = running_loss / len(train_loader)  # Средняя потеря на обучении за эпоху
        train_losses.append(train_loss)  # Добавляем в список потерь на обучении

        test_loss, accuracy = validate_model(model, test_loader, criterion)  # Валидируем модель
        test_losses.append(test_loss)  # Добавляем потерю на валидации
        accuracy_list.append(accuracy)  # Добавляем точность

        # Выводим статистику по текущей эпохе
        print(
            f'\nEpoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Путь для сохранения модели
        save_path = os.path.join(save_dir, f'model_epoch_PASPORT_PROPIS_ResNet50_{epoch + 1}_acc_{accuracy:.2f}_loss_{test_loss:.4f}.pth')
        # Сохраняем модель
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'accuracy': accuracy,
        }, save_path)
        print(f"\nМодель сохранена: {save_path}")  # Выводим сообщение о сохранении модели

    plot_training_graphs(train_losses, test_losses, accuracy_list, save_dir) # После обучения строим графики
    print('\nОбучение модели завершено.')



# Подготовка и запуск обучения модели
num_classes = 2
model = create_model(num_classes)
transform = augmented_transforms()
train_loader, test_loader = load_data(r"D:\данные\06 Паспорт 1-я страница", transform)
train_model(model, train_loader, test_loader)

print("\nРабота скрипта завершена.")