# 📖 Документация проекта: Классификатор дефектов пайки

## 📋 Содержание

1. [Описание проекта](#-описание-проекта)
2. [Структура проекта](#-структура-проекта)
3. [Python-модули](#-python-модули)
4. [Требования](#-требования)
5. [Структура датасета](#-структура-датасета)
6. [Архитектура модели](#-архитектура-модели)
7. [Обучение модели](#-обучение-модели)
8. [Использование](#-использование)
9. [Известные особенности](#-известные-особенности)
10. [Метрики](#-метрики)

---

## 🎯 Описание проекта

Проект представляет собой систему классификации дефектов пайки на основе свёрточной нейронной сети **ResNet18** с использованием трансферного обучения (pre-trained на ImageNet).

**Цель:** автоматическое определение типа дефекта пайки по изображению.

**Поддерживаемые классы дефектов:**

| Класс | Описание |
|-------|----------|
| `bridge` | Перемычка между контактами |
| `excess` | Избыток припоя |
| `unsolder` | Непропай |

---

## 📁 Структура проекта

```
project/
├── dataset/
│   ├── train/
│   │   ├── bridge/
│   │   ├── excess/
│   │   └── unsolder/
│   └── val/
│       ├── bridge/
│       ├── excess/
│       └── unsolder/
├── models/                      # Сохранённые модели (.pth файлы)
│   ├── v1.0.0.pth
│   └── metadata.json
├── train.py                     # Скрипт обучения
├── inference.py                 # Скрипт инференса
├── model_manager.py             # Управление моделями
├── utils.py                     # Вспомогательные функции
└── README.md
```

---

## 🐍 Python-модули

### `train.py`

**Назначение:** Скрипт для обучения модели.

**Основная функция:**

```python
def train_and_save_model() -> Tuple[nn.Module, ModelManager]:
    """
    Обучение модели ResNet18 и сохранение лучшей версии.
    
    Returns:
        model: Обученная PyTorch модель
        model_manager: Экземпляр ModelManager для работы с моделью
    """
```

**Что делает:**
- Загружает датасет из `dataset/train` и `dataset/val`
- Применяет аугментации к обучающей выборке
- Обучает модель в течение 20 эпох
- Сохраняет лучшую модель по точности на валидации
- Возвращает обученную модель и менеджер

**Использование:**

```python
from train import train_and_save_model

model, model_manager = train_and_save_model()
```

---

### `model_manager.py`

**Назначение:** Класс для управления жизненным циклом моделей (сохранение, загрузка, версионирование).

**Класс `ModelManager`:**

```python
class ModelManager:
    def __init__(self, models_dir: str = "models"):
        """
        Args:
            models_dir: Директория для хранения моделей
        """
    
    def save_model(
        self,
        model: nn.Module,
        class_names: List[str],
        metrics: Dict[str, Any],
        version: str
    ) -> str:
        """
        Сохранение модели и метаданных.
        
        Args:
            model: PyTorch модель
            class_names: Список имён классов
            metrics: Словарь с метриками обучения
            version: Версия модели (например, "1.0.0")
        
        Returns:
            Путь к сохранённому файлу модели
        """
    
    def load_model(self, version: str = "latest") -> Tuple[nn.Module, Dict]:
        """
        Загрузка модели по версии.
        
        Args:
            version: Версия модели или "latest" для последней
        
        Returns:
            Кортеж (модель, метаданные)
        """
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Получение списка всех сохранённых моделей.
        
        Returns:
            Список словарей с информацией о моделях
        """
```

**Использование:**

```python
from model_manager import ModelManager

# Создание менеджера
manager = ModelManager(models_dir="models")

# Загрузка последней модели
model, metadata = manager.load_model(version="latest")

# Просмотр всех моделей
all_models = manager.list_models()
for m in all_models:
    print(f"Version: {m['version']}, Accuracy: {m['metrics']['val_accuracy']:.2f}%")
```

---

### `inference.py`

**Назначение:** Скрипт для инференса (предсказания) на новых изображениях.

**Основные функции:**

```python
def predict(
    image_path: str,
    model_manager: ModelManager = None,
    model: nn.Module = None,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Предсказание класса для одного изображения.
    
    Args:
        image_path: Путь к изображению
        model_manager: ModelManager для загрузки модели (если model не указан)
        model: Загруженная модель (опционально)
        class_names: Список имён классов (опционально)
    
    Returns:
        Словарь с результатами:
        {
            'class': str,              # Предсказанный класс
            'confidence': float,       # Уверенность (0-1)
            'probabilities': Dict      # Вероятности всех классов
        }
    """

def predict_batch(
    image_paths: List[str],
    model_manager: ModelManager = None,
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Пакетное предсказание для нескольких изображений.
    
    Args:
        image_paths: Список путей к изображениям
        model_manager: ModelManager для загрузки модели
        batch_size: Размер батча
    
    Returns:
        Список результатов для каждого изображения
    """
```

**Использование:**

```python
from inference import predict, predict_batch
from model_manager import ModelManager

# Загрузка модели
manager = ModelManager()

# Предсказание для одного изображения
result = predict(
    image_path="test_image.png",
    model_manager=manager
)
print(f"Класс: {result['class']}, Уверенность: {result['confidence']:.2%}")

# Пакетное предсказание
results = predict_batch(
    image_paths=["img1.png", "img2.png", "img3.png"],
    model_manager=manager,
    batch_size=16
)

for i, res in enumerate(results):
    print(f"Изображение {i+1}: {res['class']} ({res['confidence']:.2%})")
```

---

### `utils.py`

**Назначение:** Вспомогательные функции для работы с данными и визуализации.

**Основные функции:**

```python
def visualize_predictions(
    image_paths: List[str],
    predictions: List[Dict[str, Any]],
    save_path: str = None
) -> None:
    """
    Визуализация результатов предсказания.
    
    Args:
        image_paths: Список путей к изображениям
        predictions: Список результатов предсказания
        save_path: Путь для сохранения визуализации (опционально)
    """

def get_transforms(phase: str = "val") -> transforms.Compose:
    """
    Получение трансформаций для указанной фазы.
    
    Args:
        phase: "train" или "val"
    
    Returns:
        Compose объект с трансформациями
    """

def plot_training_history(metrics: Dict[str, List[float]]) -> None:
    """
    Построение графиков истории обучения.
    
    Args:
        metrics: Словарь с метриками по эпохам
    """
```

**Использование:**

```python
from utils import visualize_predictions, plot_training_history

# Визуализация предсказаний
visualize_predictions(
    image_paths=["test1.png", "test2.png"],
    predictions=[result1, result2],
    save_path="predictions.png"
)

# Построение графиков обучения
plot_training_history({
    'train_acc': [85.2, 88.5, 91.3],
    'val_acc': [82.1, 85.7, 89.2]
})
```

---

## ⚙️ Требования

- Python 3.10+
- PyTorch 2.0+
- torchvision
- Pillow
- matplotlib (для визуализации)
- CUDA (опционально, для ускорения на GPU)

**Установка зависимостей:**

```bash
pip install torch torchvision Pillow matplotlib
```

---

## 🗂 Структура датасета

Датасет организован по принципу `ImageFolder` из torchvision:

```
dataset/
├── train/
│   ├── bridge/       # 3583 изображения (.png)
│   ├── excess/       # 2471 изображение (.png)
│   └── unsolder/     # 1281 изображение (.png)
└── val/
    ├── bridge/       # 768 изображений (.png)
    ├── excess/       # 530 изображений (.png)
    └── unsolder/     # 275 изображений (.png)
```

### Поддерживаемые форматы изображений

`.jpg`, `.jpeg`, `.png`, `.ppm`, `.bmp`, `.pgm`, `.tif`, `.tiff`, `.webp`

> ⚠️ **Важно:** В текущем датасете используются только файлы формата `.png`.

---

## 🧠 Архитектура модели

Используется предобученная модель **ResNet18** с модифицированным классификатором:

```python
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Linear(128, NUM_CLASSES)  # 3 класса
)
```

### Параметры модели

| Параметр | Значение |
|----------|----------|
| Базовая архитектура | ResNet18 |
| Предобучение | ImageNet1K_V1 |
| Dropout | 0.2 |
| Скрытый слой | 128 нейронов |
| Выходов | 3 (по числу классов) |
| Размер входного изображения | 224 × 224 |

---

## 🚀 Обучение модели

### Конфигурация обучения

| Параметр | Значение |
|----------|----------|
| Batch size | 32 |
| Количество эпох | 20 |
| Learning rate | 0.001 |
| Оптимизатор | Adam |
| Функция потерь | CrossEntropyLoss |
| Устройство | CUDA (если доступно), иначе CPU |

### Аугментации для обучающей выборки

- `Resize(224, 224)` — приведение к фиксированному размеру
- `RandomHorizontalFlip()` — случайное отражение по горизонтали
- `RandomRotation(15)` — случайный поворот до ±15°
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` — нормализация ImageNet

### Для валидации

Только `Resize` и `Normalize` (без аугментаций).

---

## 💻 Использование

### 1. Обучение модели

```python
from train import train_and_save_model

model, model_manager = train_and_save_model()
```

### 2. Загрузка обученной модели

```python
from model_manager import ModelManager

manager = ModelManager()
model, metadata = manager.load_model(version="latest")

print(f"Версия: {metadata['version']}")
print(f"Точность: {metadata['metrics']['best_val_accuracy']:.2f}%")
```

### 3. Предсказание на новом изображении

```python
from inference import predict

result = predict(
    image_path="new_image.png",
    model_manager=manager
)

print(f"Предсказанный класс: {result['class']}")
print(f"Уверенность: {result['confidence']:.2%}")
print(f"Вероятности всех классов:")
for cls, prob in result['probabilities'].items():
    print(f"  {cls}: {prob:.2%}")
```

### 4. Пакетная обработка

```python
from inference import predict_batch

image_paths = ["img1.png", "img2.png", "img3.png"]
results = predict_batch(image_paths, model_manager=manager)

for path, result in zip(image_paths, results):
    print(f"{path}: {result['class']} ({result['confidence']:.2%})")
```

---

## ⚠️ Известные особенности

### 1. Служебные папки Jupyter

При работе в Google Colab или Jupyter автоматически создаются папки `.ipynb_checkpoints`, которые `ImageFolder` ошибочно воспринимает как классы.

**Решение:** удалить их перед обучением:

```python
import shutil
from pathlib import Path

DATA_DIR = "dataset"
for split in ['train', 'val']:
    checkpoint_path = Path(DATA_DIR) / split / ".ipynb_checkpoints"
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
```

### 2. Актуальный список классов

Список классов определяется **на основе имён папок в датасете**, а не переменной `CLASS_NAMES`. Убедитесь, что они совпадают:

```python
CLASS_NAMES = ['bridge', 'excess', 'unsolder']  # Должны соответствовать папкам
```

### 3. Регистр расширений файлов

`ImageFolder` чувствителен к регистру расширений. Файлы с расширением `.JPG` не будут найдены.

**Решение:** привести все расширения к нижнему регистру:

```python
from pathlib import Path

for path in Path("dataset").rglob('*'):
    if path.is_file() and path.suffix != path.suffix.lower():
        path.rename(path.with_suffix(path.suffix.lower()))
```

---

## 📊 Метрики

После обучения сохраняются следующие метрики:

- `train_accuracy` — точность на обучающей выборке
- `val_accuracy` — точность на валидации в последней эпохе
- `best_val_accuracy` — лучшая точность на валидации за всё обучение
- `epochs` — количество эпох
- `batch_size` — размер батча
- `learning_rate` — скорость обучения

Модель сохраняется только при достижении новой лучшей точности на валидации.

---

## 📝 Changelog

### v1.0.0 (2026-06-17)
- ✅ Базовая реализация обучения на ResNet18
- ✅ Поддержка 3 классов дефектов: `bridge`, `excess`, `unsolder`
- ✅ Аугментации данных для обучающей выборки
- ✅ Сохранение лучшей модели через `ModelManager`
- ✅ Модуль `inference.py` для предсказаний
- ✅ Модуль `utils.py` для визуализации
- ✅ Исправлена проблема со служебными папками `.ipynb_checkpoints`

---

## 📬 Контакты

При возникновении вопросов или проблем — создайте issue в репозитории проекта.
