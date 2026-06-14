"""
Сервис для классификации дефектов пайки радиоэлементов на печатной плате
"""

import io
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from torchvision import models, transforms


# Pydantic модели для ответов
class PredictionResponse(BaseModel):
    class_name: str
    probability: float
    all_classes: Dict[str, float]


class HealthResponse(BaseModel):
    status: str


# Конфигурация
CLASS_NAMES = ['normal', 'unsolder', 'cold', 'excess', 'bridge']
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def create_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Создание модели ResNet18 с классификационной головой"""
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return model


class ImagePreprocessor:
    """Препроцессинг изображений"""

    def __init__(self, image_size: int = IMAGE_SIZE, device: str = 'cpu'):
        self.image_size = image_size
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)


class ModelInference:
    """Инференс модели"""

    def __init__(self, model: nn.Module, class_names: List[str], device: str = 'cpu'):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.preprocessor = ImagePreprocessor(device=device)
        self.model.eval()

    def predict_single(self, image_bytes: bytes) -> Dict:
        input_tensor = self.preprocessor.preprocess_image(image_bytes)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        probs_np = probabilities.cpu().numpy()[0]
        pred_idx = int(probs_np.argmax())

        all_classes = {
            self.class_names[i]: float(probs_np[i])
            for i in range(len(self.class_names))
        }

        return {
            'class_name': self.class_names[pred_idx],
            'probability': float(probs_np[pred_idx]),
            'all_classes': all_classes
        }


# Глобальные переменные для модели
_model = None
_inference = None


def load_model(model_path: str = "models/resnet18_soldering.pth") -> nn.Module:
    """Загрузка модели при старте приложения"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(NUM_CLASSES)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Модель загружена из {model_path}")
    else:
        print(f"⚠️ Файл модели не найден: {model_path}")
        print("   Используется новая модель со случайными весами")

    model = model.to(device)
    model.eval()
    return model


# Создание FastAPI приложения
app = FastAPI(
    title="Soldering Defect Detection API",
    description="Сервис для классификации дефектов пайки радиоэлементов на печатной плате",
    version="1.0.0"
)

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте сервиса"""
    global _model, _inference
    _model = load_model()
    _inference = ModelInference(_model, CLASS_NAMES, str(next(_model.parameters()).device))
    print(f"✅ Сервис запущен. Классы: {CLASS_NAMES}")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Проверка работоспособности сервиса"""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(file: UploadFile = File(...)):
    """Классификация дефекта пайки на изображении"""

    # Валидация формата файла
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = f".{file.filename.split('.')[-1].lower()}" if "." in file.filename else ""

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
        )

    # Валидация размера файла
    max_size_bytes = 10 * 1024 * 1024  # 10 МБ
    min_size_bytes = 1024  # 1 КБ

    image_bytes = await file.read()

    if len(image_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail="Размер файла превышает 10 МБ"
        )

    if len(image_bytes) < min_size_bytes:
        raise HTTPException(
            status_code=400,
            detail="Файл слишком маленький (минимум 1 КБ)"
        )

    # Валидация изображения
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Невалидный файл изображения: {str(e)}"
        )

    # Валидация минимального разрешения
    min_size = 64
    width, height = image.size

    if width < min_size or height < min_size:
        raise HTTPException(
            status_code=400,
            detail=f"Разрешение слишком маленькое (минимум {min_size}x{min_size} пикселей)"
        )

    # Инференс
    try:
        result = _inference.predict_single(image_bytes)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении предсказания: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
