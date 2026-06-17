
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Soldering Defect Detection API",
    description="Сервис для классификации дефектов пайки радиоэлементов на печатной плате",
    version="1.0.0"
)

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """
    Проверка работоспособности сервиса.
    Возвращает статус "ok" если сервис работает.
    """
    return HealthResponse(status="ok")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Принимает изображение и возвращает классификацию дефекта пайки.

    - **file**: изображение в формате JPG, JPEG или PNG (макс. 10 МБ)

    Возвращает:
    - **class_name**: предсказанный класс дефекта
    - **probability**: вероятность предсказания
    - **all_classes**: вероятности для всех классов
    """

    # 3.2. Валидация формата файла
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = f".{file.filename.split('.')[-1].lower()}" if "." in file.filename else ""

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
        )

    # 3.2. Валидация размера файла
    max_size_mb = 10  # Максимальный размер из задания 1.2
    max_size_bytes = max_size_mb * 1024 * 1024

    # Чтение файла
    image_bytes = await file.read()

    if len(image_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Размер файла превышает {max_size_mb} МБ (максимальный размер: 10 МБ)"
        )

    if len(image_bytes) < 1024:  # Минимум 1 КБ из задания 1.2
        raise HTTPException(
            status_code=400,
            detail="Файл слишком маленький или повреждён (минимальный размер: 1 КБ)"
        )

    # 3.2. Валидация изображения (проверка, что файл является корректным изображением)
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Невалидный файл изображения: {str(e)}"
        )

    # 3.2. Валидация минимального разрешения (из задания 1.2)
    min_size = 64  # Минимальное разрешение из задания 1.2
    width, height = image.size

    if width < min_size or height < min_size:
        raise HTTPException(
            status_code=400,
            detail=f"Разрешение изображения слишком маленькое (минимум {min_size}x{min_size} пикселей)"
        )

    # Инференс модели с использованием существующего класса ModelInference
    try:
        # Используем метод predict_single из класса ModelInference
        result = inference.predict_single(image_bytes)

        # Формируем ответ в требуемом формате
        return PredictionResponse(
            class_name=result['class_name'],
            probability=result['probability'],
            all_classes=result['all_classes']
        )

    except Exception as e:
        # 3.2. Корректная обработка ошибок инференса
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении предсказания: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Корректная обработка ошибок с понятными сообщениями"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Обработка непредвиденных ошибок"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Внутренняя ошибка сервера: {str(exc)}"}
    )
