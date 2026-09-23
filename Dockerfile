# Базовый образ для x86_64 (сервер/мини-ПК). Для Jetson используйте
# официальный образ NVIDIA L4T с предустановленным TensorRT вместо этого Dockerfile.
FROM python:3.11-slim

# Системные зависимости, нужные OpenCV для работы с видео (даже headless-сборке)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Модель YOLOv8n скачается автоматически при первом запуске, если ее нет в образе.
# Чтобы не тянуть интернет на каждый холодный старт контейнера в проде, лучше
# заранее положить yolov8n.pt в каталог проекта перед сборкой образа.

CMD ["python", "smart_retail_monitor.py"]
