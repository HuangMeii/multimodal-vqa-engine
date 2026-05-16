FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install "numpy<2"

# Install PyTorch 2.4.0 CPU FIRST (trước requirements để tránh accelerate kéo torch CUDA ~GB)
RUN pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install requirements (transformers==4.44.0 tương thích với torch 2.4)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./services ./services
COPY ./routers ./routers
COPY ./main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
