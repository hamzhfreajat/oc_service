FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV (required by EasyOCR)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the EasyOCR models at build time so the first request is fast
RUN python -c "import easyocr; easyocr.Reader(['en', 'ar'], gpu=False)"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
