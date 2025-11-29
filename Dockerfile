FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps required for Playwright, Tesseract, PDF tools, and audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    wget \
    curl \
    gnupg \
    libglib2.0-0 \
    libnss3 \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libgconf-2-4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    tesseract-ocr \
    poppler-utils \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install Playwright browsers (with required deps)
RUN python -m playwright install --with-deps chromium

# Copy application code
COPY . /app

EXPOSE 7860

# Run the FastAPI server on port 7860 (Hugging Face Spaces commonly uses 7860)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
