FROM python:3.11-slim

# Update package lists and install system dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        poppler-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove heavy OCR dependencies from requirements
RUN pip uninstall -y paddlex paddlepaddle || true

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p temp_files

# Expose port
EXPOSE 8000

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application with timeout configuration
CMD ["uvicorn", "simple_pdf_api_prod:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75", "--limit-concurrency", "10"]
