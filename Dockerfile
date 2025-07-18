FROM python:3.11

# Update package lists and install system dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Remove Tesseract and related dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir paddlepaddle paddlex[ocr]

# Pre-download paddlex OCR models
RUN python -c "from paddlex import create_pipeline; create_pipeline(pipeline='table_recognition')"

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p temp_files

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "simple_pdf_api_prod:app", "--host", "0.0.0.0", "--port", "8000"]
