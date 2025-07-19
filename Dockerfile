# Use an official Python base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        poppler-utils \
        tcl \
        tk \
        zlib1g \
        git \
        ghostscript \
        openjdk-17-jre-headless \
        && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        beautifulsoup4 \
        email-validator \
        flask \
        flask-sqlalchemy \
        gunicorn \
        openpyxl \
        "paddlex[ocr]" \
        pandas \
        pdf2image \
        pillow \
        psycopg2-binary \
        werkzeug \
        pdfplumber \
        camelot-py[cv] \
        tabula-py \
        PyPDF2 \
        uvicorn[standard]

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Run the application with timeout configuration
CMD ["uvicorn", "simple_pdf_api_prod:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75", "--limit-concurrency", "10"]
