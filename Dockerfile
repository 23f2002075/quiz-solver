FROM python:3.12-slim

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    chromium \
    chromium-driver \
    curl \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies with pre-built wheels
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code (NOT .env - it's in .gitignore)
COPY app.py agent.py tools.py ./

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Run application
# Environment variables must be set at runtime via Render dashboard
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
