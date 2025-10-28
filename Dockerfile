# Use official Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Upgrade pip tools and install CPU-only torch from PyTorch index first,
# then install the rest of requirements to avoid pip trying to fetch torch+cpu from PyPI.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu || \
    python -m pip install --no-cache-dir torch==2.9.0 --no-deps && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
