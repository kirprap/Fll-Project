# Use official Python 3.11 slim image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=5000 \
    LANG=C.UTF-8

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    liblcms2-dev \
    libjpeg-dev \
    libtiff-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    tk \
    zlib1g-dev \
    locales \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen en_US.UTF-8

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose internal port
EXPOSE 5000

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]
