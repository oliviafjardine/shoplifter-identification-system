# Multi-stage Dockerfile for Shoplifting Detection System
# Implements REQ-022: Containerized deployment with Docker

# Stage 1: Base image with system dependencies
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstreamer-plugins-base1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    git \
    vim \
    htop \
    supervisor \
    nginx \
    redis-server \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash shoplifter
USER shoplifter
WORKDIR /home/shoplifter

# Stage 2: Python dependencies
FROM base as python-deps

# Create virtual environment
RUN python3 -m venv /home/shoplifter/venv
ENV PATH="/home/shoplifter/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY --chown=shoplifter:shoplifter requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Stage 3: Application code
FROM python-deps as app

# Set working directory
WORKDIR /home/shoplifter/app

# Copy application code
COPY --chown=shoplifter:shoplifter . .

# Create necessary directories
RUN mkdir -p \
    logs \
    evidence \
    models \
    data \
    config \
    static \
    templates \
    uploads \
    backups

# Set permissions
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Stage 4: Production image
FROM app as production

# Switch to root for system configuration
USER root

# Copy supervisor configuration
COPY deployment/supervisor/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy nginx configuration
COPY deployment/nginx/nginx.conf /etc/nginx/sites-available/default

# Create log directories
RUN mkdir -p /var/log/shoplifting-detection && \
    chown -R shoplifter:shoplifter /var/log/shoplifting-detection

# Install health check script
COPY deployment/scripts/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# Create startup script
COPY deployment/scripts/startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

# Switch back to application user
USER shoplifter
WORKDIR /home/shoplifter/app

# Expose ports
EXPOSE 8080 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/startup.sh"]

# Default command
CMD ["production"]

# Stage 5: Development image
FROM app as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    jupyter \
    ipython \
    vim \
    tmux \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install \
    jupyter \
    ipython \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit

USER shoplifter
WORKDIR /home/shoplifter/app

# Expose additional ports for development
EXPOSE 8888 5432 6379

# Development command
CMD ["development"]

# Stage 6: Testing image
FROM development as testing

# Copy test configuration
COPY tests/ tests/
COPY pytest.ini .
COPY .coveragerc .

# Run tests during build (optional)
# RUN python -m pytest tests/ --cov=src --cov-report=html

CMD ["test"]

# Build arguments for different stages
ARG BUILD_STAGE=production
FROM ${BUILD_STAGE} as final
