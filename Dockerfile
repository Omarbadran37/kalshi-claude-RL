# NFL Trading System Dockerfile

# Use Python 3.11 as base image
FROM python:3.14-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=production

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data/raw /app/data/processed /app/data/external /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash nfltrader && \
    chown -R nfltrader:nfltrader /app

# Switch to non-root user
USER nfltrader

# Expose ports for monitoring and health checks
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Default command
CMD ["python", "-m", "src.nfl_trading.main", "--config", "configs/prod.yaml"]