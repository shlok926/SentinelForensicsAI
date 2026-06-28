# ==========================================
# ENTERPRISE DOCKER DEPLOYMENT SPECIFICATION
# ==========================================

# Base Builder stage
FROM python:3.10-slim AS builder

WORKDIR /build

# Install compilation tools and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install virtualenv and build dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final Production runtime stage
FROM python:3.10-slim AS runner

WORKDIR /app

# Install runtime system packages (FFmpeg and OpenCV GL drivers are required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY app/ app/
COPY ai_engine/ ai_engine/
COPY README.md .

# Configure Environment Variables
ENV PORT=8000
ENV HOST=0.0.0.0
ENV ENV=production
ENV DEBUG=False
ENV PYTHONUNBUFFERED=1

# Expose API and Prometheus metrics ports
EXPOSE 8000

# Start server using Uvicorn worker bound to FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
