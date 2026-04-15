# ── Assignment 3: Dockerize the ML application ───────────────────────────────
# Base image: lightweight Python runtime
FROM python:3.11-slim

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="NLP Sentiment Analyzer - MLOps Edition"
LABEL version="1.1"

# Set working directory inside container
WORKDIR /app

# Copy requirements first (layer caching — avoids re-installing on code changes)
COPY sentiment-analyzer/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY sentiment-analyzer/ ./sentiment-analyzer/
COPY model_versions.json .

# Create a non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the Flask port
EXPOSE 5000

# Health check (Kubernetes readiness probe compatible)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run the Flask app
CMD ["python", "sentiment-analyzer/app.py"]
