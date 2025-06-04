# 1. Base Image
FROM python:3.9-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
# For HuggingFace cache (can be overridden by user mount)
ENV HF_HOME /app/.cache/huggingface
ENV TRANSFORMERS_CACHE /app/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE /app/.cache/huggingface/datasets
# For Prometheus multiprocess mode with Gunicorn
ENV PROMETHEUS_MULTIPROC_DIR /app/prometheus_metrics

# 3. Set Working Directory
WORKDIR $APP_HOME

# 4. Create necessary directories and set permissions
RUN mkdir -p $HF_HOME/transformers $HF_HOME/datasets $PROMETHEUS_MULTIPROC_DIR && \
    chmod -R 777 $HF_HOME $PROMETHEUS_MULTIPROC_DIR
# Note: 777 is permissive; adjust if running as non-root user with specific GID/UID.

# 5. Install system dependencies (if any, e.g., libgomp1 for some torch operations)
# RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 6. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code into the container
COPY ./app $APP_HOME/app

# 8. Expose the port the app runs on
EXPOSE 8000

# 9. Define a volume for HuggingFace cache to persist models across container restarts
# This allows users to mount an external volume to this path.
VOLUME $HF_HOME

# 10. Healthcheck (optional but good practice)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

# 11. Define the command to run the application
# Gunicorn with Uvicorn workers.
# -w: number of worker processes. (2 * num_cores) + 1 is a common recommendation.
# --preload: If set, app is loaded in master before forking workers.
#            This means lru_cache would be shared if it's pickleable and not thread-local.
#            However, HuggingFace pipelines are complex objects, often not trivially shareable.
#            Default (no --preload) means each worker loads its own app & cache.
#            For model serving, per-worker loading/caching is often safer and simpler for memory management.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000", "--log-level", "info", "--timeout", "300"]