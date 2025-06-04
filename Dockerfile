# 1. Base Image
FROM python:3.9-slim

# 2. Install Nginx and other utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
# For HuggingFace cache (can be overridden by user mount)
ENV HF_HOME /app/.cache/huggingface
ENV TRANSFORMERS_CACHE /app/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE /app/.cache/huggingface/datasets
# For Prometheus multiprocess mode with Gunicorn
ENV PROMETHEUS_MULTIPROC_DIR /app/prometheus_metrics

# 4. Set Working Directory
WORKDIR $APP_HOME

# 5. Create necessary directories and set permissions
RUN mkdir -p $HF_HOME/transformers $HF_HOME/datasets $PROMETHEUS_MULTIPROC_DIR $APP_HOME/app $APP_HOME/static && \
    chown -R www-data:www-data $HF_HOME $PROMETHEUS_MULTIPROC_DIR $APP_HOME/static && \
    chmod -R 775 $HF_HOME $PROMETHEUS_MULTIPROC_DIR $APP_HOME/static
# Gunicorn needs to write to its dirs and Nginx (www-data) needs to read static files.

# 6. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code into the container and Nginx conf
COPY ./app $APP_HOME/app
COPY nginx.conf /etc/nginx/nginx.conf

# Ensure permissions for static files after they are copied
RUN chown -R www-data:www-data $APP_HOME/app/static && \
    find $APP_HOME/app/static -type d -exec chmod 755 {} \; && \
    find $APP_HOME/app/static -type f -exec chmod 644 {} \;

# 8. Ensure Nginx logs go to stdout/stderr for Docker logging
RUN ln -sf /dev/stdout /var/log/nginx/access.log && \
    ln -sf /dev/stderr /var/log/nginx/error.log

# 9. Expose Nginx Port
EXPOSE 80

# 10. Define a volume for HuggingFace cache to persist models across container restarts
VOLUME $HF_HOME

# 11. Healthcheck targeting Nginx which proxies to the app's healthz
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:80/api/healthz || exit 1

# 12. Define the command to run the application
# Gunicorn with Uvicorn workers.
# Simple startup: Start Gunicorn in background, then Nginx in foreground
# For production, use supervisord or a more robust entrypoint.
CMD export GUNICORN_CMD_ARGS="--workers ${GUNICORN_WORKERS:-4} --bind 127.0.0.1:8000 --timeout ${GUNICORN_TIMEOUT:-300} --log-level info -k uvicorn.workers.UvicornWorker" && \
    echo "Starting Gunicorn: gunicorn app.main:app $GUNICORN_CMD_ARGS" && \
    cd /app/app && gunicorn main:app $GUNICORN_CMD_ARGS & \
    echo "Starting Nginx..." && \
    nginx -g 'daemon off;'