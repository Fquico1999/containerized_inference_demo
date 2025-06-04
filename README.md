# Containerized HuggingFace Inference Demo

This repository provides a small FastAPI service that exposes HuggingFace model
inference through a Dockerized application. Requests are served via an Nginx
front end which proxies calls under the `/api` path to the FastAPI application
and serves a basic HTML/JS playground from the root path. It also includes a
Prometheus sidecar for metrics collection and a helper script to launch both the
inference API and the Prometheus server.

## Overview
- **Language**: Python 3.9
- **Framework**: FastAPI with Gunicorn/Uvicorn workers
- **Metrics**: Prometheus via the `prometheus-client` package

The service dynamically loads HuggingFace pipelines on demand and caches them using an LRU cache. Metrics about requests, latencies and model load times are exposed for Prometheus scraping.

## API Endpoints
The API is defined in [`app/main.py`](app/main.py) and provides the following endpoints:

| Method & Path | Description |
| --- | --- |
| `GET /api/` | Returns basic information including the active device and cache status. |
| `POST /api/predict` | Performs inference given a model name, task and input payload. Supports batch inputs. |
| `GET /api/healthz` | Simple liveness check returning `{"status": "ok"}`. |
| `GET /api/readiness` | Ensures the default model can load; returns `{"status": "ready"}` when the service is ready. |
| `GET /api/metrics` | Exposes Prometheus metrics such as `hf_requests_total`. |
| `GET /api/cache_info` | Returns statistics about the internal LRU model cache. |

## Building and Running with Docker
1. **Build the image**
   ```bash
   docker build -t hf-inference-service .
   ```

2. **Launch the service and Prometheus**
   Use the provided script which creates a network, starts Prometheus and then runs the API container:
   ```bash
   ./run_hf_service.sh
   ```
   The API will be available on `http://localhost:8080/api` and Prometheus on `http://localhost:9090`.
   A small web playground is served from `http://localhost:8080/`.

## Checking Prometheus
- View the Prometheus container logs:
  ```bash
  docker logs prometheus_server
  ```
- Open the web interface at [http://localhost:9090](http://localhost:9090) and run a query such as `hf_requests_total` to see the total number of inference requests collected from the API.

## Example Request
Once the containers are running you can test the API with `curl`:
```bash
curl -X POST http://localhost:8080/api/predict \
     -H 'Content-Type: application/json' \
     -d '{"model_name": "distilbert-base-uncased-finetuned-sst-2-english", "task": "sentiment-analysis", "inputs": "I love this API!"}'
```

This will return the prediction and increment the Prometheus counters.
