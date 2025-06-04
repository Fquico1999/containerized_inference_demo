# Containerized HuggingFace Inference Demo

This repository provides a small FastAPI service that exposes HuggingFace model inference through a Dockerized application. It includes a Prometheus sidecar for metrics collection and a helper script to launch both the inference API and the Prometheus server.

## Overview
- **Language**: Python 3.9
- **Framework**: FastAPI with Gunicorn/Uvicorn workers
- **Metrics**: Prometheus via the `prometheus-client` package

The service dynamically loads HuggingFace pipelines on demand and caches them using an LRU cache. Metrics about requests, latencies and model load times are exposed for Prometheus scraping.

## API Endpoints
The API is defined in [`app/main.py`](app/main.py) and provides the following endpoints:

| Method & Path | Description |
| --- | --- |
| `GET /` | Returns basic information including the active device and cache status. |
| `POST /predict` | Performs inference given a model name, task and input payload. Supports batch inputs. |
| `GET /healthz` | Simple liveness check returning `{"status": "ok"}`. |
| `GET /readiness` | Ensures the default model can load; returns `{"status": "ready"}` when the service is ready. |
| `GET /metrics` | Exposes Prometheus metrics such as `hf_requests_total`. |
| `GET /cache_info` | Returns statistics about the internal LRU model cache. |

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
   The API will be available on `http://localhost:8000` and Prometheus on `http://localhost:9090`.

## Checking Prometheus
- View the Prometheus container logs:
  ```bash
  docker logs prometheus_server
  ```
- Open the web interface at [http://localhost:9090](http://localhost:9090) and run a query such as `hf_requests_total` to see the total number of inference requests collected from the API.

## Example Request
Once the containers are running you can test the API with `curl`:
```bash
curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"model_name": "distilbert-base-uncased-finetuned-sst-2-english", "task": "sentiment-analysis", "inputs": "I love this API!"}'
```

This will return the prediction and increment the Prometheus counters.
