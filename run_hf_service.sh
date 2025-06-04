#/usr/bin/bash

# Create cache folders
mkdir -p ./hf_cache
mkdir -p ./prometheus_metrics && chmod 777 ./prometheus_metrics/
# Create docker network to host both inference and Prometheus container
docker network create hf_inference_network

# Run Prometheus container
docker run -d --rm -p 9090:9090 \
    --network hf_inference_network \
    -v "$(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml" \
    --name prometheus_server prom/prometheus

# Run inference container
docker run --rm -p 8000:8000 \
    --network hf_inference_network \
    -v "$(pwd)/app:/app/app" \
    -v "$(pwd)/hf_cache:/app/.cache/huggingface" \
    -v "$(pwd)/prometheus_metrics:/app/prometheus_metrics" \
    --name hf_service hf-inference-service
