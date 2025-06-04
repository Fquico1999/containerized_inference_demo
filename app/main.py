from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline
import logging
import os
import torch
from functools import lru_cache
from typing import Union, List, Dict, Any
from collections import defaultdict
import time

# Prometheus metrics
from prometheus_client import Counter, Histogram, CollectorRegistry, multiprocess, generate_latest, REGISTRY

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - PID:%(process)d - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dynamic HuggingFace Inference API",
    description="An API to get inferences from any HuggingFace model, supporting batch requests and parallel processing.",
    version="1.0.0"
)

# --- Prometheus Metrics Initialization ---
prometheus_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
if prometheus_dir:
    if not os.path.exists(prometheus_dir):
        os.makedirs(prometheus_dir, exist_ok=True)
    logger.info(f"Prometheus multiprocess mode enabled, directory: {prometheus_dir}")
else:
    logger.warning("PROMETHEUS_MULTIPROC_DIR not set. Prometheus metrics may not aggregate correctly across workers in multiprocess environments like Gunicorn.")

REQUEST_COUNT = Counter(
    'hf_requests_total',
    'Total number of inference requests',
    ['model_name', 'task', 'worker_pid', 'http_status']
)
INFERENCE_LATENCY = Histogram(
    'hf_inference_latency_seconds',
    'Inference latency in seconds for the pipeline execution itself',
    ['model_name', 'task']
)
MODEL_LOAD_TIME = Histogram(
    'hf_model_load_time_seconds',
    'Time taken to load/retrieve a model pipeline object',
    ['model_name', 'task']
)
REQUEST_TOTAL_LATENCY = Histogram(
    'hf_request_total_latency_seconds',
    'Total request handling latency in seconds',
    ['model_name', 'task']
)


# --- Model Loading & Caching ---
device_id = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device_id == 0 else "CPU"
logger.info(f"PID: {os.getpid()} - Using device: {device_name} (id: {device_id})")

@lru_cache(maxsize=10)
def get_pipeline_cached(model_name: str, task: str, device: int) -> Pipeline:
    logger.info(f"PID: {os.getpid()} - LRU Cache: Attempting to load/retrieve pipeline for model: '{model_name}', task: '{task}' on device_id: {device}")
    _load_start_time = time.perf_counter()
    try:
        pipe = pipeline(task=task, model=model_name, device=device)
        _load_duration = time.perf_counter() - _load_start_time
        MODEL_LOAD_TIME.labels(model_name=model_name, task=task).observe(_load_duration)
        logger.info(f"PID: {os.getpid()} - LRU Cache: Successfully loaded/retrieved pipeline for model: '{model_name}', task: '{task}' in {_load_duration:.2f}s")
        return pipe
    except Exception as e:
        logger.error(f"PID: {os.getpid()} - LRU Cache: Failed to load pipeline for model: '{model_name}', task: '{task}': {e}")
        # Re-raise as ValueError to be caught by endpoint logic if needed
        raise ValueError(f"Failed to load pipeline for {model_name} (task: {task}): {e}")


DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_TASK = "sentiment-analysis"

@app.on_event("startup")
async def startup_event():
    logger.info(f"PID: {os.getpid()} - Application startup: Preloading default model if specified.")
    if DEFAULT_MODEL_NAME and DEFAULT_TASK:
        try:
            get_pipeline_cached(model_name=DEFAULT_MODEL_NAME, task=DEFAULT_TASK, device=device_id)
            logger.info(f"PID: {os.getpid()} - Default model '{DEFAULT_MODEL_NAME}' for task '{DEFAULT_TASK}' preloading initiated.")
        except Exception as e:
            logger.error(f"PID: {os.getpid()} - Failed to preload default model '{DEFAULT_MODEL_NAME}': {e}")
    logger.info(f"PID: {os.getpid()} - Application startup complete. API is ready.")

# --- Pydantic Models ---
class InferenceRequest(BaseModel):
    model_name: str = Field(..., example="distilbert-base-uncased-finetuned-sst-2-english", description="Name of the HuggingFace model from the hub.")
    task: str = Field(..., example="sentiment-analysis", description="The task for the pipeline (e.g., 'sentiment-analysis', 'text-generation', 'summarization').")
    inputs: Union[str, List[str]] = Field(..., example="I love this API!", description="A single string or a list of strings for batch processing.")
    pipeline_kwargs: Dict[str, Any] = Field({}, example={"max_length": 50}, description="Optional keyword arguments to pass to the pipeline call.")

# --- API Endpoints ---
@app.get("/", summary="Root endpoint", description="Basic API information.")
async def root():
    # Check cache info for default model status (heuristic)
    default_model_status = "Not configured for preload"
    if DEFAULT_MODEL_NAME and DEFAULT_TASK:
        info = get_pipeline_cached.cache_info()
        # If current size > 0, it means *something* is cached. The default model was attempted.
        if info.currsize > 0:
             default_model_status = "Preload attempted, cache populated"
        else:
             default_model_status = "Preload attempted, cache appears empty"

    return {
        "message": "HuggingFace Dynamic Inference API",
        "active_device": device_name,
        "default_model_status": default_model_status,
        "current_cache_size": get_pipeline_cached.cache_info().currsize
    }

@app.post("/predict", summary="Perform inference", description="Runs inference using a specified HuggingFace model and task, supports batch inputs.")
async def predict(request: InferenceRequest):
    worker_pid = os.getpid()
    request_received_time = time.perf_counter()

    # Get cache stats *before* calling the cached function
    cache_info_before = get_pipeline_cached.cache_info()

    try:
        # This call will either hit the cache or load the model
        pipe = get_pipeline_cached(
            model_name=request.model_name,
            task=request.task,
            device=device_id
        )

        # Get cache stats *after* calling the cached function
        cache_info_after = get_pipeline_cached.cache_info()

        # Determine if this call resulted in an LRU cache hit for the pipeline object
        pipeline_from_lru_cache = cache_info_after.hits > cache_info_before.hits
        logger.info(f"PID: {worker_pid} - Processing request for model '{request.model_name}', task '{request.task}'. Pipeline from LRU cache: {pipeline_from_lru_cache}")
        
        inference_start_time = time.perf_counter()
        results = pipe(request.inputs, **request.pipeline_kwargs)
        inference_duration_ms = (time.perf_counter() - inference_start_time) * 1000
        
        total_request_duration_ms = (time.perf_counter() - request_received_time) * 1000
        
        INFERENCE_LATENCY.labels(model_name=request.model_name, task=request.task).observe(inference_duration_ms / 1000.0)
        REQUEST_TOTAL_LATENCY.labels(model_name=request.model_name, task=request.task).observe(total_request_duration_ms / 1000.0)
        REQUEST_COUNT.labels(model_name=request.model_name, task=request.task, worker_pid=str(worker_pid), http_status='200').inc()
        
        logger.info(f"PID: {worker_pid} - Prediction successful for '{request.model_name}'. Inference time: {inference_duration_ms:.2f}ms, Total request time: {total_request_duration_ms:.2f}ms")
        
        return {
            "model_name": request.model_name,
            "task": request.task,
            "predictions": results,
            "worker_pid": worker_pid,
            "total_request_time_ms": round(total_request_duration_ms, 2),
            "inference_execution_time_ms": round(inference_duration_ms, 2),
            "pipeline_from_lru_cache": pipeline_from_lru_cache
        }

    except ValueError as ve: # Model loading failed within get_pipeline_cached
        total_request_duration_ms = (time.perf_counter() - request_received_time) * 1000
        REQUEST_TOTAL_LATENCY.labels(model_name=request.model_name, task=request.task).observe(total_request_duration_ms / 1000.0) # Also log latency for failed requests
        REQUEST_COUNT.labels(model_name=request.model_name, task=request.task, worker_pid=str(worker_pid), http_status='400').inc()
        logger.error(f"PID: {worker_pid} - Value error for model '{request.model_name}': {ve}. Total request time: {total_request_duration_ms:.2f}ms")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        total_request_duration_ms = (time.perf_counter() - request_received_time) * 1000
        REQUEST_TOTAL_LATENCY.labels(model_name=request.model_name, task=request.task).observe(total_request_duration_ms / 1000.0)
        REQUEST_COUNT.labels(model_name=request.model_name, task=request.task, worker_pid=str(worker_pid), http_status='500').inc()
        logger.error(f"PID: {worker_pid} - Prediction error for model '{request.model_name}': {e}. Total request time: {total_request_duration_ms:.2f}ms")
        logger.exception("Full stack trace:")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction with model '{request.model_name}': {str(e)}")

@app.get("/healthz", summary="Health check", description="Returns 'ok' if the API server is running.")
async def healthz():
    return {"status": "ok", "pid": os.getpid()}

@app.get("/readiness", summary="Readiness check", description="Returns 'ready' if the API can serve requests (e.g., default model loaded).")
async def readiness():
    try:
        if DEFAULT_MODEL_NAME and DEFAULT_TASK:
            # Attempt to get the default model pipeline (from cache if preloaded)
            get_pipeline_cached(model_name=DEFAULT_MODEL_NAME, task=DEFAULT_TASK, device=device_id)
            # To be truly ready, we should ensure it didn't throw an error.
            return {"status": "ready", "message": f"Default model '{DEFAULT_MODEL_NAME}' accessible/loadable.", "pid": os.getpid()}
        else:
            return {"status": "ready", "message": "API is operational (no default model specified for preload).", "pid": os.getpid()}
    except Exception as e:
        logger.warning(f"Readiness check failed: Default model not ready or error: {e}")
        raise HTTPException(status_code=503, detail=f"Service not fully ready. Default model issue: {str(e)}")

@app.get("/metrics", summary="Prometheus metrics", description="Exposes Prometheus-compatible metrics.")
async def metrics():
    registry_to_generate = REGISTRY
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        temp_registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(temp_registry)
        registry_to_generate = temp_registry
    return Response(generate_latest(registry_to_generate), media_type="text/plain; version=0.0.4; charset=utf-8")

@app.get("/cache_info", summary="LRU Cache Info", description="Information about the loaded model pipeline cache.")
async def cache_info_endpoint():
    # get_pipeline_cached.cache_info() is the public API.
    return {
        "pid": os.getpid(),
        "lru_cache_stats": str(get_pipeline_cached.cache_info())
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run(app, host="0.0.0.0", port=8000)