# Sharifsetup NER

High-performance, production-ready NER/IE service built with FastAPI and GLiNER2.

## Simplified Architecture

```text
configs/
  ner.yaml                        # Runtime config (model + pipeline + limits)

src/
  api/
    app_ner.py                    # FastAPI app factory + middleware + health/readiness
    dependencies_ner.py           # Service singleton + warmup/runtime state
    routers/
      ner.py                      # /ner endpoints
      ie.py                       # /ie endpoints

  engines/nlp/ner/
    service.py                    # Core orchestration (inference, batching, normalization)
    gliner_model.py               # GLiNER2 loader/runtime wrapper
    interfaces.py                 # Core datatypes/contracts
    model.py                      # Backward-compatibility re-export layer
    pipeline/                     # Enrichment / QA / conflict resolution helpers
    utils/                        # Normalization / offsets / semantic resolution

  core/
    usage.py                      # O(1) sliding-window rate usage tracker
```

## Production Features
- App factory pattern (`create_app`) for clean deployment and testing.
- Startup model warmup with status lifecycle: `cold -> warming -> ready/failed`.
- Readiness endpoints:
  - `GET /health`
  - `GET /health/ready`
- Performance middleware:
  - GZip compression
  - Process time header (`X-Process-Time-MS`)
  - Trusted host middleware
  - CORS middleware
- Local-first/offline model loading with optional revision pinning.
- Concurrency controls and batch inference for stable throughput.

## Key Performance Controls (`configs/ner.yaml`)
- `pipeline.request_concurrency_limit`
- `pipeline.inference_concurrency_limit`
- `pipeline.performance.batch_size`
- `pipeline.performance.batch_max_chars`
- `pipeline.performance.max_inflight_batches`
- `pipeline.rate_limit.chars_per_window`
- `pipeline.rate_limit.window_seconds`
- `system.local_model_dir` (set to `sharifsetup_NER`, loaded strictly from this directory)
- `system.local_files_only`
- `system.offline_mode`
- `system.warmup_on_load`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 src/api/app_ner.py
```

Open:
- Web app: `http://localhost:8000/app`
- API docs: `http://localhost:8000/docs`

## Production Run (Recommended)

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export RELOAD=0
export ENABLE_API_DOCS=0
uvicorn api.app_ner:app --host 0.0.0.0 --port 8000 --workers 2
```

## Docker

```bash
docker build -t ie-platform -f docker/Dockerfile.ner .
docker run --gpus all -p 8000:8000 ie-platform
```
