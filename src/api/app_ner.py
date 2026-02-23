from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root and src are on sys.path before importing internal modules.
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for p in (SRC_DIR, ROOT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from .dependencies_ner import get_ner_runtime_state, get_startup_mode, warmup_ner_service
from .routers.ie import router as ie_router
from .routers.ner import router as ner_router

VERSION_CODE = "3.0.0"
APP_DISPLAY_NAME = "sharifsetup-NER-v2.0.1"
STATIC_DIR = ROOT_DIR / "static"


def _parse_csv_env(value: str, *, fallback: list[str]) -> list[str]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or fallback


try:
    import orjson  # noqa: F401
    from fastapi.responses import ORJSONResponse

    DEFAULT_RESPONSE_CLASS = ORJSONResponse
except Exception:
    DEFAULT_RESPONSE_CLASS = JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_mode = get_startup_mode()
    logger.info("Startup: preparing NER service (mode=%s)", startup_mode)

    warmup_task: asyncio.Task | None = None
    if startup_mode == "background":
        warmup_task = asyncio.create_task(warmup_ner_service())
        app.state.warmup_task = warmup_task
        logger.info("Startup: warmup running in background")
    else:
        await warmup_ner_service()
        logger.info("Startup: NER service ready")

    try:
        yield
    finally:
        if warmup_task and not warmup_task.done():
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task
        logger.info("Shutdown: API stopping")


def create_app() -> FastAPI:
    docs_enabled = os.getenv("ENABLE_API_DOCS", "1").strip().lower() not in {"0", "false", "no", "off"}
    docs_url = "/docs" if docs_enabled else None
    redoc_url = "/redoc" if docs_enabled else None
    openapi_url = "/openapi.json" if docs_enabled else None

    app = FastAPI(
        title=APP_DISPLAY_NAME,
        description="Production-ready REST API for Named Entity Recognition and Information Extraction.",
        version=VERSION_CODE,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
        default_response_class=DEFAULT_RESPONSE_CLASS,
    )

    _configure_middlewares(app)
    _configure_routes(app)

    return app


def _configure_middlewares(app: FastAPI) -> None:
    gzip_minimum_size = int(os.getenv("GZIP_MINIMUM_SIZE", "1024"))
    app.add_middleware(GZipMiddleware, minimum_size=gzip_minimum_size)

    allowed_origins = _parse_csv_env(os.getenv("CORS_ALLOW_ORIGINS", "*"), fallback=["*"])
    allow_credentials = allowed_origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    trusted_hosts = _parse_csv_env(os.getenv("TRUSTED_HOSTS", "*"), fallback=["*"])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-MS"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return response


def _configure_routes(app: FastAPI) -> None:
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": APP_DISPLAY_NAME,
            "version": VERSION_CODE,
            "docs": "/docs",
            "health": "/health",
            "studio": "/app",
        }

    @app.get("/app", include_in_schema=False)
    async def studio_app():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return JSONResponse(status_code=404, content={"detail": "Web app UI not found"})
        return FileResponse(index_path)

    @app.get("/health")
    async def health():
        runtime = get_ner_runtime_state()
        status = runtime.get("status", "cold")

        if status == "failed":
            http_status = 503
            state = "error"
        elif status == "ready":
            http_status = 200
            state = "ok"
        else:
            http_status = 200
            state = "starting"

        return JSONResponse(
            status_code=http_status,
            content={
                "status": state,
                "ready": status == "ready",
                "runtime": runtime,
            },
        )

    @app.get("/health/ready")
    async def health_ready():
        runtime = get_ner_runtime_state()
        ready = runtime.get("status") == "ready"
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"ready": ready, "runtime": runtime},
        )

    app.include_router(ner_router)
    app.include_router(ie_router)


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "1").strip().lower() in {"1", "true", "yes", "on"}

    logger.info("Starting uvicorn server on %s:%d", host, port)
    uvicorn.run("api.app_ner:app", host=host, port=port, reload=reload_enabled, log_level="info")


__all__ = ["app", "create_app"]
