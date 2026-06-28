import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from app.config.config import settings
from app.database.connection import Base, engine

# Import Routers
from app.routes import agent, health, history, models, predict, report, upload
from app.utils.bootstrap import bootstrap_directories
from app.utils.exceptions import ForensicsPlatformException
from app.utils.logging import get_request_logger, get_system_logger, setup_logger

# Setup specialized loggers
system_logger = get_system_logger()
request_logger = get_request_logger()

# 1. Bootstrap directories on startup
bootstrap_directories()

# 2. Initialize DB Tables (SQLite / Postgres)
try:
    Base.metadata.create_all(bind=engine)
    system_logger.info("Database schemas initialized / verified.")
except Exception as e:
    system_logger.critical(f"Failed to bootstrap database schemas: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    system_logger.info(f"Platform successfully started in [{settings.ENV}] environment.")
    yield


# 2. Setup ASGI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise-grade AI Digital Media Forensics & Verification API Platform.",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# 3. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_hosts_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Integrate Prometheus Metrics ASGI sub-app
if settings.PROMETHEUS_METRICS_ENABLED:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Custom latency/request counters
    HTTP_REQUESTS_TOTAL = Counter("http_requests_total", "Total HTTP requests count", ["method", "endpoint", "status"])
    HTTP_REQUEST_DURATION = Histogram(
        "http_request_duration_seconds", "HTTP request processing latency", ["method", "endpoint"]
    )


# 6. Global Request Logging and Metrics Middleware
@app.middleware("http")
async def log_requests_and_metrics(request: Request, call_next):
    start_time = time.time()
    method = request.method
    path = request.url.path

    # Log incoming request details
    request_logger.info(
        f"Incoming: {method} {path} from Client: {request.client.host if request.client else 'Unknown'}"
    )

    response = await call_next(request)

    duration = time.time() - start_time
    status_code = response.status_code

    # Log complete transaction details
    request_logger.info(f"Outgoing: {method} {path} - Status: {status_code} - Duration: {duration:.4f}s")

    # Record metrics
    if settings.PROMETHEUS_METRICS_ENABLED:
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=path, status=status_code).inc()
        HTTP_REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)

    return response


# 7. Global Exception Handler mapping custom ForensicsPlatformException to API contracts
@app.exception_handler(ForensicsPlatformException)
async def custom_exception_handler(request: Request, exc: ForensicsPlatformException):
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


# 8. Catch-all unexpected internal server error mapper
@app.exception_handler(Exception)
async def fallback_exception_handler(request: Request, exc: Exception):
    system_logger.error(f"Unhandled critical system exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "ERR_INTERNAL_SERVER_ERROR",
                "description": "An unexpected critical system failure occurred.",
                "solution": "Review server logs in storage/logs/errors.log or contact site administrator.",
                "details": {"system_message": str(exc)},
            },
        },
    )


# 9. Register Routers
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(predict.router)
app.include_router(report.router)
app.include_router(history.router)
app.include_router(models.router)
app.include_router(agent.router)
