from datetime import datetime, timezone

import torch
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config.config import settings
from app.database.connection import get_db
from app.schemas.api_schemas import HealthResponse, VersionResponse

router = APIRouter(tags=["Diagnostics"])


@router.get("/health", response_model=HealthResponse)
def get_health(db: Session = Depends(get_db)):
    """
    Exposes diagnostics for container health monitoring (Kubernetes liveness/readiness probes).
    """
    database_connected = True
    try:
        # Perform quick execution probe
        db.execute(text("SELECT 1"))
    except Exception:
        database_connected = False

    cuda_available = torch.cuda.is_available()
    gpu_devices_count = torch.cuda.device_count() if cuda_available else 0

    return HealthResponse(
        status="healthy" if database_connected else "unhealthy",
        database_connected=database_connected,
        cuda_available=cuda_available,
        gpu_devices_count=gpu_devices_count,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/version", response_model=VersionResponse)
def get_version():
    """
    Returns platform versioning info.
    """
    return VersionResponse(version="1.0.0", api_prefix=settings.API_VERSION, environment=settings.ENV)
