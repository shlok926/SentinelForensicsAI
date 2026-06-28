from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.schemas.api_schemas import ModelListResponse, ModelRegistryItem

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("", response_model=ModelListResponse)
def get_available_models(db: Session = Depends(get_db)):
    """
    Returns lists of available deep learning models and checkpoints configured in the active registry.
    """
    # Architecture stub (Query model registry records from DB)

    models = [
        ModelRegistryItem(
            model_id=1,
            name="ResNet18_Face_CNN",
            version="1.0.0",
            status="active",
            accuracy=0.887,
            last_updated=datetime.now(timezone.utc),
        ),
        ModelRegistryItem(
            model_id=2,
            name="MelSpectrogram_2D_CNN",
            version="1.0.0",
            status="active",
            accuracy=0.912,
            last_updated=datetime.now(timezone.utc),
        ),
        ModelRegistryItem(
            model_id=3,
            name="Multimodal_Fusion_Forensics",
            version="1.0.0",
            status="active",
            accuracy=0.946,
            last_updated=datetime.now(timezone.utc),
        ),
    ]

    return ModelListResponse(success=True, available_models=models)
