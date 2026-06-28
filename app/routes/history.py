from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.schemas.api_schemas import HistoryListResponse, PredictionHistoryItem

router = APIRouter(prefix="/history", tags=["Inference History Logs"])


@router.get("", response_model=HistoryListResponse)
def get_prediction_history(
    limit: int = Query(default=10, ge=1, le=100), offset: int = Query(default=0, ge=0), db: Session = Depends(get_db)
):
    """
    Retrieves history logs of previously executed predictions and analyses.
    """
    # Architecture stub (Query predictions joined with videos table)

    mock_item_1 = PredictionHistoryItem(
        prediction_id=1,
        video_name="manipulated_speech.mp4",
        model_type="multimodal",
        prediction_score=0.985,
        is_fake=True,
        created_at=datetime.now(timezone.utc),
    )
    mock_item_2 = PredictionHistoryItem(
        prediction_id=2,
        video_name="interview_clean.mp4",
        model_type="video",
        prediction_score=0.012,
        is_fake=False,
        created_at=datetime.now(timezone.utc),
    )

    return HistoryListResponse(success=True, total_records=2, data=[mock_item_1, mock_item_2])
