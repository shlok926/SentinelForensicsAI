from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config.config import settings
from app.database.connection import get_db
from app.schemas.api_schemas import ReportResponse

router = APIRouter(prefix="/report", tags=["Digital Media Forensics Reports"])


@router.get("/{prediction_id}", response_model=ReportResponse)
def generate_forensic_report(prediction_id: int, db: Session = Depends(get_db)):
    """
    Generates and saves a detailed forensic PDF/JSON report compiling verification metrics.
    """
    # Architecture stub (call report generators)
    # 1. Fetch prediction details from DB
    # 2. Compile metrics, graphs and feature spectrogram visualizers into PDF
    # 3. Save PDF to settings.REPORTS_DIR

    mock_report_id = 505
    report_filename = f"report_prediction_{prediction_id}.pdf"

    return ReportResponse(
        success=True,
        prediction_id=prediction_id,
        report_id=mock_report_id,
        report_path=f"{settings.REPORTS_DIR}/{report_filename}",
        download_url=f"/static/reports/{report_filename}",
        created_at=datetime.now(timezone.utc),
    )
