from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database.connection import Base


class UserModel(Base):
    """
    User database model representing platform administrators and investigators.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="analyst")  # admin, analyst, guest
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    videos = relationship("VideoModel", back_populates="uploader")


class VideoModel(Base):
    """
    Video entity mapping uploaded media files.
    """

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_mb = Column(Float, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    sha256_hash = Column(String(64), unique=True, index=True, nullable=False)
    uploader_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    uploader = relationship("UserModel", back_populates="videos")
    predictions = relationship("PredictionModel", back_populates="video", cascade="all, delete-orphan")


class PredictionModel(Base):
    """
    Records deep learning results from the classification models.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    model_type = Column(String(50), nullable=False)  # video, audio, multimodal
    model_version = Column(String(50), nullable=False)  # e.g. v1.0.0
    prediction_score = Column(Float, nullable=False)  # Probability (0.0 to 1.0)
    is_fake = Column(Boolean, nullable=False)
    details_json = Column(Text, nullable=True)  # Full inference detail dict (JSON string)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    video = relationship("VideoModel", back_populates="predictions")
    report = relationship("ReportModel", uselist=False, back_populates="prediction", cascade="all, delete-orphan")


class ReportModel(Base):
    """
    Stores paths to generated digital forensics PDF or JSON audit files.
    """

    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    report_path = Column(String(512), nullable=False)
    generated_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    prediction = relationship("PredictionModel", back_populates="report")


class LogModel(Base):
    """
    DB database logger for tracking sensitive administrative operations or platform failures.
    """

    __tablename__ = "platform_logs"

    id = Column(Integer, primary_key=True, index=True)
    severity = Column(String(20), nullable=False)  # INFO, WARNING, ERROR
    logger_name = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    details_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelRegistryModel(Base):
    """
    Maintains status of loaded deep learning checkpoints in the active inference engine.
    """

    __tablename__ = "models_registry"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)  # e.g. ResNet18_Face_CNN
    version = Column(String(50), nullable=False)
    status = Column(String(30), default="inactive")  # active, inactive, testing
    accuracy = Column(Float, nullable=True)
    path = Column(String(512), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DatasetRegistryModel(Base):
    """
    Database registry tracking local and downloaded training dataset configurations.
    """

    __tablename__ = "datasets_registry"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    version = Column(String(50), nullable=False)
    samples_count = Column(Integer, default=0)
    local_path = Column(String(512), nullable=True)
    registry_status = Column(String(30), default="unverified")  # verified, unverified, downloading
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
