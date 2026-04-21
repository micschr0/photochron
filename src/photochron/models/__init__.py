"""
Pydantic models for PhotoChron data tables and AI model clients.

These models represent the structured data stored in the SQLite Feature Store
and are used for validation, serialization, and type-safe data handling.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Ollama client for vision LLM integration
from .ollama_client import (
    ContextAnalysisResult,
    ModelType,
    OllamaClient,
    OllamaConfig,
    get_ollama_client,
)


class PhotoBase(BaseModel):
    """Base model for photo metadata."""

    content_hash: str = Field(..., description="MD5 hash of file content")
    file_path: str = Field(..., description="Original file path")
    downsample_path: str | None = Field(None, description="Path to downsampled thumbnail")
    exif_datetime: str | None = Field(None, description="EXIF DateTimeOriginal if present")
    make: str | None = Field(None, description="Camera make")
    model: str | None = Field(None, description="Camera model")
    perceptual_hash: str | None = Field(None, description="Perceptual hash for near-duplicate detection")


class PhotoCreate(PhotoBase):
    """Model for creating a new photo record."""

    pass


class Photo(PhotoBase):
    """Complete photo model with database fields."""

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PersonBase(BaseModel):
    """Base model for person information."""

    person_id: str = Field(..., description="User-defined ID (e.g., 'person_mama')")
    name: str = Field(..., description="Display name")
    birthday: str | None = Field(None, description="YYYY-MM-DD format")


class PersonCreate(PersonBase):
    """Model for creating a new person record."""

    pass


class Person(PersonBase):
    """Complete person model with database fields."""

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FaceBase(BaseModel):
    """Base model for face detection results."""

    photo_id: int = Field(..., description="Reference to photo")
    person_id: int | None = Field(None, description="Reference to person if known")
    embedding: bytes | None = Field(None, description="Face embedding vector")
    age_estimate: float | None = Field(None, description="Estimated age in years")
    age_std: float | None = Field(None, description="Standard deviation of age estimate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence 0.0-1.0")
    bbox_x1: float = Field(..., description="Bounding box coordinate")
    bbox_y1: float = Field(..., description="Bounding box coordinate")
    bbox_x2: float = Field(..., description="Bounding box coordinate")
    bbox_y2: float = Field(..., description="Bounding box coordinate")


class FaceCreate(FaceBase):
    """Model for creating a new face record."""

    pass


class Face(FaceBase):
    """Complete face model with database fields."""

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ContextBase(BaseModel):
    """Base model for LLM context analysis."""

    photo_id: int = Field(..., description="Reference to photo")
    decade: str | None = Field(None, description="Estimated decade range (e.g., '1985-1990')")
    decade_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    season: str | None = Field(None, description="'spring', 'summer', 'autumn', 'winter'")
    season_confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence in season estimate (0.0-1.0)")
    event_hint: str | None = Field(None, description="Event hint from LLM")
    event_confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence in event hint (0.0-1.0)")
    photo_medium: str = Field(..., description="'print_scan', 'digital', 'polaroid', etc.")
    photo_medium_confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in photo medium estimate (0.0-1.0)",
    )
    visual_evidence: list[str] | None = Field(
        None, description="List of specific visual cues that informed the analysis"
    )
    alternative_decades: list[str] | None = Field(None, description="Alternative decade possibilities when uncertain")
    uncertainty_flag: bool | None = Field(None, description="Flag indicating high uncertainty in analysis")
    hypothesis_notes: str | None = Field(None, description="Explanation when multiple hypotheses exist")
    raw_json: str = Field(..., description="Full LLM response JSON")


class ContextCreate(ContextBase):
    """Model for creating a new context record."""

    pass


class Context(ContextBase):
    """Complete context model with database fields."""

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RankingBase(BaseModel):
    """Base model for chronological ranking results."""

    photo_id: int = Field(..., description="Reference to photo")
    sort_rank: int = Field(..., ge=0, description="Final chronological rank (0-based)")
    estimated_year: int | None = Field(None, description="Estimated year")
    estimated_month: int | None = Field(None, ge=1, le=12, description="Estimated month if known (1-12)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence 0.0-1.0")
    review_needed: bool = Field(False, description="Flag for low confidence results")
    ranking_json: str = Field(..., description="Full ranking details JSON")


class RankingCreate(RankingBase):
    """Model for creating a new ranking record."""

    pass


class Ranking(RankingBase):
    """Complete ranking model with database fields."""

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PipelineRunBase(BaseModel):
    """Base model for pipeline execution tracking."""

    run_id: str = Field(..., description="Unique identifier for this run")
    schema_version: int = Field(1, description="Database schema version")
    config_hash: str = Field(..., description="Hash of config used")
    insightface_version: str | None = Field(None, description="Model version for cache invalidation")
    ollama_version: str | None = Field(None, description="Model version for cache invalidation")
    start_time: datetime = Field(..., description="Pipeline start timestamp")
    end_time: datetime | None = Field(None, description="Pipeline end timestamp")
    status: str = Field(..., description="'running', 'completed', 'failed'")
    photos_processed: int = Field(0, description="Number of photos processed")


class PipelineRunCreate(PipelineRunBase):
    """Model for creating a new pipeline run record."""

    pass


class PipelineRun(PipelineRunBase):
    """Complete pipeline run model with database fields."""

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
