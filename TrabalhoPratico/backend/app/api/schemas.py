from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    app_name: str
    version: str
    static_model_loaded: bool
    dynamic_model_loaded: bool
    model_dir: str
    mediapipe_version: str


class FramePredictionRequest(BaseModel):
    frame: str = Field(..., min_length=16, description="Base64-encoded image frame")


class DynamicPredictionRequest(BaseModel):
    frames: list[str] = Field(..., min_length=1, max_length=64)


class StaticPredictionResponse(BaseModel):
    letter: str | None
    confidence: float
    hand_detected: bool
    annotated_frame_b64: str | None = None
    backend: str = "python-mediapipe"


class RankedPrediction(BaseModel):
    label: str
    confidence: float


class DynamicPredictionResponse(BaseModel):
    word: str | None
    confidence: float
    body_detected: bool
    top_predictions: list[RankedPrediction] = Field(default_factory=list)
    annotated_frame_b64: str | None = None
    backend: str = "python-mediapipe"


class PhraseCorrectionRequest(BaseModel):
    phrase: str = Field(..., min_length=1)


class PhraseCorrectionResponse(BaseModel):
    original: str
    corrected: str
    strategy: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeechRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SpeechResponse(BaseModel):
    success: bool
    text_spoken: str
