from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter

from app.api.schemas import (
    DynamicPredictionRequest,
    DynamicPredictionResponse,
    FramePredictionRequest,
    HealthResponse,
    PhraseCorrectionRequest,
    PhraseCorrectionResponse,
    SpeechRequest,
    SpeechResponse,
    StaticPredictionResponse,
)
from app.core.config import get_settings
from app.core.model_registry import build_model_registry
from app.services.inference import InferenceService
from app.services.speech import speak_async
from app.services.text_processing import basic_spell_correct_phrase

router = APIRouter()


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService(build_model_registry())


@router.get("/health")
def healthcheck() -> HealthResponse:
    service = get_inference_service()
    return HealthResponse(**service.health_payload())


@router.post("/predict")
def predict_static(request: FramePredictionRequest) -> StaticPredictionResponse:
    service = get_inference_service()
    return StaticPredictionResponse(**service.predict_static(request.frame))


@router.post("/predict_dynamic")
def predict_dynamic(request: DynamicPredictionRequest) -> DynamicPredictionResponse:
    service = get_inference_service()
    result = service.predict_dynamic(request.frames)
    top_predictions = result.pop("top_predictions", [])
    return DynamicPredictionResponse(top_predictions=top_predictions, **result)


@router.post("/llm_correct")
def correct_phrase(request: PhraseCorrectionRequest) -> PhraseCorrectionResponse:
    corrected = basic_spell_correct_phrase(request.phrase)
    strategy = "local_rules_or_textblob"
    if corrected == request.phrase:
        strategy = "identity"
    return PhraseCorrectionResponse(
        original=request.phrase,
        corrected=corrected,
        strategy=strategy,
        metadata={"llm_enabled": get_settings().enable_llm_correction},
    )


@router.post("/speak")
def speak(request: SpeechRequest) -> SpeechResponse:
    if get_settings().enable_tts:
        speak_async(request.text)
    return SpeechResponse(success=True, text_spoken=request.text)
