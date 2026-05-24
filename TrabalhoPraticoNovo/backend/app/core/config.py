from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASLAI_", env_file=".env", extra="ignore")

    app_name: str = "ASLAI API"
    app_version: str = "0.1.0"
    api_prefix: str = "/api"
    enable_tts: bool = True
    enable_llm_correction: bool = False
    max_dynamic_frames: int = 64          # must match MAX_FRAMES used during training
    sequence_length: int = 64             # fallback if checkpoint lacks max_frames
    minimum_dynamic_frames: int = 10      # reject sequences shorter than this
    min_hand_frames_ratio: float = 0.20   # fraction of frames that must have hand detected
    dynamic_confidence_threshold: float = 0.15  # min confidence to emit a prediction
    static_confidence_threshold: float = 0.62
    model_artifacts_dir: Path = Field(
        default=Path(__file__).resolve().parents[3] / "shared_models"
    )
    legacy_model_artifacts_dir: Path = Field(
        default=Path(__file__).resolve().parents[4] / "TrabalhoPratico" / "models"
    )
    hand_landmarker_task_path: Path = Field(
        default=Path(__file__).resolve().parents[3] / "shared_models" / "hand_landmarker.task"
    )
    hand_landmarker_task_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    pose_landmarker_task_path: Path = Field(
        default=Path(__file__).resolve().parents[3] / "shared_models" / "pose_landmarker_full.task"
    )
    pose_landmarker_task_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    )

    @property
    def resolved_model_dir(self) -> Path:
        if self.model_artifacts_dir.exists():
            return self.model_artifacts_dir
        return self.legacy_model_artifacts_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
