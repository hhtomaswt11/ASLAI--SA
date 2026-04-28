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
    max_dynamic_frames: int = 60
    sequence_length: int = 30
    minimum_dynamic_frames: int = 5
    static_confidence_threshold: float = 0.75
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
        default=Path(__file__).resolve().parents[3] / "shared_models" / "pose_landmarker_lite.task"
    )
    pose_landmarker_task_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )

    @property
    def resolved_model_dir(self) -> Path:
        if self.model_artifacts_dir.exists():
            return self.model_artifacts_dir
        return self.legacy_model_artifacts_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
