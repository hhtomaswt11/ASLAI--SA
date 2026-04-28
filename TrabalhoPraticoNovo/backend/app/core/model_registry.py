from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from keras.models import load_model

from app.core.config import get_settings


@dataclass(slots=True)
class StaticArtifacts:
    classifier: Any | None
    scaler: Any | None
    label_encoder: Any | None


@dataclass(slots=True)
class DynamicArtifacts:
    classifier: Any | None
    label_encoder: Any | None
    norm_mean: np.ndarray | None
    norm_std: np.ndarray | None


@dataclass(slots=True)
class ModelRegistry:
    model_dir: Path
    static: StaticArtifacts
    dynamic: DynamicArtifacts

    @property
    def has_static_model(self) -> bool:
        return all(
            item is not None
            for item in (self.static.classifier, self.static.scaler, self.static.label_encoder)
        )

    @property
    def has_dynamic_model(self) -> bool:
        return all(
            item is not None
            for item in (
                self.dynamic.classifier,
                self.dynamic.label_encoder,
                self.dynamic.norm_mean,
                self.dynamic.norm_std,
            )
        )


def _safe_joblib_load(path: Path) -> Any | None:
    return joblib.load(path) if path.exists() else None


def _safe_numpy_load(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def _safe_keras_load(path: Path) -> Any | None:
    return load_model(path) if path.exists() else None


def build_model_registry() -> ModelRegistry:
    settings = get_settings()
    model_dir = settings.resolved_model_dir

    return ModelRegistry(
        model_dir=model_dir,
        static=StaticArtifacts(
            classifier=_safe_joblib_load(model_dir / "mlp_asl_landmarks.joblib"),
            scaler=_safe_joblib_load(model_dir / "scaler_asl_landmarks.joblib"),
            label_encoder=_safe_joblib_load(model_dir / "label_encoder_asl_landmarks.joblib"),
        ),
        dynamic=DynamicArtifacts(
            classifier=_safe_keras_load(model_dir / "lstm_wlasl100_best.keras"),
            label_encoder=_safe_joblib_load(model_dir / "label_encoder_wlasl100.joblib"),
            norm_mean=_safe_numpy_load(model_dir / "lstm_norm_mean.npy"),
            norm_std=_safe_numpy_load(model_dir / "lstm_norm_std.npy"),
        ),
    )
