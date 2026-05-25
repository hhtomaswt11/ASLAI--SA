from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StaticArtifacts:
    classifier: Any | None
    scaler: Any | None
    label_encoder: Any | None


@dataclass(slots=True)
class DynamicArtifacts:
    classifier: Any | None  # PyTorch checkpoint (dict)
    model: torch.nn.Module | None = None  # Instantiated model
    label_encoder: Any | None = None
    norm_mean: np.ndarray | None = None
    norm_std: np.ndarray | None = None
    pytorch_config: dict[str, Any] | None = None  # Model architecture config
    model_kind: str = "unknown"


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
        # norm_mean / norm_std are legacy LSTM artefacts — not used by the Transformer
        return all(
            item is not None
            for item in (
                self.dynamic.classifier,
                self.dynamic.model,
                self.dynamic.label_encoder,
            )
        )


def _safe_joblib_load(path: Path) -> Any | None:
    return joblib.load(path) if path.exists() else None


def _safe_numpy_load(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def _safe_pytorch_load(path: Path) -> dict[str, Any] | None:
    """Safely load PyTorch checkpoint and return the loaded state dict."""
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load PyTorch model from {path}: {e}")
        return None


def build_model_registry() -> ModelRegistry:
    settings = get_settings()
    model_dir = settings.resolved_model_dir

    # Load PyTorch dynamic model (can be LSTM or Transformer)
    dynamic_artifacts = _load_pytorch_model(model_dir)

    static_classifier = _safe_joblib_load(model_dir / "mlp_asl_landmarks.joblib")
    static_scaler = _safe_joblib_load(model_dir / "scaler_asl_landmarks.joblib")
    static_label_encoder = _safe_joblib_load(model_dir / "label_encoder_asl_landmarks.joblib")

    # Clean up feature names to avoid UserWarning when predicting with numpy arrays
    for obj in (static_classifier, static_scaler):
        if obj is not None and hasattr(obj, "feature_names_in_"):
            try:
                delattr(obj, "feature_names_in_")
            except AttributeError:
                pass

    return ModelRegistry(
        model_dir=model_dir,
        static=StaticArtifacts(
            classifier=static_classifier,
            scaler=static_scaler,
            label_encoder=static_label_encoder,
        ),
        dynamic=dynamic_artifacts,
    )


def _load_pytorch_model(model_dir: Path) -> DynamicArtifacts:
    """Load PyTorch dynamic model (LSTM or Transformer)."""
    # Try to load Transformer first
    checkpoint = _safe_pytorch_load(model_dir / "asl_transformer_v2_100_2.pt")
    model_kind = "transformer"
    
    if checkpoint is None:
        # Fall back to LSTM PyTorch model
        logger.info("Transformer model not found, trying LSTM")
        checkpoint = _safe_pytorch_load(model_dir / "asl_lstm_final.pt")
        model_kind = "lstm"
    
    if checkpoint is None:
        logger.error("No PyTorch dynamic model found (neither Transformer nor LSTM)")
        return DynamicArtifacts(
            classifier=None,
            label_encoder=None,
            norm_mean=None,
            norm_std=None,
            pytorch_config=None,
            model_kind="none",
        )

    # Extract metadata from checkpoint
    model_config = checkpoint.get("model_config", {})
    sign2idx = checkpoint.get("sign2idx", {})
    
    # Prioritize label encoder from checkpoint's sign2idx
    label_encoder = None
    if sign2idx:
        try:
            # Sort labels by their corresponding index value to ensure correct mapping
            sorted_labels = [label for label, _ in sorted(sign2idx.items(), key=lambda item: item[1])]
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(sorted_labels)
            logger.info(f"Created label encoder from checkpoint classes ({len(sorted_labels)} classes)")
        except Exception as e:
            logger.error(f"Failed to create label encoder from sign2idx: {e}")

    # Fallback to external joblib if sign2idx missing or failed
    if label_encoder is None:
        le_path = model_dir / "label_encoder_wlasl100.joblib"
        if le_path.exists():
            try:
                label_encoder = joblib.load(le_path)
                logger.info("Loaded label encoder from external joblib")
            except Exception as e:
                logger.warning(f"Could not load label encoder from file: {e}")

    # Build the model object
    # Determine device safely: torch.cuda.is_available() can return True even
    # when the GPU's compute capability is not supported by the installed PyTorch.
    def _probe_device() -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            (torch.zeros(1, device="cuda") + 1)
            return torch.device("cuda")
        except Exception:
            logger.warning("CUDA incompatible with this GPU; loading model on CPU.")
            return torch.device("cpu")

    device = _probe_device()

    input_dim = checkpoint.get("input_dim", 225)
    num_classes = checkpoint.get("num_classes", 50)
    
    try:
        from app.core.models import ASLLSTMModel, ASLTransformerModel
        if model_kind == "lstm":
            model = ASLLSTMModel(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=model_config.get("hidden_dim", model_config.get("hidden_size", 256)),
                num_layers=model_config.get("num_layers", 2),
                dropout=model_config.get("dropout", 0.3),
                bidirectional=model_config.get("bidirectional", True),
            )
        else:
            model = ASLTransformerModel(
                input_dim=input_dim,
                num_classes=num_classes,
                d_model=model_config.get("d_model", 256),
                nhead=model_config.get("nhead", 4),
                num_layers=model_config.get("num_layers", 6),
                dim_feedforward=model_config.get("dim_feedforward", 1024),
                dropout=model_config.get("dropout", 0.1),
                max_len=512,
            )
        
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if state_dict:
            model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        logger.info(f"Successfully built and loaded {model_kind} model on {device}")
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        model = None

    return DynamicArtifacts(
        classifier=checkpoint,
        model=model,
        label_encoder=label_encoder,
        norm_mean=_safe_numpy_load(model_dir / "lstm_norm_mean.npy"),
        norm_std=_safe_numpy_load(model_dir / "lstm_norm_std.npy"),
        pytorch_config=model_config,
        model_kind=model_kind,
    )
