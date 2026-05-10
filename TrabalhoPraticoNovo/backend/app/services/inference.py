from __future__ import annotations

import logging
import math
import threading
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import requests
import torch
import torch.nn as nn

from app.core.config import get_settings
from app.core.model_registry import ModelRegistry
from app.utils.image import decode_base64_frame, encode_frame

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """Positional encoding sinusoidal (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ASLTransformerModel(nn.Module):
    """Transformer Encoder for ASL sign recognition."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, frames, features)
        B, T, _ = x.shape

        x = self.input_proj(x)
        x = self.pos_enc(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if pad_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            pad_mask = torch.cat([cls_mask, pad_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        cls_out = x[:, 0]

        return self.classifier(cls_out)


class ASLLSTMModel(nn.Module):
    """BiLSTM baseline for ASL sign recognition."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.num_dirs
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        
        if pad_mask is not None:
            # Calculate lengths for packing
            lengths = (~pad_mask).sum(dim=1).cpu().int()
            # Ensure lengths are at least 1 to avoid crash
            lengths = torch.clamp(lengths, min=1)
            
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden_state, _) = self.lstm(x_packed)
        else:
            _, (hidden_state, _) = self.lstm(x)

        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden_dim)
            # Take the last layer's forward and backward states
            h_fwd = hidden_state[-2]
            h_bwd = hidden_state[-1]
            features = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            features = hidden_state[-1]

        return self.classifier(features)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class InferenceService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.settings = get_settings()
        self._static_detector = self._build_static_detector()
        self._dynamic_pose_detector = self._build_dynamic_pose_detector()
        self._dynamic_hand_detector = self._build_hand_detector(num_hands=2)
        self._static_detector_lock = threading.Lock()
        self._dynamic_pose_detector_lock = threading.Lock()
        self._dynamic_hand_detector_lock = threading.Lock()

    def health_payload(self) -> dict[str, Any]:
        return {
            "app_name": self.settings.app_name,
            "version": self.settings.app_version,
            "static_model_loaded": self.registry.has_static_model,
            "dynamic_model_loaded": self.registry.has_dynamic_model,
            "model_dir": str(self.registry.model_dir),
            "mediapipe_version": mp.__version__,
        }

    def predict_static(self, frame_b64: str) -> dict[str, Any]:
        frame_bgr = decode_base64_frame(frame_b64)
        if not self.registry.has_static_model or self._static_detector is None:
            return {"letter": None, "confidence": 0.0, "hand_detected": False, "annotated_frame_b64": None}

        letter: str | None = None
        confidence = 0.0
        hand_detected = False
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        with self._static_detector_lock:
            result = self._static_detector.detect(mp_image)

        if result.hand_landmarks:
            hand_detected = True
            first_hand = result.hand_landmarks[0]
            self._draw_task_hand_landmarks(frame_bgr, first_hand)

            features = self._normalize_static_landmarks(first_hand)
            if features is not None:
                arr = np.array(features, dtype=np.float32).reshape(1, -1)
                arr_scaled = self.registry.static.scaler.transform(arr)
                pred = self.registry.static.classifier.predict(arr_scaled)
                proba = self.registry.static.classifier.predict_proba(arr_scaled).max()
                letter = self.registry.static.label_encoder.inverse_transform(pred)[0]
                confidence = float(proba)

        return {
            "letter": letter,
            "confidence": confidence,
            "hand_detected": hand_detected,
            "annotated_frame_b64": encode_frame(frame_bgr),
        }

    def _build_static_detector(self) -> Any | None:
        if not self.registry.has_static_model:
            return None

        return self._build_hand_detector(num_hands=1)

    def _build_hand_detector(self, num_hands: int) -> Any | None:
        if not hasattr(mp, "tasks"):
            logger.warning("MediaPipe tasks API unavailable; static detector disabled")
            return None

        model_path = self._ensure_hand_landmarker_model()
        if model_path is None:
            logger.warning("Hand landmarker model unavailable; static detector disabled")
            return None

        try:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception:
            logger.exception("Failed to initialize HandLandmarker")
            return None

    def _build_dynamic_pose_detector(self) -> Any | None:
        if not self.registry.has_dynamic_model:
            return None

        if not hasattr(mp, "tasks"):
            logger.warning("MediaPipe tasks API unavailable; dynamic pose detector disabled")
            return None

        model_path = self._ensure_pose_landmarker_model()
        if model_path is None:
            logger.warning("Pose landmarker model unavailable; dynamic detector disabled")
            return None

        try:
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            return mp.tasks.vision.PoseLandmarker.create_from_options(options)
        except Exception:
            logger.exception("Failed to initialize PoseLandmarker")
            return None

    def _ensure_hand_landmarker_model(self) -> Any | None:
        model_path = self.settings.hand_landmarker_task_path.expanduser().resolve()
        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(self.settings.hand_landmarker_task_url, timeout=30)
            response.raise_for_status()
            model_path.write_bytes(response.content)
            return model_path
        except Exception:
            logger.exception("Failed to download hand_landmarker.task")
            return None

    def _ensure_pose_landmarker_model(self) -> Any | None:
        model_path = self.settings.pose_landmarker_task_path.expanduser().resolve()
        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(self.settings.pose_landmarker_task_url, timeout=30)
            response.raise_for_status()
            model_path.write_bytes(response.content)
            return model_path
        except Exception:
            logger.exception("Failed to download pose_landmarker.task")
            return None

    def _draw_task_hand_landmarks(self, frame_bgr: np.ndarray, landmarks: Any) -> None:
        height, width = frame_bgr.shape[:2]
        points: list[tuple[int, int]] = []

        for lm in landmarks:
            x = int(np.clip(lm.x * width, 0, width - 1))
            y = int(np.clip(lm.y * height, 0, height - 1))
            points.append((x, y))
            cv2.circle(frame_bgr, (x, y), 3, (0, 255, 120), thickness=-1)

        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame_bgr, points[start_idx], points[end_idx], (255, 255, 255), 2)

    def predict_dynamic(self, frames_b64: list[str]) -> dict[str, Any]:
        if (
            not self.registry.has_dynamic_model
            or self._dynamic_pose_detector is None
            or self._dynamic_hand_detector is None
        ):
            return {
                "word": None,
                "confidence": 0.0,
                "body_detected": False,
                "top_predictions": [],
                "annotated_frame_b64": None,
            }

        frame_payloads = frames_b64[: self.settings.max_dynamic_frames]
        if len(frame_payloads) < self.settings.minimum_dynamic_frames:
            return {
                "word": None,
                "confidence": 0.0,
                "body_detected": False,
                "top_predictions": [],
                "annotated_frame_b64": None,
            }

        vectors: list[np.ndarray] = []
        body_detected = False
        annotated_frame: np.ndarray | None = None

        for frame_b64 in frame_payloads:
            frame_bgr = decode_base64_frame(frame_b64)
            vector, frame_has_body, annotated = self._extract_dynamic_vector(frame_bgr)
            vectors.append(vector)
            body_detected = body_detected or frame_has_body
            annotated_frame = annotated

        # 4. Prepare sequence (crop if too long)
        checkpoint = self.registry.dynamic.classifier
        target_len = checkpoint.get("max_frames", self.settings.sequence_length) if isinstance(checkpoint, dict) else self.settings.sequence_length
        
        sequence = np.stack(vectors, axis=0) if vectors else np.zeros((1, 225), dtype=np.float32)
        if sequence.shape[0] > target_len:
            sequence = sequence[:target_len]
        
        actual_len = sequence.shape[0]

        # 5. Normalize BEFORE padding (matches notebook pipeline)
        normalized_seq = self._normalize_dynamic(sequence)
        # _normalize_dynamic returns (1, frames, features), we want (frames, features)
        normalized_seq = normalized_seq[0]

        # 6. Pad to target length
        if normalized_seq.shape[0] < target_len:
            pad = np.zeros((target_len - normalized_seq.shape[0], normalized_seq.shape[1]), dtype=np.float32)
            normalized_seq = np.vstack([normalized_seq, pad])

        # 7. Inference
        probabilities = self._predict_pytorch(normalized_seq[np.newaxis, ...], actual_len=actual_len)

        top_k = min(5, int(probabilities.shape[0]))
        top_indices = probabilities.argsort()[-top_k:][::-1]

        predicted_idx = int(top_indices[0])
        predicted_word = str(self.registry.dynamic.label_encoder.classes_[predicted_idx])
        confidence = float(probabilities[predicted_idx])

        top_predictions = [
            {
                "label": str(self.registry.dynamic.label_encoder.classes_[int(idx)]),
                "confidence": float(probabilities[int(idx)]),
            }
            for idx in top_indices
        ]

        return {
            "word": predicted_word,
            "confidence": confidence,
            "body_detected": body_detected,
            "top_predictions": top_predictions,
            "annotated_frame_b64": encode_frame(annotated_frame) if annotated_frame is not None else None,
        }

    def _normalize_static_landmarks(self, landmarks: Any) -> list[float] | None:
        wx, wy, wz = landmarks[0].x, landmarks[0].y, landmarks[0].z
        base_ids = [5, 9, 13, 17]

        distances = []
        for idx in base_ids:
            dx = landmarks[idx].x - wx
            dy = landmarks[idx].y - wy
            dz = landmarks[idx].z - wz
            distances.append(float(np.sqrt(dx * dx + dy * dy + dz * dz)))

        scale = float(np.mean(distances))
        if scale < 1e-6:
            return None

        features: list[float] = []
        for lm in landmarks:
            features.append((lm.x - wx) / scale)
            features.append((lm.y - wy) / scale)
            features.append((lm.z - wz) / scale)

        for lm in landmarks:
            dist = float(np.sqrt((lm.x - wx) ** 2 + (lm.y - wy) ** 2 + (lm.z - wz) ** 2))
            features.append(dist / scale)

        return features

    def _extract_dynamic_vector(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, bool, np.ndarray]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        with self._dynamic_pose_detector_lock:
            pose_result = self._dynamic_pose_detector.detect(mp_image)
        with self._dynamic_hand_detector_lock:
            hand_result = self._dynamic_hand_detector.detect(mp_image)

        pose_arr = self._pose_to_array(pose_result)
        left_hand, right_hand = self._split_hands_by_side(hand_result)
        lh_arr = self._hand_to_array(left_hand)
        rh_arr = self._hand_to_array(right_hand)

        if left_hand:
            self._draw_task_hand_landmarks(frame_bgr, left_hand)
        if right_hand:
            self._draw_task_hand_landmarks(frame_bgr, right_hand)

        vector = np.concatenate([lh_arr, pose_arr, rh_arr])
        body_detected = bool(pose_result.pose_landmarks or hand_result.hand_landmarks)
        return vector, body_detected, frame_bgr

    def _pose_to_array(self, pose_result: Any) -> np.ndarray:
        if not pose_result.pose_landmarks:
            return np.zeros(33 * 3, dtype=np.float32)

        pose_lms = pose_result.pose_landmarks[0]
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_lms],
            dtype=np.float32,
        ).flatten()

    def _split_hands_by_side(self, hand_result: Any) -> tuple[Any | None, Any | None]:
        left_hand: Any | None = None
        right_hand: Any | None = None

        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            handedness_list = hand_result.handedness[idx] if idx < len(hand_result.handedness) else []
            label = handedness_list[0].category_name.lower() if handedness_list else ""
            if label == "left":
                left_hand = landmarks
                continue
            if label == "right":
                right_hand = landmarks
                continue
            if left_hand is None:
                left_hand = landmarks
            elif right_hand is None:
                right_hand = landmarks

        return left_hand, right_hand

    def _hand_to_array(self, hand_landmarks: Any | None) -> np.ndarray:
        if not hand_landmarks:
            return np.zeros(21 * 3, dtype=np.float32)
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32).flatten()

    def _prepare_dynamic_sequence(self, vectors: list[np.ndarray], target_len: int | None = None) -> tuple[np.ndarray, int]:
        """Convert list of vectors to a fixed-length sequence and return actual length."""
        if not vectors:
            dim = 225
            t_len = target_len or int(self.settings.sequence_length)
            return np.zeros((t_len, dim), dtype=np.float32), 0

        sequence = np.stack(vectors, axis=0)
        actual_len = sequence.shape[0]
        seq_len = target_len or int(self.settings.sequence_length)

        if actual_len >= seq_len:
            sequence = sequence[:seq_len]
            actual_len = seq_len
        else:
            pad = np.zeros((seq_len - actual_len, sequence.shape[1]), dtype=np.float32)
            sequence = np.vstack([sequence, pad])

        return sequence, actual_len

    def _normalize_dynamic(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize dynamic sequence features.
        The 225-dim PyTorch models were trained with per-sequence X-Y centering.
        The 258-dim legacy models were trained with global Z-score normalization.
        """
        feat_dim = int(sequence.shape[1])

        # Detect if we are using the new 225-dim models
        if feat_dim == 225:
            # Implementation of the per-sequence centering from the notebooks
            # arr[:, 0::3] is X, arr[:, 1::3] is Y
            normalized = sequence.copy()
            
            # Find non-zero coordinates to calculate mean
            mask_x = normalized[:, 0::3] != 0
            mask_y = normalized[:, 1::3] != 0
            
            if mask_x.any():
                mean_x = normalized[:, 0::3][mask_x].mean()
                # Subtract from all frames (including those with zeros)
                # This matches notebook: arr[:, 0::3] -= cx
                normalized[:, 0::3] -= mean_x
                
            if mask_y.any():
                mean_y = normalized[:, 1::3][mask_y].mean()
                normalized[:, 1::3] -= mean_y
            
            return normalized[np.newaxis, ...].astype(np.float32)

        # Legacy Z-score normalization for 258-dim or other sizes
        mean = np.asarray(self.registry.dynamic.norm_mean, dtype=np.float32).reshape(-1)
        std = np.asarray(self.registry.dynamic.norm_std, dtype=np.float32).reshape(-1)

        # Handle legacy mean/std that included pose visibility (33*4 + 21*3 + 21*3 = 258)
        if mean.size != feat_dim:
            logger.warning(
                "Dynamic normalization vectors size mismatch: mean=%d expected=%d; attempting compatibility fix",
                mean.size,
                feat_dim,
            )
            if mean.size == 258 and feat_dim == 225:
                pose_mean = mean[: 33 * 4].reshape(33, 4)[:, :3].reshape(-1)
                pose_std = std[: 33 * 4].reshape(33, 4)[:, :3].reshape(-1)
                rest_mean = mean[33 * 4 :]
                rest_std = std[33 * 4 :]
                mean = np.concatenate([pose_mean, rest_mean], axis=0)
                std = np.concatenate([pose_std, rest_std], axis=0)
            else:
                if mean.size > feat_dim:
                    mean = mean[:feat_dim]
                    std = std[:feat_dim]
                else:
                    pad_mean = np.full(feat_dim - mean.size, mean[-1], dtype=mean.dtype)
                    pad_std = np.full(feat_dim - std.size, std[-1], dtype=std.dtype)
                    mean = np.concatenate([mean, pad_mean], axis=0)
                    std = np.concatenate([std, pad_std], axis=0)

        std = np.where(std == 0, 1.0, std)
        normalized = (sequence - mean) / std
        return normalized[np.newaxis, ...].astype(np.float32)

    def _predict_pytorch(self, sequence: np.ndarray, actual_len: int | None = None) -> np.ndarray:
        """Run PyTorch model inference."""
        checkpoint = self.registry.dynamic.classifier
        if not isinstance(checkpoint, dict):
            logger.error("PyTorch checkpoint not loaded correctly")
            return np.zeros(50, dtype=np.float32)

        try:
            model_config = self.registry.dynamic.pytorch_config or {}
            input_dim = checkpoint.get("input_dim", 225)
            num_classes = checkpoint.get("num_classes", 50)
            max_frames = checkpoint.get("max_frames", self.settings.sequence_length)

            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
            model_kind = self.registry.dynamic.model_kind

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
                    nhead=model_config.get("nhead", 8),
                    num_layers=model_config.get("num_layers", 4),
                    dim_feedforward=model_config.get("dim_feedforward", 512),
                    dropout=model_config.get("dropout", 0.1),
                    max_len=512,
                )

            if state_dict is not None:
                model.load_state_dict(state_dict)

            model.to(DEVICE)
            model.eval()

            # Run inference
            x = torch.tensor(sequence, dtype=torch.float32).to(DEVICE)
            
            # Create padding mask (True for padding positions)
            pad_mask = None
            if actual_len is not None:
                pad_mask = torch.zeros(1, x.size(1), dtype=torch.bool).to(DEVICE)
                if actual_len < x.size(1):
                    pad_mask[:, actual_len:] = True

            with torch.no_grad():
                logits = model(x, pad_mask=pad_mask)
                probs = torch.softmax(logits, dim=-1)

            return probs[0].cpu().numpy()
        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(50, dtype=np.float32)
