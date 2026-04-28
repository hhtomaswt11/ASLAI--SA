from __future__ import annotations

import logging
import threading
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import requests

from app.core.config import get_settings
from app.core.model_registry import ModelRegistry
from app.utils.image import decode_base64_frame, encode_frame

logger = logging.getLogger(__name__)

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

        sequence = self._prepare_dynamic_sequence(vectors)
        probabilities = self.registry.dynamic.classifier.predict(sequence, verbose=0)[0]

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

        pose = self._pose_to_array(pose_result)
        left_hand, right_hand = self._split_hands_by_side(hand_result)
        lh = self._hand_to_array(left_hand)
        rh = self._hand_to_array(right_hand)

        if left_hand:
            self._draw_task_hand_landmarks(frame_bgr, left_hand)
        if right_hand:
            self._draw_task_hand_landmarks(frame_bgr, right_hand)

        vector = np.concatenate([pose, lh, rh], axis=0)
        body_detected = bool(pose_result.pose_landmarks or hand_result.hand_landmarks)
        return vector, body_detected, frame_bgr

    def _pose_to_array(self, pose_result: Any) -> np.ndarray:
        if not pose_result.pose_landmarks:
            return np.zeros(33 * 4, dtype=np.float32)

        pose_lms = pose_result.pose_landmarks[0]
        return np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_lms],
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

    def _prepare_dynamic_sequence(self, vectors: list[np.ndarray]) -> np.ndarray:
        sequence = np.stack(vectors, axis=0)
        seq_len = int(self.settings.sequence_length)

        if sequence.shape[0] >= seq_len:
            start = (sequence.shape[0] - seq_len) // 2
            sequence = sequence[start : start + seq_len]
        else:
            last = sequence[-1:]
            pad = np.tile(last, (seq_len - sequence.shape[0], 1))
            sequence = np.vstack([sequence, pad])

        mean = np.asarray(self.registry.dynamic.norm_mean, dtype=np.float32).reshape(-1)
        std = np.asarray(self.registry.dynamic.norm_std, dtype=np.float32).reshape(-1)
        std = np.where(std == 0, 1.0, std)

        normalized = (sequence - mean) / std
        return normalized[np.newaxis, ...].astype(np.float32)
