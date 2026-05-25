import logging
import threading
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import requests
import torch

from app.core.config import get_settings
from app.core.model_registry import ModelRegistry
from app.utils.image import decode_base64_frame, encode_frame

logger = logging.getLogger(__name__)


def _get_safe_device() -> torch.device:
    """Return CUDA if it is available AND works on this GPU, otherwise CPU.

    torch.cuda.is_available() can return True even when the installed PyTorch
    was not compiled for the current GPU's compute capability (e.g. GTX 1050
    with sm_61 vs. a build that targets sm_75+). A quick probe catches this.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _probe = torch.zeros(1, device="cuda") + 1  # forces an actual CUDA kernel
        del _probe
        return torch.device("cuda")
    except Exception as exc:
        logger.warning(
            "CUDA detected but not compatible with this GPU (%s); "
            "falling back to CPU for inference.",
            exc,
        )
        return torch.device("cpu")


DEVICE = _get_safe_device()



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
        self._dynamic_hand_detector = self._build_hand_detector(num_hands=2, video_mode=True)
        self._static_detector_lock = threading.Lock()
        self._dynamic_pose_detector_lock = threading.Lock()
        self._dynamic_hand_detector_lock = threading.Lock()
        # Monotonically-increasing timestamp for MediaPipe VIDEO-mode detectors.
        # VIDEO mode requires strictly increasing timestamps across ALL calls to
        # detect_for_video, even between separate HTTP requests.
        self._ts_lock = threading.Lock()
        self._last_ts_ms: int = 0

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

            should_mirror = False
            if result.handedness:
                handedness_list = result.handedness[0]
                # Como o modelo foi treinado com a mão ESQUERDA , 
                # precisamos aplicar o flip (espelhar) quando usamos a mão direita.
                if handedness_list and handedness_list[0].category_name.lower() == "left":
                    should_mirror = True

            features = self._normalize_static_landmarks(first_hand, mirror_x=should_mirror)
            if features is not None:
                arr = np.array(features, dtype=np.float32).reshape(1, -1)
                arr_scaled = self.registry.static.scaler.transform(arr)

                probas = self.registry.static.classifier.predict_proba(arr_scaled)[0]
                order = np.argsort(probas)[::-1]

                best_idx = int(order[0])
                second_idx = int(order[1]) if len(order) > 1 else best_idx

                best_class = self.registry.static.classifier.classes_[best_idx]
                letter = self.registry.static.label_encoder.inverse_transform([best_class])[0]

                confidence = float(probas[best_idx])
                second_confidence = float(probas[second_idx])
                margin = confidence - second_confidence

                top_debug = []
                for idx in order[:5]:
                    class_id = self.registry.static.classifier.classes_[idx]
                    label = self.registry.static.label_encoder.inverse_transform([class_id])[0]
                    top_debug.append((str(label), float(probas[idx])))

                logger.info("Static top predictions: %s", top_debug)

                if confidence < self.settings.static_confidence_threshold or margin < 0.10:
                    letter = None

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

    def _build_hand_detector(self, num_hands: int, video_mode: bool = False) -> Any | None:
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
                running_mode=mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE,
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
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
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

    def _next_ts(self) -> int:
        """Return a timestamp (ms) that is strictly greater than the last one returned.

        MediaPipe VIDEO-mode detectors maintain state across calls and require
        monotonically-increasing timestamps. Using wall-clock time per request
        fails when requests arrive within the same millisecond or when the
        detector's internal clock is ahead of the new request's wall time.
        """
        with self._ts_lock:
            now_ms = int(time.time() * 1000)
            # Advance by at least 1 ms beyond the last timestamp we issued
            self._last_ts_ms = max(now_ms, self._last_ts_ms + 1)
            return self._last_ts_ms

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
        hand_mask: list[bool] = []   # True where hand landmarks were detected
        body_detected = False
        annotated_frame: np.ndarray | None = None

        for i, frame_b64 in enumerate(frame_payloads):
            try:
                frame_bgr = decode_base64_frame(frame_b64)
                if frame_bgr is None:
                    # Keep timing consistent even for failed/missing frames
                    vectors.append(np.zeros(225, dtype=np.float32))
                    hand_mask.append(False)
                    continue
                ts_ms = self._next_ts()  # guaranteed monotonically increasing
                vector, frame_has_body, frame_has_hand, annotated = self._extract_dynamic_vector(
                    frame_bgr, timestamp_ms=ts_ms
                )
                vectors.append(vector)
                hand_mask.append(frame_has_hand)
                body_detected = body_detected or frame_has_body
                annotated_frame = annotated
            except Exception as e:
                logger.warning(f"Error processing frame {i} in dynamic sequence: {e}")
                vectors.append(np.zeros(225, dtype=np.float32))
                hand_mask.append(False)

        # --- Reject sequences with too few hand frames ---
        hand_frame_count = sum(hand_mask)
        min_required = max(1, int(len(vectors) * self.settings.min_hand_frames_ratio))
        if hand_frame_count < min_required:
            logger.debug(
                f"Rejecting: {hand_frame_count}/{len(vectors)} hand frames "
                f"(need {min_required})"
            )
            return {
                "word": None,
                "confidence": 0.0,
                "body_detected": body_detected,
                "top_predictions": [],
                "annotated_frame_b64": encode_frame(annotated_frame) if annotated_frame is not None else None,
            }

        # --- Determine target sequence length from checkpoint ---
        checkpoint = self.registry.dynamic.classifier
        target_len = (
            checkpoint.get("max_frames", self.settings.sequence_length)
            if isinstance(checkpoint, dict)
            else self.settings.sequence_length
        )

        # --- Trim to active gesture region, then pad ---
        sequence = np.stack(vectors, axis=0)
        sequence, actual_len = self._trim_to_active_frames(sequence, hand_mask, target_len)

        # --- Normalize BEFORE padding (matches notebook pipeline) ---
        normalized_seq = self._normalize_dynamic(sequence)[0]  # strip batch dim

        if normalized_seq.shape[0] < target_len:
            pad = np.zeros(
                (target_len - normalized_seq.shape[0], normalized_seq.shape[1]),
                dtype=np.float32,
            )
            normalized_seq = np.vstack([normalized_seq, pad])

        # --- Inference with Test-Time Augmentation ---
        probabilities = self._predict_with_tta(normalized_seq[np.newaxis, ...], actual_len=actual_len)

        top_k = min(5, int(probabilities.shape[0]))
        top_indices = probabilities.argsort()[-top_k:][::-1]
        predicted_idx = int(top_indices[0])
        confidence = float(probabilities[predicted_idx])

        classes = self.registry.dynamic.label_encoder.classes_
        if predicted_idx < len(classes):
            predicted_word: str | None = str(classes[predicted_idx])
        else:
            predicted_word = None
            logger.error(f"Predicted index {predicted_idx} out of bounds (len={len(classes)})")

        # --- Apply confidence threshold ---
        if confidence < self.settings.dynamic_confidence_threshold:
            logger.debug(
                f"Low-confidence prediction suppressed: {predicted_word} "
                f"({confidence:.3f} < {self.settings.dynamic_confidence_threshold})"
            )
            predicted_word = None

        top_predictions = []
        for idx in top_indices:
            idx_int = int(idx)
            label = str(classes[idx_int]) if idx_int < len(classes) else "Unknown"
            top_predictions.append({"label": label, "confidence": float(probabilities[idx_int])})

        return {
            "word": predicted_word,
            "confidence": confidence,
            "body_detected": body_detected,
            "top_predictions": top_predictions,
            "annotated_frame_b64": encode_frame(annotated_frame) if annotated_frame is not None else None,
        }


    def _normalize_static_landmarks(self, landmarks: Any, mirror_x: bool = False) -> list[float] | None:
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
            dx = lm.x - wx
            if mirror_x:
                dx = -dx
            features.append(dx / scale)
            features.append((lm.y - wy) / scale)
            features.append((lm.z - wz) / scale)

        for lm in landmarks:
            dist = float(np.sqrt((lm.x - wx) ** 2 + (lm.y - wy) ** 2 + (lm.z - wz) ** 2))
            features.append(dist / scale)

        return features

    def _extract_dynamic_vector(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> tuple[np.ndarray, bool, bool, np.ndarray]:
        """Return (feature_vector, body_detected, hand_detected, annotated_frame)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        with self._dynamic_pose_detector_lock:
            if timestamp_ms is not None:
                pose_result = self._dynamic_pose_detector.detect_for_video(mp_image, timestamp_ms)
            else:
                pose_result = self._dynamic_pose_detector.detect(mp_image)

        with self._dynamic_hand_detector_lock:
            if timestamp_ms is not None:
                hand_result = self._dynamic_hand_detector.detect_for_video(mp_image, timestamp_ms)
            else:
                hand_result = self._dynamic_hand_detector.detect(mp_image)

        pose_arr = self._pose_to_array(pose_result)
        left_hand, right_hand = self._split_hands_by_side(hand_result)
        lh_arr = self._hand_to_array(left_hand)
        rh_arr = self._hand_to_array(right_hand)

        if left_hand:
            self._draw_task_hand_landmarks(frame_bgr, left_hand)
        if right_hand:
            self._draw_task_hand_landmarks(frame_bgr, right_hand)

        # Order matches notebook pivot sort: left_hand → pose → right_hand
        vector = np.concatenate([lh_arr, pose_arr, rh_arr])
        hand_detected = bool(hand_result.hand_landmarks)
        body_detected = bool(pose_result.pose_landmarks or hand_result.hand_landmarks)
        return vector, body_detected, hand_detected, frame_bgr


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

    def _trim_to_active_frames(
        self,
        sequence: np.ndarray,
        hand_mask: list[bool],
        target_len: int,
    ) -> tuple[np.ndarray, int]:
        """
        Trim leading and trailing frames without hand detection to isolate the
        active gesture region. This aligns the inference sequence with the
        training distribution, where each sample is a complete, well-isolated gesture.

        A small context buffer is kept around the active region to preserve
        motion onset/offset information.
        """
        active = [i for i, has_hand in enumerate(hand_mask) if has_hand]
        if not active:
            # No hand detected at all — return as-is; caller already handled this
            return sequence, min(sequence.shape[0], target_len)

        CONTEXT = 2  # frames of context to keep before/after the gesture
        start = max(0, active[0] - CONTEXT)
        end = min(sequence.shape[0], active[-1] + 1 + CONTEXT)

        trimmed = sequence[start:end]
        if trimmed.shape[0] > target_len:
            trimmed = trimmed[:target_len]

        actual_len = trimmed.shape[0]
        return trimmed, actual_len


    def _normalize_dynamic(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize dynamic sequence features using per-sequence X-Y centering.
        """
        normalized = sequence.copy()
        
        # Find non-zero coordinates to calculate mean
        mask_x = normalized[:, 0::3] != 0
        mask_y = normalized[:, 1::3] != 0
        
        if mask_x.any():
            mean_x = normalized[:, 0::3][mask_x].mean()
            normalized[:, 0::3] -= mean_x
            
        if mask_y.any():
            mean_y = normalized[:, 1::3][mask_y].mean()
            normalized[:, 1::3] -= mean_y
        
        return normalized[np.newaxis, ...].astype(np.float32)

    def _predict_pytorch(self, sequence: np.ndarray, actual_len: int | None = None) -> np.ndarray:
        """Run PyTorch model inference."""
        model = self.registry.dynamic.model
        num_classes = (
            self.registry.dynamic.label_encoder.classes_.shape[0]
            if self.registry.dynamic.label_encoder is not None
            else 50
        )
        if model is None:
            logger.error("Dynamic model not loaded in registry")
            return np.zeros(num_classes, dtype=np.float32)

        try:
            model.eval()  # Ensure eval mode (disables dropout, batchnorm in training mode)
            x = torch.tensor(sequence, dtype=torch.float32).to(DEVICE)

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
            return np.zeros(num_classes, dtype=np.float32)
    def _predict_with_tta(self, sequence: np.ndarray, actual_len: int | None = None) -> np.ndarray:
        """
        Test-Time Augmentation: run inference on the original sequence and a
        horizontally-mirrored version (x-coordinates negated), then return the
        geometric mean of the two probability vectors.

        Rationale: ASL signs are often performed with the dominant hand. MediaPipe
        handedness labels can differ between the pre-recorded training data and live
        camera input (mirror effect). Averaging over both orientations makes the
        model more robust to this ambiguity at the cost of 2× inference time.
        """
        probs_orig = self._predict_pytorch(sequence, actual_len=actual_len)

        # Flip x-coordinates (every 3rd value starting at index 0)
        seq_flipped = sequence.copy()
        seq_flipped[:, :, 0::3] *= -1
        probs_flip = self._predict_pytorch(seq_flipped, actual_len=actual_len)

        # Geometric mean is more conservative than arithmetic mean for probabilities
        eps = 1e-9
        probs_geo = np.sqrt(np.maximum(probs_orig, eps) * np.maximum(probs_flip, eps))
        # Re-normalise so probabilities sum to 1
        total = probs_geo.sum()
        if total > 0:
            probs_geo /= total

        return probs_geo

