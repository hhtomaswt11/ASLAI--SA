from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image


class InvalidFrameError(ValueError):
    pass


def decode_base64_frame(frame_b64: str) -> np.ndarray:
    try:
        payload = base64.b64decode(frame_b64)
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # pragma: no cover
        raise InvalidFrameError("Invalid base64 frame payload") from exc


def encode_frame(frame_bgr: np.ndarray, quality: int = 80) -> str:
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise InvalidFrameError("Could not encode annotated frame")
    return base64.b64encode(buffer).decode("utf-8")
