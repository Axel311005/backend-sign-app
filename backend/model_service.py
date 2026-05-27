from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf


LETTERS_ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXY"
NUMBERS_ALPHABET = "0123456789"


@dataclass
class Prediction:
    label: str
    score: float


class ModelService:
    def __init__(self, code_dir: Path) -> None:
        self.code_dir = code_dir
        self.letters_model = self._load_model("modelo_2209_miguel.h5")
        self.numbers_model = self._load_model("modelmiguelnumber.h5")

    def _load_model(self, filename: str) -> tf.keras.Model:
        model_path = self.code_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return tf.keras.models.load_model(model_path)

    def predict(self, mode: str, image_bgr: np.ndarray) -> Dict[str, Any]:
        if mode == "letters":
            model = self.letters_model
            alphabet = LETTERS_ALPHABET
        elif mode == "numbers":
            model = self.numbers_model
            alphabet = NUMBERS_ALPHABET
        else:
            raise ValueError("mode must be 'letters' or 'numbers'")

        processed = preprocess_image(image_bgr)
        raw = model.predict(processed, verbose=0)[0]
        top_indices = np.argsort(raw)[-3:][::-1]
        top3 = [
            Prediction(label=alphabet[i], score=float(raw[i]))
            for i in top_indices
        ]
        best = top3[0]
        return {
            "label": best.label,
            "score": best.score,
            "top3": [prediction.__dict__ for prediction in top3],
        }


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Invalid image data")

    if len(image_bgr.shape) == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    resized = cv2.resize(image_bgr, (56, 56), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 56, 56, 3)


def decode_base64_image(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        decoded = base64.b64decode(data)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid base64 image data") from exc

    array = np.frombuffer(decoded, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")
    return image
