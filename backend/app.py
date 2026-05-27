from __future__ import annotations

import os
import time
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from model_service import ModelService, decode_base64_image


BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = (BASE_DIR / ".." / "Code").resolve()

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

service = ModelService(CODE_DIR)


@app.get("/health")
def health() -> tuple[dict, int]:
    return {"status": "ok"}, 200


@app.post("/predict")
def predict() -> tuple[dict, int]:
    payload = request.get_json(silent=True) or {}
    mode = (payload.get("mode") or request.args.get("mode") or "letters").lower()
    image_data = payload.get("image") or payload.get("image_base64")

    if not image_data:
        return {"error": "Missing image data"}, 400

    try:
        image = decode_base64_image(image_data)
    except ValueError as exc:
        return {"error": str(exc)}, 400

    start = time.perf_counter()
    try:
        result = service.predict(mode, image)
    except ValueError as exc:
        return {"error": str(exc)}, 400

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "mode": mode,
        "label": result["label"],
        "score": result["score"],
        "top3": result["top3"],
        "latency_ms": latency_ms,
    }, 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
