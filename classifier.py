"""Real-time waste classification using the INT8 ONNX model + config.yaml.

Unlike a hard-coded inference script, this one reads every preprocessing and
I/O parameter from ``config.yaml`` so it stays correct if the model is
retrained at a different resolution, with a different normalization, or with a
different class set. The only things hard-coded here are the camera index and
the OpenCV display loop.

Preprocessing mirrors the training-time eval transform exactly:
    Resize(shorter side = config.input.resize.shorter_side)
    -> CenterCrop(config.input.resize.center_crop)
    -> /255 -> Normalize(mean, std)
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import time
import yaml

# --- CONFIGURATION ---
MODEL_PATH = "best_model_int8.onnx"
CONFIG_PATH = "config.yaml"
CAMERA_INDEX = 0


def load_config(path: str) -> Dict[str, Any]:
    """Load and minimally validate the export config.yaml."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    required = ("input", "output", "classes")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config.yaml is missing required sections: {missing}")
    return cfg


def class_names_from_config(cfg: Dict[str, Any]) -> List[str]:
    """Return class names in index order. Prefer the explicit names list."""
    classes = cfg["classes"]
    if "names" in classes:
        return list(classes["names"])
    # Fallback: build from index_to_name (keys may be ints or strings).
    idx_map = classes["index_to_name"]
    items = sorted(idx_map.items(), key=lambda kv: int(kv[0]))
    return [name for _, name in items]


def build_preprocessor(cfg: Dict[str, Any]):
    """Construct a preprocess(frame_bgr) -> np.ndarray closure from the config.

    The closure captures all parameters once so the per-frame call is just
    array ops — no dict lookups in the hot path.
    """
    in_cfg = cfg["input"]
    image_size = int(in_cfg["image_size"])
    resize_cfg = in_cfg.get("resize", {})
    shorter_side = int(resize_cfg.get("shorter_side", image_size))
    crop = int(resize_cfg.get("center_crop", image_size))

    mean = np.array(in_cfg["normalize"]["mean"], dtype=np.float32)
    std = np.array(in_cfg["normalize"]["std"], dtype=np.float32)

    channel_order = in_cfg.get("channel_order", "RGB").upper()
    bgr_to_rgb = channel_order == "RGB"  # OpenCV gives us BGR; flip if model wants RGB.

    # /255 scaling iff the config says the model wants [0, 1] before normalize.
    value_range = in_cfg.get("value_range", [0.0, 1.0])
    scale = 1.0 / 255.0 if float(value_range[1]) <= 1.5 else 1.0

    def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        # Resize keeping aspect ratio so the shorter side equals shorter_side.
        ratio = shorter_side / min(h, w)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        img = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop to (crop, crop).
        y0 = (new_h - crop) // 2
        x0 = (new_w - crop) // 2
        img = img[y0:y0 + crop, x0:x0 + crop]

        if bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) * scale
        img = (img - mean) / std

        # HWC -> CHW, add batch dim, ensure contiguous for ORT.
        img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(img[None, ...])

    return preprocess, image_size


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main() -> None:
    cfg_path = Path(CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path.resolve()}")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {Path(MODEL_PATH).resolve()}")

    cfg = load_config(str(cfg_path))
    labels = class_names_from_config(cfg)
    preprocess, image_size = build_preprocessor(cfg)

    print(f"Model:        {MODEL_PATH}")
    print(f"Config:       {cfg_path}")
    print(f"Input size:   {image_size}x{image_size}")
    print(f"Classes ({len(labels)}): {labels}")
    if "metrics" in cfg:
        m = cfg["metrics"]
        print(
            "Reported test accuracy: "
            f"{m.get('test_accuracy', float('nan')):.4f} | "
            f"macro-F1: {m.get('test_macro_f1', float('nan')):.4f}"
        )

    # CPUExecutionProvider works everywhere; ORT will pick XNNPACK kernels on
    # ARM where available. Add 'CUDAExecutionProvider' first if you have a GPU
    # build of onnxruntime-gpu installed.
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    # Sanity check: the config's declared I/O names should match the model's.
    expected_input = cfg.get("onnx", {}).get("input_names", [input_name])[0]
    if expected_input != input_name:
        print(f"WARNING: config says input='{expected_input}' but model has '{input_name}'.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}.")

    print("Starting Waste Classifier... Press 'q' to exit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_data = preprocess(frame)

            start = time.time()
            outputs = session.run(output_names, {input_name: input_data})
            inference_ms = (time.time() - start) * 1000.0

            logits = outputs[0]
            probs = softmax(logits, axis=1)[0]
            idx = int(np.argmax(probs))
            label = labels[idx]
            confidence = float(probs[idx])

            text = f"{label} ({confidence:.2f}) | {inference_ms:.1f}ms"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Waste Classification", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()