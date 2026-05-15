"""Real-time waste classification using the trained .pth checkpoint.

This script mirrors the ONNX inference script but loads the PyTorch checkpoint
directly. The model definition (CBAM-MobileNetV3-Small) is replicated here so
the script is self-contained — you don't need the original training notebook.

Preprocessing matches the training pipeline's eval transform:
    Resize(shorter side = round(224 * 256/224) = 256) -> CenterCrop(224)
    -> ToTensor -> Normalize(ImageNet mean/std).

For a live webcam feed we approximate this with cv2.resize to 256 on the
shorter side and a center crop to 224 — visually identical, much faster.
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

# --- CONFIGURATION ---
MODEL_PATH = "best_model.pth"
# Class order MUST match training (alphabetical, as set by EXPECTED_CLASSES).
LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224
RESIZE_SHORTER = 256  # round(IMG_SIZE * 256 / 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- MODEL DEFINITION (must match training architecture) ---
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(channels // reduction_ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sigmoid()
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        attention = self.gate(pooled)
        self.last_attention = attention.detach()
        return attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.gate(self.conv(torch.cat([avg_map, max_map], dim=1)))
        self.last_attention = attention.detach()
        return attention


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class CBAMMobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.25):
        super().__init__()
        # weights=None: we load fine-tuned weights from the checkpoint anyway,
        # so we skip the ImageNet download.
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        in_features = backbone.classifier[0].in_features  # 576
        self.attention = CBAM(in_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        return self.classifier(x)


def load_model(path: str, num_classes: int, device: torch.device) -> nn.Module:
    """Load the .pth checkpoint into a fresh CBAMMobileNetV3Small."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Pull dropout from the saved config if present, otherwise use the default.
    dropout = 0.25
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        dropout = float(checkpoint["config"].get("dropout", dropout))

    model = CBAMMobileNetV3Small(num_classes=num_classes, dropout=dropout)

    # The training script saves under "model_state_dict". Fall back to a raw
    # state_dict for robustness.
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    # Strip the "module." prefix DataParallel adds when training on >=2 GPUs.
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    if isinstance(checkpoint, dict):
        epoch = checkpoint.get("epoch", "?")
        val_f1 = checkpoint.get("val_macro_f1", None)
        val_acc = checkpoint.get("val_accuracy", None)
        info = f"Loaded checkpoint (epoch {epoch}"
        if val_f1 is not None:
            info += f", val_macro_f1={val_f1:.4f}"
        if val_acc is not None:
            info += f", val_accuracy={val_acc:.4f}"
        info += ")"
        print(info)
    return model


def preprocess(frame: np.ndarray, size: int) -> torch.Tensor:
    """BGR HxWx3 uint8 frame -> normalized 1x3xHxW float32 tensor.

    Mirrors the training-time eval transform: resize shorter side to 256,
    center-crop to ``size``, BGR->RGB, /255, ImageNet normalize.
    """
    h, w = frame.shape[:2]
    # Resize so the shorter side is RESIZE_SHORTER.
    scale = RESIZE_SHORTER / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop to size x size.
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    img = img[y0:y0 + size, x0:x0 + size]

    # BGR -> RGB, uint8 -> float32 in [0, 1], ImageNet normalize.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW, add batch dim.
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def main() -> None:
    print(f"Device: {DEVICE}")
    model = load_model(MODEL_PATH, num_classes=len(LABELS), device=DEVICE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("Starting Waste Classifier... Press 'q' to exit.")

    import time
    softmax = nn.Softmax(dim=1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = preprocess(frame, IMG_SIZE).to(DEVICE)

            start = time.time()
            with torch.inference_mode():
                logits = model(input_tensor)
                probs = softmax(logits)
            # Sync on GPU so the timing is accurate; cheap on CPU.
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            inference_ms = (time.time() - start) * 1000.0

            probs_np = probs.squeeze(0).cpu().numpy()
            idx = int(np.argmax(probs_np))
            confidence = float(probs_np[idx])
            label = LABELS[idx]

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