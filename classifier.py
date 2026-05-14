import cv2
import numpy as np
import onnxruntime as ort
import time

# --- CONFIGURATION ---
MODEL_PATH = "best_model_int8.onnx"
# Update these labels based on your specific training classes
LABELS = ["cardboard", "metal", "glass", "paper", "palstic", "trash"]
IMG_SIZE = 224  # Standard size for most classification models

# 1. Load the quantized ONNX model
# The 'CPUExecutionProvider' is optimized for ARM processors via XNNPACK
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def preprocess(frame, size):
    """Convert OpenCV BGR frame to Model-ready NCHW format."""
    # Resize and convert to RGB
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] - adjust if your model expects [-1, 1]
    img = img.astype(np.float32) / 255.0
    
    # HWC to CHW format (Height, Width, Channels -> Channels, Height, Width)
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension (1, C, H, W)
    img = np.expand_dims(img, axis=0)
    return img

# 2. Initialize USB Camera
cap = cv2.VideoCapture(0)

print("Starting Waste Classifier... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-process the frame
    input_data = preprocess(frame, IMG_SIZE)

    # 3. Run Inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})
    inference_time = (time.time() - start_time) * 1000

    # 4. Post-process results
    logits = outputs[0]
    result_index = np.argmax(logits)
    confidence = np.max(np.exp(logits) / np.sum(np.exp(logits))) # Softmax
    
    label = LABELS[result_index]

    # 5. Display output on frame
    text = f"{label} ({confidence:.2f}) | {inference_time:.1f}ms"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Waste Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()