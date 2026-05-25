# =============================================================
# detect.py — Face Mask Detection: Real-Time Webcam Detection
# Uses OpenCV DNN face detector + trained MobileNetV2 model
# =============================================================

import cv2
import numpy as np
import argparse
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ─── Configuration ───────────────────────────────────────────
MODEL_PATH          = "mask_detector.keras"   # ✅ FIXED HERE
FACE_PROTOTXT       = "face_detector/deploy.prototxt"
FACE_WEIGHTS        = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
MIN_CONFIDENCE      = 0.5
IMG_SIZE            = (224, 224)
LABELS              = ["Mask", "No Mask"]
COLORS              = {
    "Mask":    (0, 255, 0),
    "No Mask": (0, 0, 255),
}

# ─── Helper: detect & predict on a single frame ──────────────
def detect_and_predict(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces  = []
    locs   = []
    preds  = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < MIN_CONFIDENCE:
            continue

        box  = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY)     = (min(w - 1, endX), min(h - 1, endY))

        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, IMG_SIZE)
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((startX, startY, endX, endY))

    if faces:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# ─── Helper: annotate frame ──────────────────────────────────
def annotate_frame(frame, locs, preds):
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (maskProb, noMaskProb) = pred

        label = "Mask" if maskProb > noMaskProb else "No Mask"
        confidence = max(maskProb, noMaskProb) * 100
        color = COLORS[label]

        text = f"{label}: {confidence:.1f}%"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame,
                      (startX, startY - th - 10),
                      (startX + tw + 6, startY),
                      color, -1)

        cv2.putText(frame, text,
                    (startX + 3, startY - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    return frame

# ─── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Face Mask Detector")
    parser.add_argument("-i", "--image",
                        help="Path to input image (omit for webcam)")
    args = parser.parse_args()

    # Load face detector
    if not os.path.exists(FACE_PROTOTXT) or not os.path.exists(FACE_WEIGHTS):
        print("[ERROR] Face detector files not found.")
        return

    print("[INFO] Loading face detector...")
    faceNet = cv2.dnn.readNet(FACE_PROTOTXT, FACE_WEIGHTS)

    # Load mask model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Mask model not found at: {MODEL_PATH}")
        print("Run train.py first.")
        return

    print("[INFO] Loading mask detector model...")
    maskNet = load_model(MODEL_PATH)

    # Image mode
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[ERROR] Could not read image: {args.image}")
            return
        (locs, preds) = detect_and_predict(frame, faceNet, maskNet)
        frame = annotate_frame(frame, locs, preds)

        cv2.imshow("Face Mask Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Webcam mode
    else:
        print("[INFO] Starting webcam stream... Press 'q' to quit.")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            (locs, preds) = detect_and_predict(frame, faceNet, maskNet)
            frame = annotate_frame(frame, locs, preds)

            cv2.imshow("Face Mask Detection — Real Time", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()