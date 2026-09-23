
import cv2
import time
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

# Change this one line to test different pretrained models
# Examples:
# yolo11n.pt
# yolo11s.pt
# yolo11m.pt
# yolo26n.pt
# yolo26s.pt

MODEL_PATH = "yolo11n.pt"

# Your RTSP URL
RTSP_URL = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"

# YOLO confidence threshold
CONFIDENCE = 0.25

# Input image size for YOLO
IMAGE_SIZE = 640


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading model: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print("Classes:")
print(model.names)


# ============================================================
# OPEN RTSP STREAM
# ============================================================

print("Connecting to RTSP stream...")

cap = cv2.VideoCapture(RTSP_URL)

# Reduce OpenCV buffering if supported
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Could not open RTSP stream.")
    exit()

print("RTSP stream connected successfully.")


# ============================================================
# FPS VARIABLES
# ============================================================

previous_time = time.time()

fps = 0

frame_count = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from RTSP stream.")
        time.sleep(0.1)
        continue

    frame_count += 1

    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    inference_start = time.time()

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    inference_time = (time.time() - inference_start) * 1000

    # --------------------------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------------------------

    annotated_frame = results[0].plot()

    # Number of detected objects
    boxes = results[0].boxes

    if boxes is not None:
        detection_count = len(boxes)
    else:
        detection_count = 0

    # --------------------------------------------------------
    # CALCULATE FPS
    # --------------------------------------------------------

    current_time = time.time()

    elapsed_time = current_time - previous_time

    if elapsed_time > 0:
        fps = 1 / elapsed_time

    previous_time = current_time

    # --------------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------------

    cv2.putText(
        annotated_frame,
        f"Model: {MODEL_PATH}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Inference: {inference_time:.1f} ms",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Objects: {detection_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # DISPLAY FRAME
    # --------------------------------------------------------

    cv2.imshow("Pretrained YOLO - RTSP Test", annotated_frame)

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("RTSP stream closed.")
print("Program finished.")

