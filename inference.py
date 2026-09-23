import cv2
import time
from ultralytics import YOLO


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_PATH = "best.pt"

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully")


# ============================================================
# CAMERA
# ============================================================

camera_url = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"


# ============================================================
# DISPLAY SETTINGS
# ============================================================

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080


# ============================================================
# RECONNECTION SETTINGS
# ============================================================

RECONNECT_DELAY = 2  # seconds


# ============================================================
# YOLO SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.45
IMAGE_SIZE = 640


# ============================================================
# CAMERA CONNECTION
# ============================================================

def connect_camera():
    """Connect to the RTSP camera and return the VideoCapture object."""

    print("Connecting to camera...")

    cap = cv2.VideoCapture(camera_url)

    # Reduce buffering
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if cap.isOpened():
        print("Camera connected successfully")
        return cap

    cap.release()

    print("Could not connect to camera")
    return None


# ============================================================
# INITIAL CONNECTION
# ============================================================

cap = connect_camera()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # If camera is not connected, keep trying
    # --------------------------------------------------------

    if cap is None or not cap.isOpened():

        print(f"Retrying connection in {RECONNECT_DELAY} seconds...")
        time.sleep(RECONNECT_DELAY)

        cap = connect_camera()

        continue


    # --------------------------------------------------------
    # Read frame
    # --------------------------------------------------------

    ret, frame = cap.read()


    # --------------------------------------------------------
    # If frame was not received
    # --------------------------------------------------------

    if not ret:

        print("Failed to receive frame")
        print("Reconnecting to camera...")

        # Close current connection
        cap.release()
        cap = None

        # Wait before reconnecting
        time.sleep(RECONNECT_DELAY)

        continue


    # --------------------------------------------------------
    # Resize frame to 1920 x 1080
    # --------------------------------------------------------

    frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )


    # ========================================================
    # YOLO INFERENCE
    # ========================================================

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    # --------------------------------------------------------
    # Draw YOLO detections on frame
    # --------------------------------------------------------

    annotated_frame = results[0].plot()


    # --------------------------------------------------------
    # Display stream
    # --------------------------------------------------------

    cv2.imshow("YOLO Live Detection", annotated_frame)


    # --------------------------------------------------------
    # Press Q to manually stop
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print("Stream stopped manually")
        break


# ============================================================
# CLEANUP
# ============================================================

if cap is not None:
    cap.release()

cv2.destroyAllWindows()