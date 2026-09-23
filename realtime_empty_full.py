import cv2
import torch
import threading
import time
import os
from PIL import Image
from torchvision import models, transforms


# =========================================================
# CONFIGURATION
# =========================================================

RTSP_URL = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"

MODEL_PATH = "efficientnet_b0_strongempty.pth"

EXPECTED_WIDTH = 3840
EXPECTED_HEIGHT = 2160

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# ---------------------------------------------------------
# FIXED ROIs - 3840 x 2160
# Format: x, y, width, height
# ---------------------------------------------------------

LEFT_ROI = (969, 456, 711, 837)
RIGHT_ROI = (2250, 510, 651, 891)

# ---------------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------------

CLASS_NAMES = {
    0: "EMPTY",
    1: "FULL"
}

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==========================================")
print("   EMPTY / FULL BOX CLASSIFIER")
print("==========================================")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================================================
# CHECK MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found!")
    print(f"Expected: {os.path.abspath(MODEL_PATH)}")
    exit()


# =========================================================
# LOAD EFFICIENTNET-B0
# =========================================================

print("\nLoading model...")

model = models.efficientnet_b0(weights=None)

# Replace classifier for 2 classes
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    2
)


# =========================================================
# LOAD CHECKPOINT
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["state_dict"]
    )

else:

    model.load_state_dict(checkpoint)


model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# =========================================================
# INFERENCE TRANSFORM
# =========================================================

# IMPORTANT:
# Training augmentation is NOT used during inference.

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# RTSP READER
# =========================================================

class RTSPReader:

    def __init__(self, url):

        self.url = url

        self.cap = None

        self.latest_frame = None

        self.lock = threading.Lock()

        self.running = False

        self.thread = None

        self.failed_reads = 0

        self.MAX_FAILED_READS = 10


    def connect(self):

        print("\nConnecting to RTSP...")

        if self.cap is not None:

            self.cap.release()

        self.cap = cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG
        )

        # Try to reduce buffering
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        if self.cap.isOpened():

            width = int(
                self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            print(
                f"RTSP connected: {width}x{height}"
            )

            return True

        print("RTSP connection failed.")

        return False


    def update(self):

        while self.running:

            if self.cap is None or not self.cap.isOpened():

                if not self.connect():

                    time.sleep(2)

                    continue


            ret, frame = self.cap.read()

            if ret:

                self.failed_reads = 0

                with self.lock:

                    # Keep ONLY the latest frame
                    self.latest_frame = frame

            else:

                self.failed_reads += 1

                print(
                    f"Frame read failed "
                    f"({self.failed_reads}/{self.MAX_FAILED_READS})"
                )

                if self.failed_reads >= self.MAX_FAILED_READS:

                    print("Reconnecting RTSP...")

                    self.cap.release()

                    self.cap = None

                    self.failed_reads = 0

                    time.sleep(1)


    def start(self):

        self.running = True

        self.connect()

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()


    def get_frame(self):

        with self.lock:

            if self.latest_frame is None:

                return None

            return self.latest_frame.copy()


    def stop(self):

        self.running = False

        if self.cap is not None:

            self.cap.release()


# =========================================================
# ROI CROP
# =========================================================

def crop_roi(frame, roi):

    x, y, w, h = roi

    return frame[
        y:y + h,
        x:x + w
    ]


# =========================================================
# CLASSIFY ROI
# =========================================================

@torch.no_grad()
def classify_roi(crop):

    # OpenCV BGR → RGB
    rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL
    image = Image.fromarray(rgb)

    # Preprocess
    tensor = inference_transform(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Move to GPU/CPU
    tensor = tensor.to(DEVICE)

    # Model inference
    output = model(tensor)

    # Probabilities
    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )

    class_id = predicted.item()

    confidence = confidence.item()

    return (
        CLASS_NAMES[class_id],
        confidence
    )


# =========================================================
# START RTSP
# =========================================================

reader = RTSPReader(RTSP_URL)

reader.start()

print("\nStarting live inference...")
print("Press Q to quit.")


# =========================================================
# FPS VARIABLES
# =========================================================

fps_counter = 0

fps_start_time = time.time()

display_fps = 0


# =========================================================
# MAIN LOOP
# =========================================================

try:

    while True:

        frame = reader.get_frame()

        if frame is None:

            time.sleep(0.01)

            continue


        # -------------------------------------------------
        # Check resolution
        # -------------------------------------------------

        height, width = frame.shape[:2]

        if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:

            print(
                f"WARNING: Expected "
                f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, "
                f"got {width}x{height}"
            )


        # -------------------------------------------------
        # Crop LEFT and RIGHT
        # -------------------------------------------------

        left_crop = crop_roi(
            frame,
            LEFT_ROI
        )

        right_crop = crop_roi(
            frame,
            RIGHT_ROI
        )


        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        left_status, left_conf = classify_roi(
            left_crop
        )

        right_status, right_conf = classify_roi(
            right_crop
        )


        # -------------------------------------------------
        # Draw ROI rectangles
        # -------------------------------------------------

        x, y, w, h = LEFT_ROI

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            4
        )


        x, y, w, h = RIGHT_ROI

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            4
        )


        # -------------------------------------------------
        # Resize for display
        # -------------------------------------------------

        display = cv2.resize(
            frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )


        # Scale ROI coordinates to display
        scale_x = DISPLAY_WIDTH / width
        scale_y = DISPLAY_HEIGHT / height


        # -------------------------------------------------
        # LEFT label
        # -------------------------------------------------

        left_text = (
            f"LEFT: {left_status} "
            f"{left_conf * 100:.1f}%"
        )

        cv2.putText(
            display,
            left_text,
            (
                int(LEFT_ROI[0] * scale_x),
                int(LEFT_ROI[1] * scale_y) - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        # -------------------------------------------------
        # RIGHT label
        # -------------------------------------------------

        right_text = (
            f"RIGHT: {right_status} "
            f"{right_conf * 100:.1f}%"
        )

        cv2.putText(
            display,
            right_text,
            (
                int(RIGHT_ROI[0] * scale_x),
                int(RIGHT_ROI[1] * scale_y) - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        # -------------------------------------------------
        # FPS
        # -------------------------------------------------

        fps_counter += 1

        elapsed = time.time() - fps_start_time

        if elapsed >= 1.0:

            display_fps = fps_counter / elapsed

            fps_counter = 0

            fps_start_time = time.time()


        cv2.putText(
            display,
            f"FPS: {display_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # -------------------------------------------------
        # SHOW
        # -------------------------------------------------

        cv2.imshow(
            "4th Press - Empty Full Detection",
            display
        )


        # -------------------------------------------------
        # QUIT
        # -------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


finally:

    print("\nStopping...")

    reader.stop()

    cv2.destroyAllWindows()

    print("Program stopped.")