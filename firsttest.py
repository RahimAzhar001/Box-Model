import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import threading
import time
import os


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "efficientnet_b0_box_nobox_best.pth"

# ------------------------------------------------------------
# PUT YOUR ACTUAL RTSP URL HERE
# ------------------------------------------------------------

RTSP_URL = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"


# ============================================================
# ORIGINAL CAMERA RESOLUTION
# ============================================================

EXPECTED_WIDTH = 3840
EXPECTED_HEIGHT = 2160


# ============================================================
# FIXED ROIs
#
# Format:
# (x, y, width, height)
# ============================================================

LEFT_ROI = (969, 456, 711, 837)

RIGHT_ROI = (2250, 510, 651, 891)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "BOX",
    1: "NOBOX"
}


# ============================================================
# DISPLAY SETTINGS
# ============================================================

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

WINDOW_NAME = "4th Press - EfficientNet Box Detection"


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("\nCUDA available")
    print("GPU:", torch.cuda.get_device_name(0))

else:

    DEVICE = torch.device("cpu")

    print("\nCUDA not available")
    print("Using CPU")


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n"
        f"{os.path.abspath(MODEL_PATH)}\n\n"
        f"Place efficientnet_b0_box_nobox_best.pth "
        f"in the boxmodel folder."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    2
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# Support different checkpoint formats

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# Remove possible "module." prefix

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[7:]

    clean_state_dict[key] = value


model.load_state_dict(clean_state_dict)

model = model.to(DEVICE)

model.eval()


print("EfficientNet-B0 loaded successfully.")
print("Inference device:", DEVICE)


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# RTSP READER CLASS
# ============================================================

class RTSPReader:

    def __init__(self, rtsp_url):

        self.rtsp_url = rtsp_url

        self.cap = None

        self.frame = None

        self.lock = threading.Lock()

        self.running = False

        self.connected = False

        self.failed_reads = 0

        self.total_frames = 0


    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    def connect(self):

        print("\nConnecting to RTSP...")

        if self.cap is not None:

            try:
                self.cap.release()
            except:
                pass


        self.cap = cv2.VideoCapture(
            self.rtsp_url,
            cv2.CAP_FFMPEG
        )


        if not self.cap.isOpened():

            print("RTSP connection failed.")

            self.connected = False

            return False


        self.connected = True

        self.failed_reads = 0

        print("RTSP connected successfully.")

        return True


    # --------------------------------------------------------
    # READER THREAD
    # --------------------------------------------------------

    def read_loop(self):

        self.running = True

        while self.running:

            # ------------------------------------------------
            # Connect if necessary
            # ------------------------------------------------

            if self.cap is None or not self.connected:

                success = self.connect()

                if not success:

                    print(
                        "Retrying RTSP connection in 2 seconds..."
                    )

                    time.sleep(2)

                    continue


            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            ret, frame = self.cap.read()


            if not ret or frame is None:

                self.failed_reads += 1

                if self.failed_reads == 1:

                    print(
                        "\nRTSP frame read failed..."
                    )

                if self.failed_reads >= 10:

                    print(
                        "RTSP connection appears lost."
                    )

                    self.connected = False

                    try:
                        self.cap.release()
                    except:
                        pass

                    self.cap = None

                    self.failed_reads = 0

                time.sleep(0.01)

                continue


            # ------------------------------------------------
            # Successful frame
            # ------------------------------------------------

            self.failed_reads = 0

            self.total_frames += 1


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Keep ONLY the latest frame.
            #
            # We do not build a queue.
            # ------------------------------------------------

            with self.lock:

                self.frame = frame


    # --------------------------------------------------------
    # GET LATEST FRAME
    # --------------------------------------------------------

    def get_frame(self):

        with self.lock:

            if self.frame is None:

                return None

            # Copy so reader thread can continue safely

            return self.frame.copy()


    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop(self):

        self.running = False

        self.connected = False

        if self.cap is not None:

            try:
                self.cap.release()
            except:
                pass


# ============================================================
# ROI CROPPING
# ============================================================

def crop_roi(frame, roi):

    x, y, w, h = roi

    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)


    if x1 >= x2 or y1 >= y2:

        return None


    return frame[y1:y2, x1:x2]


# ============================================================
# PREPARE ROI
# ============================================================

def prepare_roi(roi_image):

    rgb = cv2.cvtColor(
        roi_image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb)

    tensor = transform(pil_image)

    return tensor


# ============================================================
# EFFICIENTNET INFERENCE
# ============================================================

@torch.inference_mode()
def classify_rois(left_image, right_image):

    left_tensor = prepare_roi(left_image)

    right_tensor = prepare_roi(right_image)


    # --------------------------------------------------------
    # BATCH BOTH ROIs
    #
    # Shape:
    # [2, 3, 224, 224]
    # --------------------------------------------------------

    batch = torch.stack([
        left_tensor,
        right_tensor
    ])


    batch = batch.to(
        DEVICE,
        non_blocking=True
    )


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    outputs = model(batch)


    probabilities = torch.softmax(
        outputs,
        dim=1
    )


    confidences, predictions = torch.max(
        probabilities,
        dim=1
    )


    left_class = predictions[0].item()

    left_confidence = confidences[0].item()


    right_class = predictions[1].item()

    right_confidence = confidences[1].item()


    return (
        left_class,
        left_confidence,
        right_class,
        right_confidence
    )


# ============================================================
# DRAW ROI
# ============================================================

def draw_roi(
    frame,
    roi,
    label,
    confidence,
    scale_x,
    scale_y
):

    x, y, w, h = roi


    # Convert original 3840x2160
    # coordinates to display coordinates

    x1 = int(x * scale_x)

    y1 = int(y * scale_y)

    x2 = int((x + w) * scale_x)

    y2 = int((y + h) * scale_y)


    # --------------------------------------------------------
    # ROI rectangle
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = (
        f"{label} "
        f"{confidence * 100:.1f}%"
    )


    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65

    thickness = 2


    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )


    text_y = max(
        text_height + 5,
        y1 - 8
    )


    # Background

    cv2.rectangle(
        frame,
        (
            x1,
            text_y - text_height - baseline
        ),
        (
            x1 + text_width + 8,
            text_y + 5
        ),
        (0, 0, 0),
        -1
    )


    # Text

    cv2.putText(
        frame,
        text,
        (x1 + 4, text_y),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")

    print(
        "   EFFICIENTNET-B0 REAL-TIME BOX TEST"
    )

    print("==========================================")

    print(
        f"Expected resolution: "
        f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
    )

    print(
        f"Left ROI:  {LEFT_ROI}"
    )

    print(
        f"Right ROI: {RIGHT_ROI}"
    )


    # ========================================================
    # START RTSP READER
    # ========================================================

    reader = RTSPReader(
        RTSP_URL
    )


    reader_thread = threading.Thread(
        target=reader.read_loop,
        daemon=True
    )


    reader_thread.start()


    print("\nWaiting for camera frame...")


    # ========================================================
    # WAIT FOR FIRST FRAME
    # ========================================================

    frame = None

    wait_start = time.time()


    while frame is None:

        frame = reader.get_frame()

        if time.time() - wait_start > 15:

            print(
                "ERROR: No frame received "
                "within 15 seconds."
            )

            reader.stop()

            return


        time.sleep(0.01)


    print("First frame received.")


    # ========================================================
    # GET RESOLUTION
    # ========================================================

    frame_height, frame_width = frame.shape[:2]


    print(
        f"Actual camera resolution: "
        f"{frame_width}x{frame_height}"
    )


    # ========================================================
    # CHECK ROI
    # ========================================================

    if frame_width < 3000 or frame_height < 2000:

        print(
            "\nWARNING:"
        )

        print(
            "Camera resolution is different "
            "from expected 3840x2160."
        )

        print(
            "The configured ROIs may be incorrect."
        )


    # ========================================================
    # FPS VARIABLES
    # ========================================================

    processed_frames = 0

    fps = 0.0

    fps_start = time.perf_counter()

    inference_time_ms = 0.0

    last_frame_time = time.perf_counter()


    # Last predictions

    left_class = 1

    left_confidence = 0.0

    right_class = 1

    right_confidence = 0.0


    # ========================================================
    # MAIN DISPLAY LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # Get latest frame
        # ----------------------------------------------------

        frame = reader.get_frame()


        if frame is None:

            time.sleep(0.005)

            continue


        last_frame_time = time.perf_counter()


        # ----------------------------------------------------
        # Actual dimensions
        # ----------------------------------------------------

        frame_height, frame_width = frame.shape[:2]


        # ----------------------------------------------------
        # Crop LEFT
        # ----------------------------------------------------

        left_crop = crop_roi(
            frame,
            LEFT_ROI
        )


        # ----------------------------------------------------
        # Crop RIGHT
        # ----------------------------------------------------

        right_crop = crop_roi(
            frame,
            RIGHT_ROI
        )


        if left_crop is None or right_crop is None:

            print(
                "ERROR: ROI outside camera frame."
            )

            time.sleep(0.01)

            continue


        # ====================================================
        # EFFICIENTNET INFERENCE
        # ====================================================

        inference_start = time.perf_counter()


        (
            left_class,
            left_confidence,
            right_class,
            right_confidence
        ) = classify_rois(
            left_crop,
            right_crop
        )


        inference_end = time.perf_counter()


        inference_time_ms = (
            inference_end - inference_start
        ) * 1000


        processed_frames += 1


        # ====================================================
        # FPS
        # ====================================================

        current_time = time.perf_counter()

        elapsed = current_time - fps_start


        if elapsed >= 1.0:

            fps = (
                processed_frames /
                elapsed
            )

            processed_frames = 0

            fps_start = current_time


        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        display_frame = cv2.resize(
            frame,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )


        scale_x = (
            DISPLAY_WIDTH /
            frame_width
        )


        scale_y = (
            DISPLAY_HEIGHT /
            frame_height
        )


        # ====================================================
        # DRAW LEFT ROI
        # ====================================================

        draw_roi(
            display_frame,
            LEFT_ROI,
            CLASS_NAMES[left_class],
            left_confidence,
            scale_x,
            scale_y
        )


        # ====================================================
        # DRAW RIGHT ROI
        # ====================================================

        draw_roi(
            display_frame,
            RIGHT_ROI,
            CLASS_NAMES[right_class],
            right_confidence,
            scale_x,
            scale_y
        )


        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        panel_height = 135


        cv2.rectangle(
            display_frame,
            (10, 10),
            (410, panel_height),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display_frame,
            f"Inference: {inference_time_ms:.2f} ms",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display_frame,
            f"Left: {CLASS_NAMES[left_class]}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display_frame,
            f"Right: {CLASS_NAMES[right_class]}",
            (220, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # CAMERA CONNECTION STATUS
        # ====================================================

        if reader.connected:

            status_text = "RTSP: CONNECTED"

        else:

            status_text = "RTSP: RECONNECTING"


        cv2.putText(
            display_frame,
            status_text,
            (20, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q") or key == ord("Q"):

            print(
                "\nQ pressed."
            )

            break


    # ========================================================
    # CLEANUP
    # ========================================================

    print(
        "\nStopping RTSP reader..."
    )


    reader.stop()


    cv2.destroyAllWindows()


    print(
        "Program finished."
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()

