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

# Phase 1 model
BOX_MODEL_PATH = "efficientnet_b0_box_nobox_best.pth"

# Phase 2 model
EMPTY_FULL_MODEL_PATH = "new_efficientnet_b0_empty_full_final.pth"

EXPECTED_WIDTH = 3840
EXPECTED_HEIGHT = 2160

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# =========================================================
# VIOLATION CONFIGURATION
# =========================================================

# Condition must remain continuously active
# for this many seconds before violation is confirmed.
VIOLATION_DELAY_SECONDS = 60.0
# Short opposite-state interruption tolerated without resetting confirmation.
DISTRACTION_TOLERANCE_SECONDS = 30.0


# =========================================================
# ROIs
# Format: x, y, width, height
# =========================================================
#(1100, 456, 711, 837)
LEFT_ROI = (900, 456, 711, 837)

RIGHT_ROI = (2200, 510, 651, 891)


# =========================================================
# MODEL CLASSES
# =========================================================

BOX_CLASSES = {
    0: "BOX",
    1: "NOBOX"
}

EMPTY_FULL_CLASSES = {
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
print("       4TH PRESS MONITORING SYSTEM")
print("==========================================")

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# =========================================================
# CHECK MODEL FILES
# =========================================================

if not os.path.exists(BOX_MODEL_PATH):

    print("\nERROR: Box/NoBox model not found:")
    print(os.path.abspath(BOX_MODEL_PATH))
    exit()


if not os.path.exists(EMPTY_FULL_MODEL_PATH):

    print("\nERROR: Empty/Full model not found:")
    print(os.path.abspath(EMPTY_FULL_MODEL_PATH))
    exit()


# =========================================================
# MODEL LOADER
# =========================================================

def load_efficientnet(model_path):

    print(f"\nLoading: {model_path}")

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        2
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    # -----------------------------------------------------
    # Different checkpoint formats
    # -----------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint


    # -----------------------------------------------------
    # Remove DataParallel prefix if present
    # -----------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value


    model.load_state_dict(
        cleaned_state_dict
    )

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")

    return model


# =========================================================
# LOAD MODELS
# =========================================================

box_model = load_efficientnet(
    BOX_MODEL_PATH
)

empty_full_model = load_efficientnet(
    EMPTY_FULL_MODEL_PATH
)


# =========================================================
# INFERENCE TRANSFORM
# =========================================================

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

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )


        if self.cap.isOpened():

            width = int(
                self.cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                self.cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            print(
                f"RTSP connected: "
                f"{width}x{height}"
            )

            return True


        print("RTSP connection failed.")

        return False


    def update(self):

        while self.running:

            if (
                self.cap is None
                or not self.cap.isOpened()
            ):

                if not self.connect():

                    time.sleep(2)

                    continue


            ret, frame = self.cap.read()


            if ret:

                self.failed_reads = 0

                with self.lock:

                    # Keep only latest frame
                    self.latest_frame = frame


            else:

                self.failed_reads += 1

                if self.failed_reads >= self.MAX_FAILED_READS:

                    print(
                        "RTSP read failed. "
                        "Reconnecting..."
                    )

                    self.cap.release()

                    self.cap = None

                    self.failed_reads = 0

                    time.sleep(1)


    def start(self):

        self.running = True

        if not self.connect():

            print(
                "Initial RTSP connection failed."
            )


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
# CROP ROI
# =========================================================

def crop_roi(frame, roi):

    x, y, w, h = roi

    return frame[
        y:y + h,
        x:x + w
    ]


# =========================================================
# CLASSIFICATION
# =========================================================

@torch.no_grad()
def classify(model, crop):

    rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    tensor = inference_transform(
        image
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    output = model(tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )

    return (
        predicted.item(),
        confidence.item()
    )


# =========================================================
# BUSINESS LOGIC
# =========================================================

def determine_case(
    left_box_status,
    left_fill_status,
    right_box_status,
    right_fill_status
):

    # -----------------------------------------------------
    # CASE 1
    #
    # LEFT  = BOX + FULL
    # RIGHT = BOX + EMPTY
    #
    # Normal production start / feeding
    # -----------------------------------------------------

    if (
        left_box_status == "BOX"
        and left_fill_status == "FULL"
        and right_box_status == "BOX"
        and right_fill_status == "EMPTY"
    ):

        return (
            1,
            "NORMAL",
            False
        )


    # -----------------------------------------------------
    # CASE 2
    #
    # LEFT  = BOX + FULL
    # RIGHT = BOX + FULL
    #
    # Both boxes available
    # -----------------------------------------------------

    if (
        left_box_status == "BOX"
        and left_fill_status == "FULL"
        and right_box_status == "BOX"
        and right_fill_status == "FULL"
    ):

        return (
            2,
            "NORMAL",
            False
        )


    # -----------------------------------------------------
    # CASE 3
    #
    # LEFT  = BOX + EMPTY
    # RIGHT = BOX + EMPTY
    #
    # Raw material unavailable
    # -----------------------------------------------------

    if (
        left_box_status == "BOX"
        and left_fill_status == "EMPTY"
        and right_box_status == "BOX"
        and right_fill_status == "EMPTY"
    ):

        return (
            3,
            "RAW MATERIAL UNAVAILABLE",
            True
        )


    # -----------------------------------------------------
    # CASE 4
    #
    # LEFT  = BOX + EMPTY
    # RIGHT = BOX + FULL
    #
    # Raw material unavailable
    # -----------------------------------------------------

    if (
        left_box_status == "BOX"
        and left_fill_status == "EMPTY"
        and right_box_status == "BOX"
        and right_fill_status == "FULL"
    ):

        return (
            4,
            "RAW MATERIAL UNAVAILABLE",
            True
        )


    # -----------------------------------------------------
    # CASE 5
    #
    # LEFT  = NOBOX
    # RIGHT = BOX + EMPTY
    #
    # Sliding feed
    # NO VIOLATION
    # -----------------------------------------------------

    if (
        left_box_status == "NOBOX"
        and right_box_status == "BOX"
        and right_fill_status == "EMPTY"
    ):

        return (
            5,
            "SLIDING FEED",
            False
        )


    # -----------------------------------------------------
    # CASE 6
    #
    # LEFT  = NOBOX
    # RIGHT = BOX + FULL
    #
    # Sliding feed
    # NO VIOLATION
    # -----------------------------------------------------

    if (
        left_box_status == "NOBOX"
        and right_box_status == "BOX"
        and right_fill_status == "FULL"
    ):

        return (
            6,
            "SLIDING FEED",
            False
        )


    # -----------------------------------------------------
    # CASE 7
    #
    # LEFT  = BOX + FULL
    # RIGHT = NOBOX
    #
    # Produced-material box absent
    # -----------------------------------------------------

    if (
        left_box_status == "BOX"
        and left_fill_status == "FULL"
        and right_box_status == "NOBOX"
    ):

        return (
            7,
            "PRODUCED MATERIAL BOX NOT PRESENT",
            True
        )


    # -----------------------------------------------------
    # CASE 8
    #
    # LEFT  = BOX + EMPTY
    # RIGHT = NOBOX
    #
    # Raw material unavailable +
    # production box absent
    # -----------------------------------------------------

    if (
        left_box_status == "BOX"
        and left_fill_status == "EMPTY"
        and right_box_status == "NOBOX"
    ):

        return (
            8,
            "RAW MATERIAL UNAVAILABLE + "
            "PRODUCED MATERIAL BOX NOT PRESENT",
            True
        )


    # -----------------------------------------------------
    # CASE 9
    #
    # LEFT  = NOBOX
    # RIGHT = NOBOX
    #
    # Both boxes absent
    # -----------------------------------------------------

    if (
        left_box_status == "NOBOX"
        and right_box_status == "NOBOX"
    ):

        return (
            9,
            "BOTH BOXES NOT PRESENT",
            True
        )


    # -----------------------------------------------------
    # OTHER / TRANSITIONAL STATE
    # -----------------------------------------------------

    return (
        0,
        "TRANSITION / UNKNOWN",
        False
    )


# # =========================================================
# # VIOLATION TRACKER
# # =========================================================

class ViolationTracker:
    """
    Violation confirmation tracker with short-distraction tolerance.

    START:
      - First violation detection is saved as violation_start_time.
      - 60 seconds of accumulated violation time are required for confirmation.
      - Short normal interruptions (<= tolerance) do NOT reset the accumulated
        confirmation progress.

    END:
      - First normal detection after a confirmed violation is saved as
        violation_end_time_candidate.
      - 60 seconds of accumulated normal time are required to confirm the end.
      - Short violation interruptions (<= tolerance) do NOT reset the end timer.

    IMPORTANT:
      - The saved START is the first detection timestamp, e.g. 10:45:00.
      - The saved END is the first normal-condition timestamp, e.g. 10:47:00.
      - The 60-second delays are confirmation/stability delays only.
      - Final duration = END - START.
    """

    def __init__(self, delay_seconds, distraction_tolerance_seconds=5.0):
        self.delay_seconds = float(delay_seconds)
        self.distraction_tolerance_seconds = float(distraction_tolerance_seconds)

        self.current_case = None

        # -------------------------
        # Violation start tracking
        # -------------------------
        self.violation_active = False
        self.violation_confirmed = False
        self.violation_start_time = None
        self.violation_confirmed_time = None

        # Accumulated violation confirmation time.
        self.start_confirmed_seconds = 0.0
        self.last_start_update_time = None
        self.start_interruption_time = None

        # -------------------------
        # Violation end tracking
        # -------------------------
        self.end_pending = False
        self.violation_end_time = None
        self.violation_end_confirmed_time = None

        # Accumulated normal confirmation time.
        self.end_confirmed_seconds = 0.0
        self.last_end_update_time = None
        self.end_interruption_time = None

        # Final duration of the last confirmed violation.
        self.violation_duration = 0.0

    def _base_result(self):
        return {
            "confirmed": self.violation_confirmed,
            "pending": False,
            "ending": False,
            "elapsed": 0.0,
            "remaining": 0.0,
            "end_remaining": 0.0,
            "start_time": self.violation_start_time,
            "confirmed_time": self.violation_confirmed_time,
            "end_time": self.violation_end_time,
            "end_confirmed_time": self.violation_end_confirmed_time,
            "duration": self.violation_duration,
        }

    def update(self, case_number, case_description, is_violation):
        now = time.time()

        # =====================================================
        # 1. NO CONFIRMED VIOLATION YET: CONFIRM THE START
        # =====================================================
        if not self.violation_confirmed:

            # -------------------------------------------------
            # No violation currently visible
            # -------------------------------------------------
            if not is_violation:

                # If a violation candidate is being confirmed, allow a
                # short interruption without resetting progress.
                if self.violation_active:

                    if self.start_interruption_time is None:
                        self.start_interruption_time = now

                    interruption = now - self.start_interruption_time

                    # Short distraction: keep accumulated progress.
                    if interruption <= self.distraction_tolerance_seconds:
                        result = self._base_result()
                        result["pending"] = True
                        result["elapsed"] = self.start_confirmed_seconds
                        result["remaining"] = max(
                            0.0,
                            self.delay_seconds - self.start_confirmed_seconds
                        )
                        return result

                    # Distraction exceeded tolerance -> discard candidate.
                    self._reset_all()
                    return self._base_result()

                return self._base_result()

            # -------------------------------------------------
            # Violation is visible again
            # -------------------------------------------------
            if not self.violation_active:
                # FIRST violation detection timestamp.
                self.violation_active = True
                self.current_case = case_number
                self.violation_start_time = now
                self.start_confirmed_seconds = 0.0
                self.last_start_update_time = now
                self.start_interruption_time = None

            else:
                # Resume after a short interruption.
                if self.start_interruption_time is not None:
                    self.start_interruption_time = None

                if self.last_start_update_time is not None:
                    self.start_confirmed_seconds += (
                        now - self.last_start_update_time
                    )

                self.last_start_update_time = now

            # Confirm once 60 seconds of violation time have accumulated.
            if self.start_confirmed_seconds >= self.delay_seconds:
                self.violation_confirmed = True
                self.violation_confirmed_time = now

                # Start actual duration from the FIRST detection timestamp.
                self.violation_end_time = None
                self.violation_end_confirmed_time = None
                self.end_pending = False
                self.end_confirmed_seconds = 0.0
                self.last_end_update_time = None
                self.end_interruption_time = None

                result = self._base_result()
                result["confirmed"] = True
                result["elapsed"] = max(
                    0.0, now - self.violation_start_time
                )
                return result

            result = self._base_result()
            result["pending"] = True
            result["elapsed"] = self.start_confirmed_seconds
            result["remaining"] = max(
                0.0,
                self.delay_seconds - self.start_confirmed_seconds
            )
            return result

        # =====================================================
        # 2. CONFIRMED VIOLATION: WAIT FOR CONFIRMED END
        # =====================================================

        # -----------------------------------------------------
        # Violation is still present
        # -----------------------------------------------------
        if is_violation:

            # Cancel an end-confirmation attempt after a short
            # distraction and continue the existing violation.
            if self.end_pending:
                if self.end_interruption_time is None:
                    self.end_interruption_time = now

                interruption = now - self.end_interruption_time

                if interruption <= self.distraction_tolerance_seconds:
                    result = self._base_result()
                    result["confirmed"] = True
                    result["elapsed"] = max(
                        0.0, now - self.violation_start_time
                    )
                    return result

                # Long interruption: cancel the pending end.
                self.end_pending = False
                self.end_confirmed_seconds = 0.0
                self.last_end_update_time = None
                self.end_interruption_time = None
                self.violation_end_time = None

            result = self._base_result()
            result["confirmed"] = True
            result["elapsed"] = max(
                0.0, now - self.violation_start_time
            )
            return result

        # -----------------------------------------------------
        # Condition became NORMAL: start end confirmation
        # -----------------------------------------------------
        if not self.end_pending:
            # FIRST normal detection timestamp.
            self.violation_end_time = now
            self.end_pending = True
            self.end_confirmed_seconds = 0.0
            self.last_end_update_time = now
            self.end_interruption_time = None

            result = self._base_result()
            result["confirmed"] = True
            result["ending"] = True
            result["elapsed"] = max(
                0.0, now - self.violation_start_time
            )
            result["end_remaining"] = self.delay_seconds
            return result

        # -----------------------------------------------------
        # Continue normal/end confirmation
        # -----------------------------------------------------
        if self.end_interruption_time is not None:
            # Normal has returned after a short interruption.
            self.end_interruption_time = None

        if self.last_end_update_time is not None:
            self.end_confirmed_seconds += (
                now - self.last_end_update_time
            )

        self.last_end_update_time = now

        # End confirmed after 60 seconds of accumulated normal time.
        if self.end_confirmed_seconds >= self.delay_seconds:
            self.violation_end_confirmed_time = now

            # IMPORTANT:
            # Duration uses the FIRST normal detection timestamp,
            # not the time at which the 60-sec end confirmation finishes.
            self.violation_duration = max(
                0.0,
                self.violation_end_time - self.violation_start_time
            )

            result = self._base_result()
            result["confirmed"] = False
            result["ending"] = False
            result["elapsed"] = self.violation_duration
            result["remaining"] = 0.0
            result["end_remaining"] = 0.0

            # Keep the timestamps in this result frame, then reset the
            # active state so the next violation can start cleanly.
            result["start_time"] = self.violation_start_time
            result["end_time"] = self.violation_end_time
            result["confirmed_time"] = self.violation_confirmed_time
            result["end_confirmed_time"] = self.violation_end_confirmed_time
            result["duration"] = self.violation_duration

            self._reset_all(keep_last_result=True)
            return result

        result = self._base_result()
        result["confirmed"] = True
        result["ending"] = True
        result["elapsed"] = max(
            0.0, now - self.violation_start_time
        )
        result["end_remaining"] = max(
            0.0,
            self.delay_seconds - self.end_confirmed_seconds
        )
        return result

    def _reset_all(self, keep_last_result=False):
        self.violation_active = False
        self.violation_confirmed = False
        self.current_case = None

        self.violation_start_time = None
        self.violation_confirmed_time = None

        self.start_confirmed_seconds = 0.0
        self.last_start_update_time = None
        self.start_interruption_time = None

        self.end_pending = False
        self.violation_end_time = None
        self.violation_end_confirmed_time = None

        self.end_confirmed_seconds = 0.0
        self.last_end_update_time = None
        self.end_interruption_time = None

        if not keep_last_result:
            self.violation_duration = 0.0


# =========================================================
# CREATE VIOLATION TRACKER
# =========================================================

violation_tracker = ViolationTracker(
    VIOLATION_DELAY_SECONDS,
    DISTRACTION_TOLERANCE_SECONDS
)


# =========================================================
# START RTSP
# =========================================================

reader = RTSPReader(
    RTSP_URL
)

reader.start()


print("\n==========================================")
print("4TH PRESS PIPELINE STARTED")
print(
    f"Violation confirmation delay: "
    f"{VIOLATION_DELAY_SECONDS} seconds"
)
print("Press Q to quit")
print("==========================================\n")


# =========================================================
# FPS
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


        # =================================================
        # FRAME SIZE
        # =================================================

        height, width = frame.shape[:2]


        # =================================================
        # CROP ROIs
        # =================================================

        left_crop = crop_roi(
            frame,
            LEFT_ROI
        )

        right_crop = crop_roi(
            frame,
            RIGHT_ROI
        )

 
        # =================================================
        # PHASE 1
        # BOX / NOBOX
        # =================================================

        left_box_id, left_box_conf = classify(
            box_model,
            left_crop
        )

        right_box_id, right_box_conf = classify(
            box_model,
            right_crop
        )


        left_box_status = BOX_CLASSES[
            left_box_id
        ]

        right_box_status = BOX_CLASSES[
            right_box_id
        ]


        # =================================================
        # PHASE 2
        # EMPTY / FULL
        #
        # Only classify if BOX exists
        # =================================================

        if left_box_status == "BOX":

            left_fill_id, left_fill_conf = classify(
                empty_full_model,
                left_crop
            )

            left_fill_status = EMPTY_FULL_CLASSES[
                left_fill_id
            ]

        else:

            left_fill_status = "N/A"

            left_fill_conf = 0.0


        if right_box_status == "BOX":

            right_fill_id, right_fill_conf = classify(
                empty_full_model,
                right_crop
            )

            right_fill_status = EMPTY_FULL_CLASSES[
                right_fill_id
            ]

        else:

            right_fill_status = "N/A"

            right_fill_conf = 0.0


        # =================================================
        # DETERMINE BUSINESS CASE
        # =================================================

        (
            case_number,
            case_description,
            is_violation
        ) = determine_case(

            left_box_status,
            left_fill_status,

            right_box_status,
            right_fill_status
        )


        # =================================================
        # UPDATE VIOLATION TIMER
        # =================================================

        violation_info = violation_tracker.update(

            case_number,

            case_description,

            is_violation
        )


        violation_confirmed = (
            violation_info["confirmed"]
        )

        violation_elapsed = (
            violation_info["elapsed"]
        )

        violation_remaining = (
            violation_info["remaining"]
        )

        violation_ending = (
            violation_info.get("ending", False)
        )

        violation_end_remaining = (
            violation_info.get("end_remaining", 0.0)
        )

        violation_start_time = (
            violation_info.get("start_time")
        )

        violation_end_time = (
            violation_info.get("end_time")
        )

        violation_duration = (
            violation_info.get("duration", 0.0)
        )


        # =================================================
        # DRAW LEFT ROI
        # =================================================

        x, y, w, h = LEFT_ROI

        if left_box_status == "NOBOX":
            left_color = (0, 0, 255)          # RED
        elif left_fill_status == "EMPTY":
            left_color = (0, 165, 255)        # ORANGE
        else:
            left_color = (0, 255, 0)          # GREEN

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            left_color,
            4
        )


        # =================================================
        # DRAW RIGHT ROI
        # =================================================

        x, y, w, h = RIGHT_ROI

        if right_box_status == "NOBOX":
            right_color = (0, 0, 255)         # RED
        elif right_fill_status == "EMPTY":
            right_color = (0, 165, 255)       # ORANGE
        else:
            right_color = (0, 255, 0)         # GREEN

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            right_color,
            4
)


        # =================================================
        # DISPLAY RESIZE
        # =================================================

        display = cv2.resize(
            frame,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )
        )


        scale_x = (
            DISPLAY_WIDTH / width
        )

        scale_y = (
            DISPLAY_HEIGHT / height
        )


        # =================================================
        # LEFT STATUS
        # =================================================

        if left_box_status == "BOX":

            left_text = (
                f"LEFT: BOX | "
                f"{left_fill_status} | "
                f"{left_fill_conf * 100:.1f}%"
            )

        else:

            left_text = (
                f"LEFT: NO BOX | "
                f"{left_box_conf * 100:.1f}%"
            )


        cv2.putText(
            display,
            left_text,
            (
                int(
                    LEFT_ROI[0] * scale_x
                ),
                int(
                    LEFT_ROI[1] * scale_y
                ) - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            left_color,
            2,
            cv2.LINE_AA
        )


        # =================================================
        # RIGHT STATUS
        # =================================================

        if right_box_status == "BOX":

            right_text = (
                f"RIGHT: BOX | "
                f"{right_fill_status} | "
                f"{right_fill_conf * 100:.1f}%"
            )

        else:

            right_text = (
                f"RIGHT: NO BOX | "
                f"{right_box_conf * 100:.1f}%"
            )


        cv2.putText(
            display,
            right_text,
            (
                int(
                    RIGHT_ROI[0] * scale_x
                ),
                int(
                    RIGHT_ROI[1] * scale_y
                ) - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            right_color,
            2,
            cv2.LINE_AA
        )


        # =================================================
        # CASE INFORMATION
        # =================================================

        cv2.putText(
            display,
            f"CASE: {case_number}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            display,
            case_description,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # =================================================
        # VIOLATION STATUS
        # =================================================

        if violation_ending:

            cv2.putText(
                display,
                "VIOLATION ENDING - CONFIRMING NORMAL",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Normal confirmation: "
                f"{violation_end_remaining:.1f}s",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

            if violation_start_time is not None:
                cv2.putText(
                    display,
                    "Violation started: " +
                    time.strftime(
                        "%H:%M:%S",
                        time.localtime(violation_start_time)
                    ),
                    (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            if violation_end_time is not None:
                cv2.putText(
                    display,
                    "Violation end candidate: " +
                    time.strftime(
                        "%H:%M:%S",
                        time.localtime(violation_end_time)
                    ),
                    (20, 245),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        elif violation_confirmed:

            cv2.putText(
                display,
                "VIOLATION CONFIRMED",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Duration: {violation_elapsed:.1f}s",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            if violation_start_time is not None:
                cv2.putText(
                    display,
                    "Violation started: " +
                    time.strftime(
                        "%H:%M:%S",
                        time.localtime(violation_start_time)
                    ),
                    (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        elif violation_info.get("pending", False):

            cv2.putText(
                display,
                "VIOLATION PENDING",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.80,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Confirmation remaining: "
                f"{violation_remaining:.1f}s",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

        else:

            cv2.putText(
                display,
                "STATUS: NORMAL",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


        # =================================================
        # FPS
        # =================================================

        fps_counter += 1

        elapsed_fps = (
            time.time()
            - fps_start_time
        )


        if elapsed_fps >= 1.0:

            display_fps = (
                fps_counter
                / elapsed_fps
            )

            fps_counter = 0

            fps_start_time = time.time()


        cv2.putText(
            display,
            f"FPS: {display_fps:.1f}",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # =================================================
        # SHOW
        # =================================================

        cv2.imshow(
            "4th Press - Box Monitoring",
            display
        )


        # =================================================
        # QUIT
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


finally:

    print("\nStopping pipeline...")

    reader.stop()

    cv2.destroyAllWindows()

    print("Pipeline stopped.")