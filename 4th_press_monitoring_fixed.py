import cv2
import torch
import threading
import time
import os

from PIL import Image
from torchvision import models, transforms
from database import insert_violation, update_violation

# =========================================================
# CONFIGURATION
# =========================================================

RTSP_URL = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"


# =========================================================
# MODEL PATHS
# =========================================================

# Phase 1:
# Detect whether LEFT box is present or not
BOX_MODEL_PATH = "efficientnet_b0_box_nobox_best.pth"

# Phase 2:
# If box is present, detect EMPTY or FULL
EMPTY_FULL_MODEL_PATH = "new_efficientnet_b0_empty_full_final.pth"


# =========================================================
# EXPECTED CAMERA RESOLUTION
# =========================================================

EXPECTED_WIDTH = 3840
EXPECTED_HEIGHT = 2160


# =========================================================
# DISPLAY RESOLUTION
# =========================================================

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# =========================================================
# VIOLATION CONFIGURATION
# =========================================================

# Violation must remain active for this many seconds
# before it is confirmed.
VIOLATION_DELAY_SECONDS = 60.0


# Short opposite-state interruption tolerated
# without resetting the confirmation.
DISTRACTION_TOLERANCE_SECONDS = 30.0


# =========================================================
# LEFT ROI
# Format:
# x, y, width, height
# =========================================================

LEFT_ROI = (1000, 456, 711, 837)


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

    # Create EfficientNet-B0
    model = models.efficientnet_b0(
        weights=None
    )

    # Two output classes
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        2
    )

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    # -----------------------------------------------------
    # Handle different checkpoint formats
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
    # Remove DataParallel "module." prefix
    # -----------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value


    # Load weights
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

print("\nLoading Box/NoBox model...")

box_model = load_efficientnet(
    BOX_MODEL_PATH
)


print("\nLoading Empty/Full model...")

empty_full_model = load_efficientnet(
    EMPTY_FULL_MODEL_PATH
)


# =========================================================
# INFERENCE TRANSFORM
# =========================================================

inference_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
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


    # -----------------------------------------------------
    # CONNECT TO RTSP
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # CONTINUOUS RTSP UPDATE
    # -----------------------------------------------------

    def update(self):

        while self.running:

            # ---------------------------------------------
            # Make sure camera is connected
            # ---------------------------------------------

            if (
                self.cap is None
                or not self.cap.isOpened()
            ):

                if not self.connect():

                    time.sleep(2)

                    continue


            # ---------------------------------------------
            # Read frame
            # ---------------------------------------------

            ret, frame = self.cap.read()


            if ret:

                self.failed_reads = 0

                with self.lock:

                    # Keep only latest frame
                    self.latest_frame = frame


            else:

                self.failed_reads += 1

                if (
                    self.failed_reads
                    >= self.MAX_FAILED_READS
                ):

                    print(
                        "RTSP read failed. "
                        "Reconnecting..."
                    )

                    self.cap.release()

                    self.cap = None

                    self.failed_reads = 0

                    time.sleep(1)


    # -----------------------------------------------------
    # START READER
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # GET LATEST FRAME
    # -----------------------------------------------------

    def get_frame(self):

        with self.lock:

            if self.latest_frame is None:

                return None

            return self.latest_frame.copy()


    # -----------------------------------------------------
    # STOP READER
    # -----------------------------------------------------

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

    # BGR -> RGB
    rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL
    image = Image.fromarray(
        rgb
    )

    # Apply preprocessing
    tensor = inference_transform(
        image
    )

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Move to GPU/CPU
    tensor = tensor.to(DEVICE)

    # Model inference
    output = model(
        tensor
    )

    # Convert logits to probabilities
    probabilities = torch.softmax(
        output,
        dim=1
    )

    # Get highest probability
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
#
# ONLY THREE CASES NOW
#
# CASE 1:
# LEFT = NOBOX
# -> VIOLATION
#
# CASE 2:
# LEFT = BOX + EMPTY
# -> VIOLATION
#
# CASE 3:
# LEFT = BOX + FULL
# -> NORMAL
#
# =========================================================

def determine_case(
    left_box_status,
    left_fill_status
):

    # -----------------------------------------------------
    # CASE 1
    #
    # LEFT BOX IS NOT PRESENT
    # -----------------------------------------------------

    if left_box_status == "NOBOX":

        return (
            1,
            "LEFT BOX NOT PRESENT",
            True
        )


    # -----------------------------------------------------
    # CASE 2
    #
    # LEFT BOX IS PRESENT BUT EMPTY
    # -----------------------------------------------------

    if (
        left_box_status == "BOX"
        and left_fill_status == "EMPTY"
    ):

        return (
            2,
            "LEFT BOX EMPTY",
            True
        )


    # -----------------------------------------------------
    # CASE 3
    #
    # LEFT BOX IS PRESENT AND MATERIAL IS AVAILABLE
    # -----------------------------------------------------

    if (
        left_box_status == "BOX"
        and left_fill_status == "FULL"
    ):

        return (
            3,
            "MATERIAL AVAILABLE",
            False
        )


    # -----------------------------------------------------
    # UNKNOWN / TRANSITION
    # -----------------------------------------------------

    return (
        0,
        "TRANSITION / UNKNOWN",
        False
    )


# =========================================================
# VIOLATION TRACKER
# =========================================================

class ViolationTracker:
    """
    Violation tracking logic.

    START:
        - Save the FIRST violation detection timestamp.
        - Confirm violation after 60 seconds of violation state.
        - Short normal interruptions (<= tolerance) do not reset
          the violation candidate.

    END:
        - After violation is confirmed, save the FIRST normal detection
          timestamp as violation_end_time.
        - Confirm normal after 60 seconds of normal state.
        - Short violation interruptions (<= tolerance) do not cancel
          the normal confirmation.

    DATABASE:
        - INSERT happens once at first violation detection.
        - UPDATE happens once after normal is confirmed.
        - Saved duration = first normal detection - first violation detection.
          Therefore the 60-second confirmation delays are NOT included.
    """

    def __init__(self, delay_seconds, distraction_tolerance_seconds=30.0):

        self.delay_seconds = float(delay_seconds)
        self.distraction_tolerance_seconds = float(
            distraction_tolerance_seconds
        )

        # -----------------------------------------------------
        # VIOLATION START
        # -----------------------------------------------------

        self.violation_active = False
        self.violation_confirmed = False

        self.violation_start_time = None
        self.violation_confirmed_time = None

        # Used for short normal interruption before violation confirmation
        self.start_interruption_time = None

        # -----------------------------------------------------
        # VIOLATION END
        # -----------------------------------------------------

        self.end_pending = False

        # FIRST normal detection after confirmed violation
        self.violation_end_time = None

        # Time at which normal became confirmed
        self.violation_end_confirmed_time = None

        # Used for short violation interruption during normal confirmation
        self.end_interruption_time = None

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------

        self.violation_duration = 0.0

        # Database row ID
        self.database_violation_id = None

    def _base_result(self):

        elapsed = 0.0

        if (
            self.violation_confirmed
            and self.violation_start_time is not None
        ):
            elapsed = max(
                0.0,
                time.time() - self.violation_start_time
            )

        return {
            "confirmed": self.violation_confirmed,
            "pending": False,
            "ending": self.end_pending,
            "elapsed": elapsed,
            "remaining": 0.0,
            "end_remaining": 0.0,
            "start_time": self.violation_start_time,
            "confirmed_time": self.violation_confirmed_time,
            "end_time": self.violation_end_time,
            "end_confirmed_time": self.violation_end_confirmed_time,
            "duration": self.violation_duration
        }

    def update(self, case_number, case_description, is_violation):

        now = time.time()

        # =====================================================
        # 1. VIOLATION HAS NOT BEEN CONFIRMED YET
        # =====================================================

        if not self.violation_confirmed:

            # -------------------------------------------------
            # CURRENT CONDITION = NORMAL
            # -------------------------------------------------

            if not is_violation:

                # No violation candidate exists.
                if not self.violation_active:
                    return self._base_result()

                # A violation candidate exists but current frame
                # is temporarily normal.
                if self.start_interruption_time is None:
                    self.start_interruption_time = now

                    print(
                        "[VIOLATION] Short normal interruption started."
                    )

                interruption = (
                    now - self.start_interruption_time
                )

                # Keep candidate during short interruption.
                if interruption <= self.distraction_tolerance_seconds:

                    result = self._base_result()
                    result["pending"] = True

                    elapsed = max(
                        0.0,
                        now - self.violation_start_time
                    )

                    result["elapsed"] = elapsed
                    result["remaining"] = max(
                        0.0,
                        self.delay_seconds -
                        elapsed
                    )

                    return result

                # Normal condition lasted too long.
                # Cancel the unconfirmed violation.
                print(
                    "[VIOLATION] Start candidate cancelled "
                    "after long normal interruption."
                )

                self._reset_all()

                return self._base_result()

            # -------------------------------------------------
            # CURRENT CONDITION = VIOLATION
            # -------------------------------------------------

            if not self.violation_active:

                # FIRST VIOLATION DETECTION
                self.violation_active = True
                self.violation_start_time = now

                print(
                    "\n[VIOLATION] FIRST VIOLATION DETECTED"
                )
                print(
                    "[VIOLATION] Start:",
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(
                            self.violation_start_time
                        )
                    )
                )

                # INSERT ONLY ONCE
                self.database_violation_id = insert_violation(
                    violation_type=case_description,
                    start_timestamp=self.violation_start_time
                )

                print(
                    f"[VIOLATION] Database ID: "
                    f"{self.database_violation_id}"
                )

                self.start_interruption_time = None

            else:

                # Violation has returned after a short normal
                # interruption.
                if self.start_interruption_time is not None:

                    print(
                        "[VIOLATION] Violation returned; "
                        "continuing confirmation."
                    )

                    self.start_interruption_time = None

            # -------------------------------------------------
            # CONFIRM VIOLATION USING ORIGINAL START TIME
            # -------------------------------------------------

            elapsed = (
                now - self.violation_start_time
            )

            if elapsed >= self.delay_seconds:

                self.violation_confirmed = True
                self.violation_confirmed_time = now

                print(
                    "\n[VIOLATION] ============================="
                )
                print(
                    "[VIOLATION] VIOLATION CONFIRMED"
                )
                print(
                    f"[VIOLATION] DB ID: "
                    f"{self.database_violation_id}"
                )
                print(
                    f"[VIOLATION] Start: "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.violation_start_time))}"
                )
                print(
                    "[VIOLATION] ============================="
                )

                result = self._base_result()
                result["confirmed"] = True
                result["elapsed"] = elapsed
                result["remaining"] = 0.0

                return result

            # -------------------------------------------------
            # VIOLATION STILL PENDING
            # -------------------------------------------------

            result = self._base_result()

            result["pending"] = True
            result["elapsed"] = elapsed
            result["remaining"] = max(
                0.0,
                self.delay_seconds - elapsed
            )

            return result

        # =====================================================
        # 2. CONFIRMED VIOLATION
        # WAIT FOR NORMAL CONDITION
        # =====================================================

        # -----------------------------------------------------
        # VIOLATION IS STILL PRESENT
        # -----------------------------------------------------

        if is_violation:

            # Normal was previously detected and is being
            # confirmed, but violation returned.
            if self.end_pending:

                if self.end_interruption_time is None:
                    self.end_interruption_time = now

                    print(
                        "[VIOLATION] Violation interruption during "
                        "normal confirmation."
                    )

                interruption = (
                    now - self.end_interruption_time
                )

                # Short violation interruption:
                # keep normal confirmation candidate.
                if interruption <= self.distraction_tolerance_seconds:

                    result = self._base_result()
                    result["confirmed"] = True
                    result["ending"] = True

                    result["elapsed"] = max(
                        0.0,
                        now - self.violation_start_time
                    )

                    result["end_remaining"] = max(
                        0.0,
                        self.delay_seconds -
                        (
                            now -
                            self.violation_end_time
                        )
                    )

                    return result

                # Long violation interruption:
                # cancel END confirmation.
                print(
                    "[VIOLATION] Normal confirmation cancelled "
                    "after long violation interruption."
                )

                self.end_pending = False
                self.end_interruption_time = None
                self.violation_end_time = None

            result = self._base_result()
            result["confirmed"] = True
            result["ending"] = False
            result["elapsed"] = max(
                0.0,
                now - self.violation_start_time
            )

            return result

        # =====================================================
        # 3. CURRENT CONDITION = NORMAL
        # =====================================================

        # -----------------------------------------------------
        # FIRST NORMAL DETECTION
        # -----------------------------------------------------

        if not self.end_pending:

            self.violation_end_time = now
            self.end_pending = True
            self.end_interruption_time = None

            print(
                "\n[VIOLATION] ============================="
            )
            print(
                "[VIOLATION] FIRST NORMAL DETECTED"
            )
            print(
                f"[VIOLATION] DB ID: "
                f"{self.database_violation_id}"
            )
            print(
                f"[VIOLATION] End candidate: "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.violation_end_time))}"
            )
            print(
                "[VIOLATION] Normal confirmation started: 60 seconds"
            )
            print(
                "[VIOLATION] ============================="
            )

        else:

            # Normal has continued.
            if self.end_interruption_time is not None:

                print(
                    "[VIOLATION] Normal returned; "
                    "continuing END confirmation."
                )

                self.end_interruption_time = None

        # =====================================================
        # NORMAL CONFIRMATION
        # =====================================================

        normal_elapsed = (
            now - self.violation_end_time
        )

        # -----------------------------------------------------
        # NORMAL CONDITION CONFIRMED
        # -----------------------------------------------------

        if normal_elapsed >= self.delay_seconds:

            self.violation_end_confirmed_time = now

            # IMPORTANT:
            # Duration is calculated from:
            #
            # FIRST VIOLATION DETECTION
            #          TO
            # FIRST NORMAL DETECTION
            #
            # The 60-second normal confirmation delay is excluded.

            self.violation_duration = max(
                0.0,
                self.violation_end_time -
                self.violation_start_time
            )

            print(
                "\n[VIOLATION] ============================="
            )
            print(
                "[VIOLATION] NORMAL CONDITION CONFIRMED"
            )
            print(
                f"[VIOLATION] DB ID: "
                f"{self.database_violation_id}"
            )
            print(
                f"[VIOLATION] Start: "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.violation_start_time))}"
            )
            print(
                f"[VIOLATION] End: "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.violation_end_time))}"
            )
            print(
                f"[VIOLATION] Duration: "
                f"{self.violation_duration:.2f} seconds"
            )
            print(
                "[VIOLATION] Calling database UPDATE..."
            )

            # =================================================
            # UPDATE DATABASE
            # =================================================

            if self.database_violation_id is not None:

                update_result = update_violation(
                    violation_id=self.database_violation_id,
                    end_timestamp=self.violation_end_time,
                    duration_seconds=self.violation_duration
                )

                print(
                    f"[VIOLATION] Database update result: "
                    f"{update_result}"
                )

            else:

                print(
                    "[VIOLATION] ERROR: database_violation_id "
                    "is None. Database cannot be updated."
                )

            # Prepare result BEFORE resetting tracker.
            result = self._base_result()

            result["confirmed"] = False
            result["ending"] = False
            result["elapsed"] = self.violation_duration
            result["remaining"] = 0.0
            result["end_remaining"] = 0.0
            result["start_time"] = self.violation_start_time
            result["end_time"] = self.violation_end_time
            result["confirmed_time"] = self.violation_confirmed_time
            result["end_confirmed_time"] = (
                self.violation_end_confirmed_time
            )
            result["duration"] = self.violation_duration

            # Reset AFTER database update.
            self._reset_all()

            return result

        # =====================================================
        # NORMAL CONDITION STILL BEING CONFIRMED
        # =====================================================

        result = self._base_result()

        result["confirmed"] = True
        result["ending"] = True

        result["elapsed"] = max(
            0.0,
            now - self.violation_start_time
        )

        result["end_remaining"] = max(
            0.0,
            self.delay_seconds - normal_elapsed
        )

        return result

    def _reset_all(self):

        self.violation_active = False
        self.violation_confirmed = False

        self.violation_start_time = None
        self.violation_confirmed_time = None

        self.start_interruption_time = None

        self.end_pending = False

        self.violation_end_time = None
        self.violation_end_confirmed_time = None

        self.end_interruption_time = None

        self.violation_duration = 0.0

        # IMPORTANT:
        # Clear DB ID only AFTER the row has been updated.
        self.database_violation_id = None


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
print(
    f"Interruption tolerance: "
    f"{DISTRACTION_TOLERANCE_SECONDS} seconds"
)
print("Monitoring: LEFT BOX ONLY")
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


        # =================================================
        # GET FRAME
        # =================================================

        frame = reader.get_frame()


        if frame is None:

            time.sleep(0.01)

            continue


        # =================================================
        # FRAME SIZE
        # =================================================

        height, width = frame.shape[:2]


        # =================================================
        # CROP LEFT ROI
        # =================================================

        left_crop = crop_roi(
            frame,
            LEFT_ROI
        )


        # =================================================
        # PHASE 1
        # BOX / NOBOX
        # =================================================

        left_box_id, left_box_conf = classify(

            box_model,

            left_crop
        )


        left_box_status = BOX_CLASSES[

            left_box_id
        ]


        # =================================================
        # PHASE 2
        # EMPTY / FULL
        #
        # ONLY RUN IF BOX EXISTS
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


        # =================================================
        # DETERMINE BUSINESS CASE
        # =================================================

        (
            case_number,
            case_description,
            is_violation

        ) = determine_case(

            left_box_status,

            left_fill_status
        )


        # =================================================
        # UPDATE VIOLATION TRACKER
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

            violation_info.get(
                "ending",
                False
            )
        )


        violation_end_remaining = (

            violation_info.get(
                "end_remaining",
                0.0
            )
        )


        violation_start_time = (

            violation_info.get(
                "start_time"
            )
        )


        violation_end_time = (

            violation_info.get(
                "end_time"
            )
        )


        violation_duration = (

            violation_info.get(
                "duration",
                0.0
            )
        )


        # =================================================
        # DRAW LEFT ROI
        # =================================================

        x, y, w, h = LEFT_ROI


        # -------------------------------------------------
        # ROI COLOR
        # -------------------------------------------------

        if left_box_status == "NOBOX":

            # RED
            left_color = (
                0,
                0,
                255
            )

        elif left_fill_status == "EMPTY":

            # ORANGE
            left_color = (
                0,
                165,
                255
            )

        else:

            # GREEN
            left_color = (
                0,
                255,
                0
            )


        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            left_color,

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

            DISPLAY_WIDTH
            / width
        )


        scale_y = (

            DISPLAY_HEIGHT
            / height
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
                    LEFT_ROI[0]
                    * scale_x
                ),

                int(
                    LEFT_ROI[1]
                    * scale_y
                ) - 15
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.70,

            left_color,

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


            # ---------------------------------------------
            # VIOLATION ENDING
            # ---------------------------------------------

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


            if (
                violation_start_time
                is not None
            ):

                cv2.putText(

                    display,

                    "Violation started: "
                    + time.strftime(

                        "%H:%M:%S",

                        time.localtime(
                            violation_start_time
                        )
                    ),

                    (20, 215),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    2,

                    cv2.LINE_AA
                )


            if (
                violation_end_time
                is not None
            ):

                cv2.putText(

                    display,

                    "Violation end candidate: "
                    + time.strftime(

                        "%H:%M:%S",

                        time.localtime(
                            violation_end_time
                        )
                    ),

                    (20, 245),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    2,

                    cv2.LINE_AA
                )


        elif violation_confirmed:


            # ---------------------------------------------
            # VIOLATION CONFIRMED
            # ---------------------------------------------

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

                f"Duration: "
                f"{violation_elapsed:.1f}s",

                (20, 180),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (0, 0, 255),

                2,

                cv2.LINE_AA
            )


            if (
                violation_start_time
                is not None
            ):

                cv2.putText(

                    display,

                    "Violation started: "
                    + time.strftime(

                        "%H:%M:%S",

                        time.localtime(
                            violation_start_time
                        )
                    ),

                    (20, 215),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    2,

                    cv2.LINE_AA
                )


        elif violation_info.get(
            "pending",
            False
        ):


            # ---------------------------------------------
            # VIOLATION PENDING
            # ---------------------------------------------

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


            # ---------------------------------------------
            # NORMAL
            # ---------------------------------------------

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

            "4th Press - LEFT Box Monitoring",

            display
        )


        # =================================================
        # QUIT
        # =================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


# =========================================================
# CLEANUP
# =========================================================

finally:

    print(
        "\nStopping pipeline..."
    )

    reader.stop()

    cv2.destroyAllWindows()

    print(
        "Pipeline stopped."
    )