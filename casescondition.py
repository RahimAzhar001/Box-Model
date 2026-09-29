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


# =========================================================
# ROIs
# Format: x, y, width, height
# =========================================================
#(1100, 456, 711, 837)

LEFT_ROI = (800, 456, 711, 837)

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

# class ViolationTracker:

#     def __init__(self, delay_seconds):

#         self.delay_seconds = delay_seconds

#         self.current_case = None

#         self.case_start_time = None

#         self.violation_confirmed = False


#     def update(
#         self,
#         case_number,
#         case_description,
#         is_violation
#     ):

#         current_time = time.time()


#         # -------------------------------------------------
#         # Normal / non-violation case
#         # -------------------------------------------------

#         if not is_violation:

#             self.current_case = None

#             self.case_start_time = None

#             self.violation_confirmed = False

#             return {
#                 "confirmed": False,
#                 "elapsed": 0.0,
#                 "remaining": 0.0
#             }


#         # -------------------------------------------------
#         # New violation case
#         # -------------------------------------------------

#         if case_number != self.current_case:

#             self.current_case = case_number

#             self.case_start_time = current_time

#             self.violation_confirmed = False
# =========================================================
# VIOLATION TRACKER
# =========================================================

class ViolationTracker:
    """
    Violation timing logic:

    1. When a violation is first detected, start a 60-second
       confirmation countdown. This time is SHOWN on screen but
       is NOT included in the saved violation duration.

    2. If the violation remains active for the full delay, the
       current time becomes the actual saved violation_start_time.

    3. After the violation is confirmed, when the condition first
       becomes NORMAL, start another 60-second end-confirmation
       countdown. This countdown is also shown on screen.

    4. If NORMAL remains stable for the full delay, the current
       time becomes violation_end_time.

    5. Duration = violation_end_time - violation_start_time.
       Therefore, the initial 60-second start delay is excluded,
       while the 60-second end confirmation is included in the
       saved end timestamp.
    """

    def __init__(self, delay_seconds):
        self.delay_seconds = delay_seconds

        # Current violation/case information
        self.current_case = None
        self.case_description = None

        # First detection of a violation (NOT the saved start time)
        self.pending_violation_start_time = None

        # Actual saved violation interval
        self.violation_start_time = None
        self.violation_end_time = None

        # Confirmation timestamps
        self.violation_confirmed_time = None
        self.normal_start_time = None

        # State flags
        self.violation_active = False
        self.violation_confirmed = False
        self.ending_pending = False

        # Final duration
        self.violation_duration = 0.0

    @staticmethod
    def _format_time(timestamp):
        if timestamp is None:
            return "--:--:--"
        return time.strftime("%H:%M:%S", time.localtime(timestamp))

    def _result(
        self,
        state,
        elapsed=0.0,
        remaining=0.0,
        duration=0.0,
        end_confirmed=False
    ):
        return {
            "state": state,
            "confirmed": self.violation_confirmed,
            "elapsed": elapsed,
            "remaining": remaining,
            "duration": duration,
            "end_confirmed": end_confirmed,
            "start_time": self.violation_start_time,
            "confirmed_time": self.violation_confirmed_time,
            "end_time": self.violation_end_time,
            "normal_start_time": self.normal_start_time,
            "start_time_text": self._format_time(self.violation_start_time),
            "confirmed_time_text": self._format_time(self.violation_confirmed_time),
            "end_time_text": self._format_time(self.violation_end_time),
        }

    def _reset(self):
        self.current_case = None
        self.case_description = None
        self.pending_violation_start_time = None
        self.violation_start_time = None
        self.violation_end_time = None
        self.violation_confirmed_time = None
        self.normal_start_time = None
        self.violation_active = False
        self.violation_confirmed = False
        self.ending_pending = False
        self.violation_duration = 0.0

    def update(self, case_number, case_description, is_violation):
        current_time = time.time()

        # =====================================================
        # 1. NO ACTIVE VIOLATION
        # =====================================================
        if not self.violation_active:

            if not is_violation:
                self._reset()
                return self._result("normal")

            # -------------------------------------------------
            # First frame where violation is detected.
            # This timestamp is ONLY used for the pending timer.
            # It is deliberately NOT saved as violation_start_time.
            # -------------------------------------------------
            self.current_case = case_number
            self.case_description = case_description
            self.pending_violation_start_time = current_time
            self.violation_active = True
            self.violation_confirmed = False
            self.ending_pending = False

            return self._result(
                "pending_start",
                elapsed=0.0,
                remaining=self.delay_seconds
            )

        # =====================================================
        # 2. VIOLATION IS IN START-CONFIRMATION PERIOD
        # =====================================================
        if not self.violation_confirmed:

            # If the violation disappears before confirmation,
            # discard it completely. No violation record is saved.
            if not is_violation:
                self._reset()
                return self._result("normal")

            elapsed = (
                current_time - self.pending_violation_start_time
            )

            remaining = max(
                0.0,
                self.delay_seconds - elapsed
            )

            # -------------------------------------------------
            # Start confirmation completed.
            # IMPORTANT: save THIS time as the violation start.
            # This excludes the initial 60-second delay.
            # -------------------------------------------------
            if elapsed >= self.delay_seconds:
                self.violation_confirmed = True
                self.violation_confirmed_time = current_time
                self.violation_start_time = current_time
                self.violation_end_time = None
                self.normal_start_time = None
                self.ending_pending = False
                self.violation_duration = 0.0

                return self._result(
                    "confirmed",
                    elapsed=0.0,
                    remaining=0.0,
                    duration=0.0
                )

            return self._result(
                "pending_start",
                elapsed=elapsed,
                remaining=remaining
            )

        # =====================================================
        # 3. CONFIRMED VIOLATION
        # =====================================================

        # -----------------------------------------------------
        # Condition is still a violation.
        # Cancel any pending end confirmation and continue.
        # -----------------------------------------------------
        if is_violation:

            self.normal_start_time = None
            self.ending_pending = False
            self.violation_end_time = None

            elapsed = current_time - self.violation_start_time
            self.violation_duration = max(0.0, elapsed)

            return self._result(
                "confirmed",
                elapsed=elapsed,
                remaining=0.0,
                duration=self.violation_duration
            )

        # =====================================================
        # 4. CONDITION BECAME NORMAL
        # =====================================================

        if not self.ending_pending:
            # First NORMAL frame. Start the end-confirmation timer.
            # We do NOT save this as the final end time yet.
            self.normal_start_time = current_time
            self.ending_pending = True

            # Keep duration frozen while end confirmation runs.
            elapsed = max(
                0.0,
                current_time - self.violation_start_time
            )
            self.violation_duration = elapsed

            return self._result(
                "pending_end",
                elapsed=elapsed,
                remaining=self.delay_seconds,
                duration=elapsed
            )

        # =====================================================
        # 5. NORMAL END-CONFIRMATION PERIOD
        # =====================================================

        normal_elapsed = (
            current_time - self.normal_start_time
        )

        remaining = max(
            0.0,
            self.delay_seconds - normal_elapsed
        )

        # If violation returns before 60 seconds, cancel the
        # ending process and continue the original violation.
        # The original saved start time is preserved.
        if is_violation:
            self.normal_start_time = None
            self.ending_pending = False

            elapsed = current_time - self.violation_start_time
            self.violation_duration = max(0.0, elapsed)

            return self._result(
                "confirmed",
                elapsed=elapsed,
                remaining=0.0,
                duration=self.violation_duration
            )

        # -----------------------------------------------------
        # End confirmation completed.
        # The END timestamp is the time at which the 60-second
        # NORMAL confirmation completes.
        # -----------------------------------------------------
        if normal_elapsed >= self.delay_seconds:
            self.violation_end_time = current_time
            self.violation_duration = max(
                0.0,
                self.violation_end_time - self.violation_start_time
            )

            # Keep the completed values available for this frame.
            result = self._result(
                "ended",
                elapsed=self.violation_duration,
                remaining=0.0,
                duration=self.violation_duration,
                end_confirmed=True
            )

            # Reset the active state after creating the result.
            # The returned result still contains the final timestamps.
            self._reset()
            return result

        # End confirmation is still running.
        elapsed = max(
            0.0,
            current_time - self.violation_start_time
        )
        self.violation_duration = elapsed

        return self._result(
            "pending_end",
            elapsed=elapsed,
            remaining=remaining,
            duration=elapsed
        )


# =========================================================
# CREATE VIOLATION TRACKER
# =========================================================

violation_tracker = ViolationTracker(
    VIOLATION_DELAY_SECONDS
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


        violation_state = violation_info["state"]

        violation_confirmed = (
            violation_info["confirmed"]
        )

        violation_elapsed = (
            violation_info["elapsed"]
        )

        violation_remaining = (
            violation_info["remaining"]
        )

        violation_duration = (
            violation_info["duration"]
        )

        violation_start_time = (
            violation_info["start_time"]
        )

        violation_confirmed_time = (
            violation_info["confirmed_time"]
        )

        violation_end_time = (
            violation_info["end_time"]
        )


        # =================================================
        # DRAW LEFT ROI
        # =================================================

        x, y, w, h = LEFT_ROI

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            4
        )


        # =================================================
        # DRAW RIGHT ROI
        # =================================================

        x, y, w, h = RIGHT_ROI

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
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
            (0, 255, 0),
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
            (0, 255, 0),
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

        if violation_state == "pending_start":

            waiting_text = (
                f"VIOLATION PENDING: "
                f"{violation_remaining:.1f}s"
            )

            cv2.putText(
                display,
                waiting_text,
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                "Start timer is not counted in violation duration",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 165, 255),
                1,
                cv2.LINE_AA
            )


        elif violation_state == "confirmed":

            violation_text = (
                "VIOLATION CONFIRMED"
            )

            cv2.putText(
                display,
                violation_text,
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Duration: {violation_duration:.1f}s",
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
                    f"Start: {time.strftime('%H:%M:%S', time.localtime(violation_start_time))}",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )


        elif violation_state == "pending_end":

            ending_text = (
                f"VIOLATION ENDING - CONFIRMING NORMAL: "
                f"{violation_remaining:.1f}s"
            )

            cv2.putText(
                display,
                ending_text,
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Duration: {violation_duration:.1f}s",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )


        elif violation_state == "ended":

            cv2.putText(
                display,
                "VIOLATION ENDED",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.80,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Duration: {violation_duration:.1f}s",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            if violation_start_time is not None:
                cv2.putText(
                    display,
                    f"Start: {time.strftime('%H:%M:%S', time.localtime(violation_start_time))}",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            if violation_end_time is not None:
                cv2.putText(
                    display,
                    f"End: {time.strftime('%H:%M:%S', time.localtime(violation_end_time))}",
                    (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
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