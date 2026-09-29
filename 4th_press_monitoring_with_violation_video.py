import cv2
import torch
import threading
import time
import os

from PIL import Image
from torchvision import models, transforms
from database import insert_violation, update_violation, update_violation_video

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


# # =========================================================
# # VIOLATION VIDEO CONFIGURATION
# # =========================================================



# Folder where final violation videos are stored.
# VIOLATION_VIDEO_DIR = "violation_videos"
# os.makedirs(VIOLATION_VIDEO_DIR, exist_ok=True)

# =========================================================
# VIOLATION VIDEO CONFIGURATION
# =========================================================

VIOLATION_VIDEO_DIR = "violation_videos"

# Save video at 2x playback speed.
VIDEO_SPEED = 2.0

# Reduce resolution to make files smaller.
# Original camera: 3840x2160
VIDEO_OUTPUT_WIDTH = 1920
VIDEO_OUTPUT_HEIGHT = 1080

# Lower FPS also reduces file size.
VIDEO_OUTPUT_FPS = 5.0

os.makedirs(VIOLATION_VIDEO_DIR, exist_ok=True)

# =========================================================
# LEFT ROI
# Format:
# x, y, width, height
# =========================================================

LEFT_ROI = (1000, 500, 711, 837)


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


# # =========================================================
# # VIOLATION VIDEO RECORDER
# # =========================================================

# class ViolationVideoRecorder:
#     """
#     Records the complete violation temporarily.

#     Final rule:
#         duration <= 5 minutes
#             -> save complete violation

#         duration > 5 minutes
#             -> save first 2 minutes + last 2 minutes

#     The temporary recording is removed after the final video is created.
#     """

#     def __init__(self, output_dir, fps=15.0):

#         self.output_dir = output_dir
#         self.fps = float(fps) if fps and fps > 0 else 15.0

#         self.lock = threading.Lock()

#         self.active = False
#         self.violation_id = None
#         self.temp_path = None
#         self.writer = None
#         self.frame_size = None

#     def start(self, violation_id):

#         with self.lock:

#             if self.active:
#                 return

#             self.violation_id = violation_id

#             self.temp_path = os.path.join(
#                 self.output_dir,
#                 f"temp_violation_{violation_id}.mp4"
#             )

#             self.writer = None
#             self.frame_size = None
#             self.active = True

#             print(
#                 f"[VIDEO] Recording started for violation ID "
#                 f"{violation_id}"
#             )

#     def write(self, frame):

#         with self.lock:

#             if not self.active or frame is None:
#                 return

#             if self.writer is None:

#                 height, width = frame.shape[:2]

#                 self.frame_size = (width, height)

#                 fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#                 self.writer = cv2.VideoWriter(
#                     self.temp_path,
#                     fourcc,
#                     self.fps,
#                     self.frame_size
#                 )

#                 if not self.writer.isOpened():

#                     print(
#                         f"[VIDEO] ERROR: Could not create "
#                         f"{self.temp_path}"
#                     )

#                     self.writer = None
#                     self.active = False
#                     return

#             self.writer.write(frame)

#     def cancel(self):

#         with self.lock:

#             self._release_writer()

#             temp_path = self.temp_path

#             self.active = False
#             self.violation_id = None
#             self.temp_path = None
#             self.frame_size = None

#             if temp_path and os.path.exists(temp_path):

#                 try:
#                     os.remove(temp_path)
#                     print(f"[VIDEO] Temporary video deleted: {temp_path}")
#                 except Exception as exc:
#                     print(
#                         f"[VIDEO] Could not delete temporary video: {exc}"
#                     )

#     def finalize(self, violation_id, duration_seconds):
#         """Create the final video and return its path."""

#         with self.lock:

#             self._release_writer()

#             temp_path = self.temp_path

#             self.active = False
#             self.violation_id = None
#             self.temp_path = None
#             self.frame_size = None

#         if not temp_path or not os.path.exists(temp_path):

#             print("[VIDEO] ERROR: Temporary violation video not found.")
#             return None

#         try:

#             duration_seconds = max(
#                 0.0,
#                 float(duration_seconds)
#             )

#             final_path = os.path.join(
#                 self.output_dir,
#                 f"violation_{violation_id}_"
#                 f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#             )

#             source = cv2.VideoCapture(temp_path)

#             if not source.isOpened():
#                 print("[VIDEO] ERROR: Could not open temporary video.")
#                 return None

#             source_fps = source.get(cv2.CAP_PROP_FPS)

#             if not source_fps or source_fps <= 0:
#                 source_fps = self.fps

#             width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))

#             if width <= 0 or height <= 0:
#                 source.release()
#                 print("[VIDEO] ERROR: Invalid temporary video size.")
#                 return None

#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#             final_writer = cv2.VideoWriter(
#                 final_path,
#                 fourcc,
#                 source_fps,
#                 (width, height)
#             )

#             if not final_writer.isOpened():
#                 source.release()
#                 print("[VIDEO] ERROR: Could not create final video.")
#                 return None

#             # -------------------------------------------------
#             # <= 5 minutes: keep the complete violation.
#             # > 5 minutes: keep first 2 minutes + last 2 minutes.
#             # -------------------------------------------------

#             if duration_seconds <= MAX_FULL_VIDEO_SECONDS:

#                 ranges = [
#                     (0.0, duration_seconds)
#                 ]

#                 print(
#                     f"[VIDEO] Duration {duration_seconds:.1f}s <= "
#                     f"5 minutes: saving full violation."
#                 )

#             else:

#                 ranges = [
#                     (0.0, VIDEO_SEGMENT_SECONDS),
#                     (
#                         duration_seconds - VIDEO_SEGMENT_SECONDS,
#                         duration_seconds
#                     )
#                 ]

#                 print(
#                     f"[VIDEO] Duration {duration_seconds:.1f}s > "
#                     f"5 minutes: saving first 2 min + last 2 min."
#                 )

#             # Convert time ranges into frame ranges.
#             frame_ranges = [
#                 (
#                     max(0, int(start * source_fps)),
#                     max(0, int(end * source_fps))
#                 )
#                 for start, end in ranges
#             ]

#             frame_index = 0
#             range_index = 0
#             current_range = frame_ranges[range_index]

#             while True:

#                 ret, frame = source.read()

#                 if not ret:
#                     break

#                 range_start, range_end = current_range

#                 if frame_index >= range_start and frame_index < range_end:
#                     final_writer.write(frame)

#                 if frame_index >= range_end:

#                     range_index += 1

#                     if range_index >= len(frame_ranges):
#                         break

#                     current_range = frame_ranges[range_index]

#                 frame_index += 1

#             source.release()
#             final_writer.release()

#             # The temporary file is no longer needed.
#             try:
#                 os.remove(temp_path)
#             except Exception as exc:
#                 print(
#                     f"[VIDEO] Warning: could not remove temporary "
#                     f"video: {exc}"
#                 )

#             if not os.path.exists(final_path):
#                 print("[VIDEO] ERROR: Final video was not created.")
#                 return None

#             print(
#                 f"[VIDEO] Final video saved: {final_path}"
#             )

#             return os.path.abspath(final_path)

#         except Exception as exc:

#             print(
#                 f"[VIDEO] ERROR while finalizing video: {exc}"
#             )

#             try:
#                 if os.path.exists(temp_path):
#                     os.remove(temp_path)
#             except Exception:
#                 pass

#             return None

#     def _release_writer(self):

#         if self.writer is not None:

#             self.writer.release()
#             self.writer = None

# =========================================================
# VIOLATION VIDEO RECORDER
# =========================================================

class ViolationVideoRecorder:
    """
    Records the COMPLETE violation.

    Recording starts at the FIRST violation detection
    and continues until the violation is completed.

    Final video:
        - Complete violation is preserved.
        - Playback speed = 2x.
        - Resolution = 1920x1080.
        - FPS = 10.
        - No 5-minute limit.
        - No first/last segment trimming.

    The temporary recording is deleted after finalization.
    """

    def __init__(self,output_dir,fps=15.0,playback_speed=2.0):

        self.output_dir = output_dir
        self.playback_speed = playback_speed
        self.camera_fps = (
            float(fps)
            if fps and fps > 0
            else 15.0
        )

        self.lock = threading.Lock()

        self.active = False

        self.violation_id = None

        self.temp_path = None

        self.writer = None

        self.frame_size = None

        # -------------------------------------------------
        # Recording control
        # -------------------------------------------------

        self.frame_skip_counter = 0

        self.write_every_n_frames = 1

    # =====================================================
    # START
    # =====================================================

    def start(self, violation_id):

        with self.lock:

            if self.active:
                return

            self.violation_id = violation_id

            self.temp_path = os.path.join(
                self.output_dir,
                f"temp_violation_{violation_id}.mp4"
            )

            self.writer = None

            self.frame_size = None

            self.frame_skip_counter = 0

            # -------------------------------------------------
            # We record at approximately half the camera FPS.
            #
            # This reduces the number of stored frames.
            # The final video will be played at 2x.
            # -------------------------------------------------

            target_capture_fps = (
                self.camera_fps / VIDEO_SPEED
            )

            if target_capture_fps < 1:

                target_capture_fps = 1.0

            self.write_every_n_frames = max(1,int(round(VIDEO_SPEED)))

            self.active = True

            print(
                f"\n[VIDEO] ============================="
            )

            print(
                f"[VIDEO] Recording started"
            )

            print(
                f"[VIDEO] Violation ID: {violation_id}"
            )

            print(
                f"[VIDEO] Camera FPS: "
                f"{self.camera_fps:.2f}"
            )

            print(
                f"[VIDEO] Target playback speed: "
                f"{VIDEO_SPEED}x"
            )

            print(
                f"[VIDEO] Output resolution: "
                f"{VIDEO_OUTPUT_WIDTH}x"
                f"{VIDEO_OUTPUT_HEIGHT}"
            )

            print(
                f"[VIDEO] ============================="
            )

    # =====================================================
    # WRITE FRAME
    # =====================================================

    def write(self, frame):

        with self.lock:

            if not self.active:

                return

            if frame is None:

                return

            # -------------------------------------------------
            # Skip frames to reduce storage.
            # -------------------------------------------------

            self.frame_skip_counter += 1

            if (
                self.frame_skip_counter
                < self.write_every_n_frames
            ):

                return

            self.frame_skip_counter = 0

            # -------------------------------------------------
            # Resize from 3840x2160 -> 1920x1080
            # -------------------------------------------------

            resized = cv2.resize(
                frame,
                (
                    VIDEO_OUTPUT_WIDTH,
                    VIDEO_OUTPUT_HEIGHT
                ),
                interpolation=cv2.INTER_AREA
            )

            # -------------------------------------------------
            # Create writer
            # -------------------------------------------------

            if self.writer is None:

                self.frame_size = (
                    VIDEO_OUTPUT_WIDTH,
                    VIDEO_OUTPUT_HEIGHT
                )

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                output_fps = self.camera_fps

                # Safety
                if output_fps < 1:

                    output_fps = 1.0

                self.writer = cv2.VideoWriter(

                    self.temp_path,

                    fourcc,

                    output_fps,

                    self.frame_size
                )

                if not self.writer.isOpened():

                    print(
                        "[VIDEO] ERROR: "
                        "Could not create temporary video."
                    )

                    self.writer = None

                    self.active = False

                    return

                print(
                    f"[VIDEO] Temporary recording:"
                    f" {self.temp_path}"
                )

                print(
                    f"[VIDEO] Recording FPS: "
                    f"{output_fps:.2f}"
                )

            # -------------------------------------------------
            # Write frame
            # -------------------------------------------------

            self.writer.write(
                resized
            )

    # =====================================================
    # CANCEL
    # =====================================================

    def cancel(self):

        with self.lock:

            self._release_writer()

            temp_path = self.temp_path

            self.active = False

            self.violation_id = None

            self.temp_path = None

            self.frame_size = None

            self.frame_skip_counter = 0

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                try:

                    os.remove(
                        temp_path
                    )

                    print(
                        f"[VIDEO] Temporary video deleted:"
                        f" {temp_path}"
                    )

                except Exception as exc:

                    print(
                        f"[VIDEO] Could not delete "
                        f"temporary video: {exc}"
                    )

    # =====================================================
    # FINALIZE
    # =====================================================

    def finalize(
        self,
        violation_id,
        duration_seconds
    ):
        """
        Finalize COMPLETE violation video.

        No 5-minute limitation.

        The temporary video already contains the complete
        violation, so we simply rename/copy it to the final
        violation filename.
        """

        with self.lock:

            self._release_writer()

            temp_path = self.temp_path

            self.active = False

            self.violation_id = None

            self.temp_path = None

            self.frame_size = None

            self.frame_skip_counter = 0

        # -------------------------------------------------
        # Check temporary file
        # -------------------------------------------------

        if (
            not temp_path
            or not os.path.exists(temp_path)
        ):

            print(
                "[VIDEO] ERROR: "
                "Temporary violation video not found."
            )

            return None

        try:

            final_path = os.path.join(

                self.output_dir,

                f"violation_{violation_id}_"
                f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            )

            # -------------------------------------------------
            # Rename temporary file.
            #
            # No re-encoding is required because the temporary
            # recording has already been created using the
            # desired FPS/resolution.
            # -------------------------------------------------

            os.replace(
                temp_path,
                final_path
            )

            print(
                "\n[VIDEO] ============================="
            )

            print(
                "[VIDEO] COMPLETE VIOLATION VIDEO SAVED"
            )

            print(
                f"[VIDEO] Violation ID: "
                f"{violation_id}"
            )

            print(
                f"[VIDEO] Real violation duration: "
                f"{duration_seconds:.1f} seconds"
            )

            print(
                f"[VIDEO] Playback speed: "
                f"{VIDEO_SPEED}x"
            )

            print(
                f"[VIDEO] Approx. playback duration: "
                f"{duration_seconds / VIDEO_SPEED:.1f} seconds"
            )

            print(
                f"[VIDEO] File: "
                f"{final_path}"
            )

            print(
                "[VIDEO] ============================="
            )

            return os.path.abspath(
                final_path
            )

        except Exception as exc:

            print(
                f"[VIDEO] ERROR while finalizing: "
                f"{exc}"
            )

            return None

    # =====================================================
    # RELEASE WRITER
    # =====================================================

    def _release_writer(self):

        if self.writer is not None:

            self.writer.release()

            self.writer = None

# =========================================================
# RTSP READER
# =========================================================

class RTSPReader:

    def __init__(self, url, video_recorder=None):

        self.url = url
        self.video_recorder = video_recorder

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

            camera_fps = self.cap.get(
                cv2.CAP_PROP_FPS
            )

            if (
                self.video_recorder is not None
                and camera_fps
                and camera_fps > 0
            ):
                self.video_recorder.fps = camera_fps

            print(
                f"RTSP connected: "
                f"{width}x{height} | FPS: {camera_fps:.2f}"
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

                # Record every camera frame while a violation is active.
                # This happens in the RTSP reader thread so recording uses
                # the camera stream rate instead of the slower inference loop.
                if self.video_recorder is not None:
                    self.video_recorder.write(frame)

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
            "duration": self.violation_duration,
            "database_id": self.database_violation_id,
            "cancelled": False,
            "completed": False
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

                result = self._base_result()
                result["cancelled"] = True

                self._reset_all()

                return result

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
            result["database_id"] = self.database_violation_id
            result["completed"] = True

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
# CREATE VIDEO RECORDER
# =========================================================

video_recorder = ViolationVideoRecorder(
    VIOLATION_VIDEO_DIR,
    fps=15.0,
    playback_speed=2.0
)


# =========================================================
# START RTSP
# =========================================================

reader = RTSPReader(
    RTSP_URL,
    video_recorder=video_recorder
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


        violation_database_id = violation_info.get(
            "database_id"
        )


        # =================================================
        # VIDEO RECORDING CONTROL
        # =================================================

        # Start recording from the FIRST violation detection.
        # This is before the 60-second confirmation finishes.
        if (
            violation_database_id is not None
            and violation_start_time is not None
            and not video_recorder.active
            and not violation_info.get("completed", False)
            and not violation_info.get("cancelled", False)
        ):

            video_recorder.start(
                violation_database_id
            )


        # A candidate violation can be cancelled before it is confirmed.
        # In that case its temporary video must not be kept.
        if violation_info.get("cancelled", False):

            video_recorder.cancel()


        # Normal has been confirmed. The tracker has already calculated
        # the real violation duration using FIRST violation detection ->
        # FIRST normal detection. Now create the final video.
        if violation_info.get("completed", False):

            completed_id = violation_info.get(
                "database_id"
            )

            completed_duration = violation_info.get(
                "duration",
                0.0
            )

            if completed_id is not None:

                final_video_path = video_recorder.finalize(
                    violation_id=completed_id,
                    duration_seconds=completed_duration
                )

                if final_video_path:

                    video_db_result = update_violation_video(
                        violation_id=completed_id,
                        video_path=final_video_path
                    )

                    print(
                        f"[VIDEO] Database video update result: "
                        f"{video_db_result}"
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

    # Do not keep an unfinished temporary recording when the program quits.
    if video_recorder.active:
        video_recorder.cancel()

    cv2.destroyAllWindows()

    print(
        "Pipeline stopped."
    )