import cv2
import os
import time
import re


# =========================================================
# CONFIGURATION
# =========================================================

VIDEO_PATH = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"

OUTPUT_DIR = "dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "materialimages")

os.makedirs(IMAGES_DIR, exist_ok=True)


# =========================================================
# DATASET SETTINGS
# =========================================================

MIN_BOXES = 1
MAX_BOXES = 2

# Save one image every 3 seconds
SAVE_INTERVAL_SECONDS = 3

# Camera recovery settings
MAX_FAILED_READS = 10
RECONNECT_DELAY_SECONDS = 1.0
STABLE_FRAMES_REQUIRED = 15
RECONNECT_MAX_ATTEMPTS = 0  # 0 = keep trying forever


# =========================================================
# DISPLAY SETTINGS
# =========================================================

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# =========================================================
# FIND NEXT IMAGE NUMBER
# =========================================================

def get_next_save_count():
    """
    Find the highest existing image number.

    Example:

        box_000001.jpg
        box_000002.jpg
        box_000844.jpg

    Next number will be:

        845
    """

    highest_number = 0

    for filename in os.listdir(IMAGES_DIR):

        match = re.match(
            r"box_(\d+)\.jpg$",
            filename
        )

        if match:

            number = int(match.group(1))

            if number > highest_number:
                highest_number = number

    return highest_number + 1


# =========================================================
# INITIALIZE COUNTER
# =========================================================

save_count = get_next_save_count()

print()
print("==========================================")
print("       BOX IMAGE DATASET COLLECTION")
print("==========================================")
print(f"Starting image number: {save_count}")
print(f"Save interval: {SAVE_INTERVAL_SECONDS} seconds")
print(f"Maximum boxes: {MAX_BOXES}")
print("Mode: IMAGE ONLY")
print("==========================================")


# =========================================================
# OPEN RTSP STREAM
# =========================================================

def open_camera():
    """Open/reopen the RTSP stream."""

    print("Connecting to camera...")

    camera = cv2.VideoCapture(VIDEO_PATH)

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if camera.isOpened():

        print("Camera connected.")

        return camera

    camera.release()

    print("Camera connection failed.")

    return None


cap = open_camera()

if cap is None:
    raise Exception("Could not open RTSP stream.")


# =========================================================
# VARIABLES
# =========================================================

paused = False

# List of bounding boxes
#
# Example:
#
# [
#     (x1, y1, w1, h1),
#     (x2, y2, w2, h2)
# ]
#
# These boxes are ONLY used for selecting/drawing.
# They are NOT saved as labels.

boxes = []

frame = None

last_save_time = 0

session_saved = 0


# =========================================================
# RTSP / DECODER RECOVERY STATE
# =========================================================

failed_reads = 0
reconnect_attempts = 0
stable_frame_count = 0
stream_recovering = False


# =========================================================
# SAVE IMAGE ONLY
# =========================================================

def save_image(
    image,
    image_name
):
    """
    Save image only.

    No YOLO annotation/label file is created.
    """

    image_path = os.path.join(
        IMAGES_DIR,
        image_name + ".jpg"
    )

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if os.path.exists(image_path):

        print(
            f"WARNING: {image_path} already exists."
        )

        return False


    # -----------------------------------------------------
    # Save image
    # -----------------------------------------------------

    success = cv2.imwrite(
        image_path,
        image
    )

    if not success:

        print(
            f"Failed to save image: {image_path}"
        )

        return False


    return True


# =========================================================
# CREATE WINDOW
# =========================================================

cv2.namedWindow(
    "Box Dataset Annotator",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Box Dataset Annotator",
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT
)


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    # =====================================================
    # READ FRAME
    # =====================================================

    if not paused:

        ret, new_frame = cap.read()

        if not ret or new_frame is None:

            failed_reads += 1

            print(
                f"Failed to read frame "
                f"({failed_reads}/{MAX_FAILED_READS})"
            )

            # Do not keep using a stale frame
            # after RTSP/decoder failure.

            frame = None

            if failed_reads >= MAX_FAILED_READS:

                print()
                print("==========================================")
                print("Camera stream lost. Reconnecting...")
                print("==========================================")

                cap.release()
                cap = None

                stable_frame_count = 0
                stream_recovering = True

                while cap is None:

                    reconnect_attempts += 1

                    if (
                        RECONNECT_MAX_ATTEMPTS > 0
                        and reconnect_attempts > RECONNECT_MAX_ATTEMPTS
                    ):

                        print(
                            "Maximum reconnect attempts reached."
                        )

                        break

                    time.sleep(
                        RECONNECT_DELAY_SECONDS
                    )

                    cap = open_camera()

                    if cap is None:

                        print(
                            "Reconnect failed. Retrying..."
                        )

                if cap is None:
                    break

                failed_reads = 0
                reconnect_attempts = 0

                print(
                    "Camera reconnected. "
                    "Waiting for stable frames..."
                )

            else:

                time.sleep(0.1)

            continue


        # =================================================
        # VALID FRAME RECEIVED
        # =================================================

        frame = new_frame

        failed_reads = 0


        # =================================================
        # AFTER RECONNECT
        # WAIT FOR STABLE FRAMES
        # =================================================

        if stream_recovering:

            stable_frame_count += 1

            if stable_frame_count < STABLE_FRAMES_REQUIRED:

                recovery_scale = min(
                    DISPLAY_WIDTH / frame.shape[1],
                    DISPLAY_HEIGHT / frame.shape[0]
                )

                recovery_display = cv2.resize(
                    frame,
                    (
                        int(
                            frame.shape[1]
                            * recovery_scale
                        ),
                        int(
                            frame.shape[0]
                            * recovery_scale
                        )
                    ),
                    interpolation=cv2.INTER_AREA
                )

                cv2.putText(
                    recovery_display,
                    f"CAMERA RECOVERING "
                    f"{stable_frame_count}/"
                    f"{STABLE_FRAMES_REQUIRED}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )

                cv2.imshow(
                    "Box Dataset Annotator",
                    recovery_display
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == ord("Q"):
                    break

                continue


            stream_recovering = False
            stable_frame_count = 0

            # Prevent immediate save after reconnect
            last_save_time = time.time()

            print(
                "CAMERA STABLE - "
                "Dataset collection resumed."
            )


    # =====================================================
    # FRAME SAFETY CHECK
    # =====================================================

    if frame is None:
        continue


    # =====================================================
    # ORIGINAL FRAME SIZE
    # =====================================================

    original_height, original_width = frame.shape[:2]


    # =====================================================
    # CALCULATE DISPLAY SCALE
    # =====================================================

    scale = min(
        DISPLAY_WIDTH / original_width,
        DISPLAY_HEIGHT / original_height
    )

    display_width = int(
        original_width * scale
    )

    display_height = int(
        original_height * scale
    )


    # =====================================================
    # RESIZE ONLY FOR DISPLAY
    # =====================================================

    display = cv2.resize(
        frame,
        (
            display_width,
            display_height
        ),
        interpolation=cv2.INTER_AREA
    )


    # =====================================================
    # DRAW ALL SELECTED BOXES
    # =====================================================

    for index, box in enumerate(boxes):

        x, y, w, h = box


        # -------------------------------------------------
        # Convert original coordinates
        # to display coordinates
        # -------------------------------------------------

        display_x = int(
            x * scale
        )

        display_y = int(
            y * scale
        )

        display_w = int(
            w * scale
        )

        display_h = int(
            h * scale
        )


        # -------------------------------------------------
        # Draw rectangle
        # -------------------------------------------------

        cv2.rectangle(
            display,
            (
                display_x,
                display_y
            ),
            (
                display_x + display_w,
                display_y + display_h
            ),
            (0, 255, 0),
            3
        )


        # -------------------------------------------------
        # Box number
        # -------------------------------------------------

        cv2.putText(
            display,
            f"BOX {index + 1}",
            (
                display_x,
                max(
                    30,
                    display_y - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # =====================================================
    # INFORMATION
    # =====================================================

    cv2.putText(
        display,
        f"Boxes selected: {len(boxes)}/2",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"Images saved: {session_saved}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"Resolution: "
        f"{original_width}x{original_height}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # =====================================================
    # CONTROLS
    # =====================================================

    cv2.putText(
        display,
        "S: Add Box | C: Clear | "
        "SPACE: Pause | Q: Quit",
        (
            20,
            display_height - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # =====================================================
    # PAUSED MESSAGE
    # =====================================================

    if paused:

        cv2.putText(
            display,
            "PAUSED",
            (
                20,
                145
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # =====================================================
    # SHOW FRAME
    # =====================================================

    cv2.imshow(
        "Box Dataset Annotator",
        display
    )


    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    # =====================================================
    # Q = QUIT
    # =====================================================

    if key == ord("q") or key == ord("Q"):

        break


    # =====================================================
    # SPACE = PAUSE / RESUME
    # =====================================================

    elif key == ord(" "):

        paused = not paused

        if paused:

            print("Stream paused.")

        else:

            print("Stream resumed.")


    # =====================================================
    # S = ADD BOX
    # =====================================================

    elif key == ord("s") or key == ord("S"):


        # -------------------------------------------------
        # Check maximum number of boxes
        # -------------------------------------------------

        if len(boxes) >= MAX_BOXES:

            print()

            print(
                "Maximum of 2 boxes already selected."
            )

            print(
                "Press C to clear and select again."
            )

            continue


        # -------------------------------------------------
        # Pause stream while selecting
        # -------------------------------------------------

        paused = True


        print()

        print(
            f"Select BOX {len(boxes) + 1}"
        )

        print(
            "Drag around the complete box."
        )

        print(
            "Press ENTER or SPACE to confirm."
        )


        # -------------------------------------------------
        # Select ROI
        # -------------------------------------------------

        roi = cv2.selectROI(
            "Box Dataset Annotator",
            display,
            fromCenter=False,
            showCrosshair=True
        )


        rx, ry, rw, rh = map(
            int,
            roi
        )


        # -------------------------------------------------
        # Check ROI
        # -------------------------------------------------

        if rw > 0 and rh > 0:


            # ---------------------------------------------
            # Convert display coordinates
            # back to original coordinates
            # ---------------------------------------------

            x = int(
                rx / scale
            )

            y = int(
                ry / scale
            )

            w = int(
                rw / scale
            )

            h = int(
                rh / scale
            )


            # ---------------------------------------------
            # Add box
            # ---------------------------------------------

            boxes.append(
                (
                    x,
                    y,
                    w,
                    h
                )
            )


            print()

            print(
                "=========================================="
            )

            print(
                f"BOX {len(boxes)} SELECTED"
            )

            print(
                "=========================================="
            )

            print(
                f"x      : {x}"
            )

            print(
                f"y      : {y}"
            )

            print(
                f"width  : {w}"
            )

            print(
                f"height : {h}"
            )

            print(
                "=========================================="
            )


            # ---------------------------------------------
            # If 2 boxes selected
            # ---------------------------------------------

            if len(boxes) == MAX_BOXES:

                print(
                    "Both boxes selected."
                )

                print(
                    "Dataset collection will continue."
                )


        else:

            print(
                "No box selected."
            )


    # =====================================================
    # C = CLEAR ALL BOXES
    # =====================================================

    elif key == ord("c") or key == ord("C"):

        boxes.clear()

        print()

        print(
            "All bounding boxes cleared."
        )

        print(
            "You can now select 1 or 2 boxes again."
        )


    # =====================================================
    # SAVE EVERY 3 SECONDS
    # =====================================================

    if (
        len(boxes) >= MIN_BOXES
        and not paused
    ):

        current_time = time.time()


        if (
            current_time - last_save_time
            >= SAVE_INTERVAL_SECONDS
        ):


            image_name = (
                f"box_{save_count:06d}"
            )


            # -------------------------------------------------
            # Save ONLY image
            # -------------------------------------------------

            saved = save_image(
                frame,
                image_name
            )


            if saved:

                print(
                    f"Saved: {image_name}.jpg "
                    f"with {len(boxes)} selected box(es)"
                )

                save_count += 1
                session_saved += 1

                last_save_time = current_time


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()


print()

print(
    "=========================================="
)

print(
    "Dataset collection stopped."
)

print(
    f"Images saved this session: {session_saved}"
)

print(
    f"Next image number: {save_count}"
)

print(
    f"Dataset folder: {OUTPUT_DIR}"
)

print(
    "=========================================="
)