# import cv2
# import os
# import time

# # -----------------------------
# # Configuration
# # -----------------------------
# VIDEO_PATH = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"

# OUTPUT_DIR = "dataset"
# IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
# LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# os.makedirs(IMAGES_DIR, exist_ok=True)
# os.makedirs(LABELS_DIR, exist_ok=True)

# CLASS_ID = 0          # Forklift
# DELAY = 30             # milliseconds

# # -----------------------------
# # Display size
# # -----------------------------
# DISPLAY_WIDTH = 1280
# DISPLAY_HEIGHT = 720

# # -----------------------------
# # Open RTSP Stream
# # -----------------------------
# cap = cv2.VideoCapture(VIDEO_PATH)

# if not cap.isOpened():
#     raise Exception("Could not open RTSP stream.")

# # Try to reduce buffering
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# paused = False
# current_box = None
# frame = None

# save_count = 844


# # -----------------------------
# # Save YOLO Annotation
# # -----------------------------
# def save_yolo_annotation(image, box, image_name):

#     """
#     Save image and YOLO annotation.

#     box = (x, y, w, h)
#     """

#     h_img, w_img = image.shape[:2]

#     x, y, w, h = box

#     # Convert to YOLO format
#     x_center = (x + w / 2) / w_img
#     y_center = (y + h / 2) / h_img

#     w_norm = w / w_img
#     h_norm = h / h_img

#     image_path = os.path.join(
#         IMAGES_DIR,
#         image_name + ".jpg"
#     )

#     label_path = os.path.join(
#         LABELS_DIR,
#         image_name + ".txt"
#     )

#     # Save original full-resolution image
#     cv2.imwrite(
#         image_path,
#         image
#     )

#     # Save YOLO label
#     with open(label_path, "w") as f:

#         f.write(
#             f"{CLASS_ID} "
#             f"{x_center:.6f} "
#             f"{y_center:.6f} "
#             f"{w_norm:.6f} "
#             f"{h_norm:.6f}"
#         )


# # -----------------------------
# # Create Window
# # -----------------------------
# cv2.namedWindow(
#     "Annotator",
#     cv2.WINDOW_NORMAL
# )

# cv2.resizeWindow(
#     "Annotator",
#     DISPLAY_WIDTH,
#     DISPLAY_HEIGHT
# )


# # -----------------------------
# # Main Loop
# # -----------------------------
# while True:

#     if not paused:

#         ret, frame = cap.read()

#         if not ret:

#             print("Failed to read frame.")
#             time.sleep(0.1)
#             continue

#     if frame is None:
#         continue

#     # -------------------------------------------------
#     # IMPORTANT:
#     # Resize ONLY the display image.
#     # Keep original frame for saving/annotation.
#     # -------------------------------------------------
#     original_height, original_width = frame.shape[:2]

#     scale = min(
#         DISPLAY_WIDTH / original_width,
#         DISPLAY_HEIGHT / original_height
#     )

#     display_width = int(original_width * scale)
#     display_height = int(original_height * scale)

#     display = cv2.resize(
#         frame,
#         (display_width, display_height),
#         interpolation=cv2.INTER_AREA
#     )

#     # -------------------------------------------------
#     # Draw bounding box
#     # -------------------------------------------------
#     if current_box is not None:

#         x, y, w, h = current_box

#         # Convert original coordinates
#         # to display coordinates
#         dx = int(x * scale)
#         dy = int(y * scale)

#         dw = int(w * scale)
#         dh = int(h * scale)

#         cv2.rectangle(
#             display,
#             (dx, dy),
#             (dx + dw, dy + dh),
#             (0, 255, 0),
#             2
#         )

#         # -------------------------------------------------
#         # Save original frame + original coordinates
#         # -------------------------------------------------
#         image_name = f"forklift_{save_count:06d}"

#         save_yolo_annotation(
#             frame,
#             current_box,
#             image_name
#         )

#         save_count += 1

#     # -------------------------------------------------
#     # Information text
#     # -------------------------------------------------
#     cv2.putText(
#         display,
#         f"Saved: {save_count}",
#         (20, 35),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 255),
#         2
#     )

#     cv2.putText(
#         display,
#         f"Resolution: {original_width}x{original_height}",
#         (20, 70),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         (255, 255, 255),
#         2
#     )

#     if paused:

#         cv2.putText(
#             display,
#             "PAUSED",
#             (20, 110),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 0, 255),
#             2
#         )

#     # -------------------------------------------------
#     # Show fitted frame
#     # -------------------------------------------------
#     cv2.imshow(
#         "Annotator",
#         display
#     )

#     key = cv2.waitKey(DELAY) & 0xFF

#     # Q = Quit
#     if key == ord('q'):
#         break

#     # SPACE = Pause / Resume
#     elif key == ord(' '):

#         paused = not paused

#     # S = Select bounding box
#     elif key == ord('s'):

#         paused = True

#         # Select ROI on ORIGINAL resolution
#         roi = cv2.selectROI(
#             "Annotator",
#             display,
#             fromCenter=False,
#             showCrosshair=True
#         )

#         if roi[2] > 0 and roi[3] > 0:

#             # ROI coordinates are currently based on
#             # the resized display image.

#             display_x, display_y, display_w, display_h = map(
#                 int,
#                 roi
#             )

#             # Convert display coordinates
#             # back to original 4K coordinates.
#             x = int(display_x / scale)
#             y = int(display_y / scale)

#             w = int(display_w / scale)
#             h = int(display_h / scale)

#             current_box = (
#                 x,
#                 y,
#                 w,
#                 h
#             )

#             print(
#                 f"Bounding box: "
#                 f"x={x}, y={y}, w={w}, h={h}"
#             )

#     # C = Clear bounding box
#     elif key == ord('c'):

#         current_box = None

#         print("Bounding box cleared.")


# # -----------------------------
# # Cleanup
# # -----------------------------
# cap.release()

# cv2.destroyAllWindows()

# print("Dataset collection stopped.")
# print(f"Total images saved: {save_count - 844}")



# import cv2
# import os
# import time
# import re

# # =========================================================
# # CONFIGURATION
# # =========================================================

# VIDEO_PATH = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"

# OUTPUT_DIR = "dataset"
# IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
# LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# os.makedirs(IMAGES_DIR, exist_ok=True)
# os.makedirs(LABELS_DIR, exist_ok=True)

# # =========================================================
# # CLASS
# # =========================================================

# CLASS_ID = 0          # 0 = box

# # =========================================================
# # DISPLAY SETTINGS
# # =========================================================

# DISPLAY_WIDTH = 1280
# DISPLAY_HEIGHT = 720

# DELAY = 30

# # =========================================================
# # FIND LAST IMAGE NUMBER
# # =========================================================

# def get_next_save_count():

#     """
#     Find the highest existing box number
#     and continue from the next number.

#     Example:

#         box_000001.jpg
#         box_000002.jpg
#         box_000844.jpg

#     Next number = 845
#     """

#     highest_number = 0

#     # Check existing image files
#     for filename in os.listdir(IMAGES_DIR):

#         match = re.match(
#             r"box_(\d+)\.jpg$",
#             filename
#         )

#         if match:

#             number = int(match.group(1))

#             if number > highest_number:
#                 highest_number = number

#     return highest_number + 1


# # =========================================================
# # GET NEXT IMAGE NUMBER
# # =========================================================

# save_count = get_next_save_count()

# print("==========================================")
# print("       BOX DATASET COLLECTION")
# print("==========================================")

# print(f"Next image number: {save_count}")

# print()


# # =========================================================
# # OPEN RTSP STREAM
# # =========================================================

# cap = cv2.VideoCapture(VIDEO_PATH)

# if not cap.isOpened():

#     raise Exception(
#         "Could not open RTSP stream."
#     )

# # Reduce buffering
# cap.set(
#     cv2.CAP_PROP_BUFFERSIZE,
#     1
# )

# print("Camera connected.")

# print()
# print("Controls:")
# print("S     = Select box")
# print("C     = Clear box")
# print("SPACE = Pause / Resume")
# print("Q     = Quit")
# print("==========================================")


# # =========================================================
# # VARIABLES
# # =========================================================

# paused = False

# current_box = None

# frame = None


# # =========================================================
# # SAVE YOLO ANNOTATION
# # =========================================================

# def save_yolo_annotation(
#     image,
#     box,
#     image_name
# ):

#     """
#     Save image and YOLO annotation.

#     box:
#         (x, y, width, height)
#     """

#     image_height, image_width = image.shape[:2]

#     x, y, w, h = box

#     # -----------------------------------------------------
#     # Convert bounding box to YOLO format
#     # -----------------------------------------------------

#     x_center = (
#         x + w / 2
#     ) / image_width

#     y_center = (
#         y + h / 2
#     ) / image_height

#     w_normalized = (
#         w / image_width
#     )

#     h_normalized = (
#         h / image_height
#     )

#     # -----------------------------------------------------
#     # File paths
#     # -----------------------------------------------------

#     image_path = os.path.join(
#         IMAGES_DIR,
#         image_name + ".jpg"
#     )

#     label_path = os.path.join(
#         LABELS_DIR,
#         image_name + ".txt"
#     )

#     # -----------------------------------------------------
#     # Safety check
#     # -----------------------------------------------------

#     if os.path.exists(image_path):

#         print(
#             f"WARNING: {image_path} already exists."
#         )

#         return False

#     if os.path.exists(label_path):

#         print(
#             f"WARNING: {label_path} already exists."
#         )

#         return False

#     # -----------------------------------------------------
#     # Save image
#     # -----------------------------------------------------

#     success = cv2.imwrite(
#         image_path,
#         image
#     )

#     if not success:

#         print(
#             f"Failed to save image: {image_path}"
#         )

#         return False

#     # -----------------------------------------------------
#     # Save YOLO label
#     # -----------------------------------------------------

#     with open(
#         label_path,
#         "w"
#     ) as f:

#         f.write(
#             f"{CLASS_ID} "
#             f"{x_center:.6f} "
#             f"{y_center:.6f} "
#             f"{w_normalized:.6f} "
#             f"{h_normalized:.6f}"
#         )

#     return True


# # =========================================================
# # CREATE WINDOW
# # =========================================================

# cv2.namedWindow(
#     "Box Dataset Annotator",
#     cv2.WINDOW_NORMAL
# )

# cv2.resizeWindow(
#     "Box Dataset Annotator",
#     DISPLAY_WIDTH,
#     DISPLAY_HEIGHT
# )


# # =========================================================
# # MAIN LOOP
# # =========================================================

# while True:

#     # -----------------------------------------------------
#     # READ FRAME
#     # -----------------------------------------------------

#     if not paused:

#         ret, frame = cap.read()

#         if not ret:

#             print(
#                 "Failed to read frame."
#             )

#             time.sleep(0.1)

#             continue

#     if frame is None:

#         continue


#     # -----------------------------------------------------
#     # ORIGINAL IMAGE SIZE
#     # -----------------------------------------------------

#     original_height, original_width = frame.shape[:2]


#     # -----------------------------------------------------
#     # CALCULATE DISPLAY SCALE
#     # -----------------------------------------------------

#     scale = min(
#         DISPLAY_WIDTH / original_width,
#         DISPLAY_HEIGHT / original_height
#     )

#     display_width = int(
#         original_width * scale
#     )

#     display_height = int(
#         original_height * scale
#     )


#     # -----------------------------------------------------
#     # RESIZE ONLY FOR DISPLAY
#     # -----------------------------------------------------

#     display = cv2.resize(
#         frame,
#         (
#             display_width,
#             display_height
#         ),
#         interpolation=cv2.INTER_AREA
#     )


#     # =====================================================
#     # DRAW BOUNDING BOX
#     # =====================================================

#     if current_box is not None:

#         x, y, w, h = current_box

#         # Convert original coordinates
#         # to display coordinates

#         display_x = int(
#             x * scale
#         )

#         display_y = int(
#             y * scale
#         )

#         display_w = int(
#             w * scale
#         )

#         display_h = int(
#             h * scale
#         )

#         cv2.rectangle(
#             display,
#             (
#                 display_x,
#                 display_y
#             ),
#             (
#                 display_x + display_w,
#                 display_y + display_h
#             ),
#             (0, 255, 0),
#             3
#         )

#         # -------------------------------------------------
#         # Class name
#         # -------------------------------------------------

#         cv2.putText(
#             display,
#             "BOX",
#             (
#                 display_x,
#                 max(
#                     30,
#                     display_y - 10
#                 )
#             ),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 0),
#             2
#         )


#     # =====================================================
#     # INFORMATION
#     # =====================================================

#     cv2.putText(
#         display,
#         f"Saved: {save_count - 1}",
#         (20, 35),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 255),
#         2
#     )

#     cv2.putText(
#         display,
#         f"Resolution: {original_width}x{original_height}",
#         (20, 70),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         (255, 255, 255),
#         2
#     )

#     cv2.putText(
#         display,
#         "S: Select Box | C: Clear | SPACE: Pause | Q: Quit",
#         (
#             20,
#             display_height - 20
#         ),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.6,
#         (255, 255, 255),
#         2
#     )

#     if paused:

#         cv2.putText(
#             display,
#             "PAUSED",
#             (20, 110),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 0, 255),
#             2
#         )


#     # =====================================================
#     # SHOW FRAME
#     # =====================================================

#     cv2.imshow(
#         "Box Dataset Annotator",
#         display
#     )


#     # =====================================================
#     # KEYBOARD
#     # =====================================================

#     key = cv2.waitKey(DELAY) & 0xFF


#     # =====================================================
#     # Q = QUIT
#     # =====================================================

#     if key == ord("q") or key == ord("Q"):

#         break


#     # =====================================================
#     # SPACE = PAUSE / RESUME
#     # =====================================================

#     elif key == ord(" "):

#         paused = not paused

#         if paused:

#             print(
#                 "Stream paused."
#             )

#         else:

#             print(
#                 "Stream resumed."
#             )


#     # =====================================================
#     # S = SELECT BOX
#     # =====================================================

#     elif key == ord("s") or key == ord("S"):

#         paused = True

#         print()
#         print(
#             "Select the BOX using your mouse."
#         )

#         print(
#             "Drag around the complete box."
#         )

#         print(
#             "Press ENTER or SPACE to confirm."
#         )


#         # -------------------------------------------------
#         # Select ROI on display
#         # -------------------------------------------------

#         roi = cv2.selectROI(
#             "Box Dataset Annotator",
#             display,
#             fromCenter=False,
#             showCrosshair=True
#         )


#         rx, ry, rw, rh = map(
#             int,
#             roi
#         )


#         # -------------------------------------------------
#         # Check ROI
#         # -------------------------------------------------

#         if rw > 0 and rh > 0:

#             # Convert display coordinates
#             # back to original resolution

#             x = int(
#                 rx / scale
#             )

#             y = int(
#                 ry / scale
#             )

#             w = int(
#                 rw / scale
#             )

#             h = int(
#                 rh / scale
#             )


#             current_box = (
#                 x,
#                 y,
#                 w,
#                 h
#             )


#             print()
#             print(
#                 "=========================================="
#             )

#             print(
#                 "BOX SELECTED"
#             )

#             print(
#                 "=========================================="
#             )

#             print(
#                 f"x      : {x}"
#             )

#             print(
#                 f"y      : {y}"
#             )

#             print(
#                 f"width  : {w}"
#             )

#             print(
#                 f"height : {h}"
#             )

#             print(
#                 "=========================================="
#             )

#             print(
#                 f"Starting from box_{save_count:06d}"
#             )

#             print(
#                 "Dataset collection started."
#             )

#         else:

#             print(
#                 "No box selected."
#             )


#     # =====================================================
#     # C = CLEAR BOX
#     # =====================================================

#     elif key == ord("c") or key == ord("C"):

#         current_box = None

#         print(
#             "Bounding box cleared."
#         )


#     # =====================================================
#     # SAVE IMAGE
#     # =====================================================

#     if current_box is not None and paused:

#         image_name = (
#             f"box_{save_count:06d}"
#         )

#         saved = save_yolo_annotation(
#             frame,
#             current_box,
#             image_name
#         )

#         if saved:

#             print(
#                 f"Saved: {image_name}.jpg"
#             )

#             save_count += 1


# # =========================================================
# # CLEANUP
# # =========================================================

# cap.release()

# cv2.destroyAllWindows()


# print()
# print(
#     "=========================================="
# )

# print(
#     "Dataset collection stopped."
# )

# print(
#     f"Images saved in this session: "
#     f"{save_count - get_next_save_count()}"
# )

# print(
#     f"Next image number: {save_count}"
# )

# print(
#     f"Dataset folder: {OUTPUT_DIR}"
# )

# print(
#     "=========================================="
# )




import cv2
import os
import time
import re


# =========================================================
# CONFIGURATION
# =========================================================

VIDEO_PATH = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"

OUTPUT_DIR = "dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)


# =========================================================
# CLASS
# =========================================================

CLASS_ID = 0          # 0 = box


# =========================================================
# DATASET SETTINGS
# =========================================================

MIN_BOXES = 1
MAX_BOXES = 2

# Save one image every 3 seconds
SAVE_INTERVAL_SECONDS = 3


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
    Find the highest existing box number.

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
print("       BOX DATASET COLLECTION")
print("==========================================")
print(f"Starting image number: {save_count}")
print(f"Save interval: {SAVE_INTERVAL_SECONDS} seconds")
print(f"Maximum boxes: {MAX_BOXES}")
print("==========================================")


# =========================================================
# OPEN RTSP STREAM
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise Exception(
        "Could not open RTSP stream."
    )

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

print("Camera connected.")


# =========================================================
# VARIABLES
# =========================================================

paused = False

# List of bounding boxes
# Example:
#
# [
#     (x1, y1, w1, h1),
#     (x2, y2, w2, h2)
# ]

boxes = []

frame = None

last_save_time = 0

session_saved = 0


# =========================================================
# SAVE IMAGE + YOLO LABEL
# =========================================================

def save_yolo_annotation(
    image,
    boxes,
    image_name
):

    """
    Save image and YOLO annotation.

    boxes can contain 1 or 2 bounding boxes.

    Example:

        [
            (x1, y1, w1, h1),
            (x2, y2, w2, h2)
        ]
    """

    image_height, image_width = image.shape[:2]

    image_path = os.path.join(
        IMAGES_DIR,
        image_name + ".jpg"
    )

    label_path = os.path.join(
        LABELS_DIR,
        image_name + ".txt"
    )


    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if os.path.exists(image_path):

        print(
            f"WARNING: {image_path} already exists."
        )

        return False


    if os.path.exists(label_path):

        print(
            f"WARNING: {label_path} already exists."
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


    # -----------------------------------------------------
    # Save YOLO labels
    # -----------------------------------------------------

    with open(
        label_path,
        "w"
    ) as f:

        for box in boxes:

            x, y, w, h = box

            # ---------------------------------------------
            # Convert to YOLO format
            # ---------------------------------------------

            x_center = (
                x + w / 2
            ) / image_width

            y_center = (
                y + h / 2
            ) / image_height

            w_normalized = (
                w / image_width
            )

            h_normalized = (
                h / image_height
            )


            # ---------------------------------------------
            # Write one line per box
            # ---------------------------------------------

            f.write(
                f"{CLASS_ID} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{w_normalized:.6f} "
                f"{h_normalized:.6f}\n"
            )


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

        ret, frame = cap.read()

        if not ret:

            print(
                "Failed to read frame."
            )

            time.sleep(0.1)

            continue


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


        # Convert original coordinates
        # to display coordinates

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
        f"Resolution: {original_width}x{original_height}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # -----------------------------------------------------
    # Controls
    # -----------------------------------------------------

    cv2.putText(
        display,
        "S: Add Box | C: Clear | SPACE: Pause | Q: Quit",
        (
            20,
            display_height - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


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
                "==========================================")


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
            # Save image + all selected boxes
            # -------------------------------------------------

            saved = save_yolo_annotation(
                frame,
                boxes,
                image_name
            )


            if saved:

                print(
                    f"Saved: {image_name}.jpg "
                    f"with {len(boxes)} box(es)"
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

