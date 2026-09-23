
import cv2
import time
import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

RTSP_URL = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"

WINDOW_NAME = "4th Press - Slider Motion Detection"

CONFIG_FILE = "slider_config.json"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# ============================================================
# BACKGROUND SUBTRACTION
# ============================================================

BG_HISTORY = 500
BG_VAR_THRESHOLD = 25


# ============================================================
# MOTION THRESHOLDS
# ============================================================

ZONE_A_THRESHOLD = 1000
ZONE_B_THRESHOLD = 1000


# ============================================================
# TEMPORAL SETTINGS
# ============================================================

REQUIRED_FRAMES = 3

MAX_TRANSITION_TIME = 5.0


# ============================================================
# MORPHOLOGY
# ============================================================

KERNEL_SIZE = 5


# ============================================================
# RTSP RECONNECTION SETTINGS
# ============================================================

# Number of consecutive failed reads before reconnecting
MAX_FAILED_READS = 10

# Seconds to wait before reconnecting
RECONNECT_DELAY = 3

# Number of frames used to rebuild the background model
BACKGROUND_WARMUP_FRAMES = 30


# ============================================================
# GLOBAL VARIABLES
# ============================================================

drawing = False
start_point = None
current_point = None

selection_complete = False

selected_rect = None

selection_mode = "SLIDER"

slider_roi = None
zone_a = None
zone_b = None

saved_config = False


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_point
    global current_point
    global selected_rect

    if selection_complete:
        return

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_point = (x, y)
        current_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:
            current_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        current_point = (x, y)

        x1 = min(start_point[0], current_point[0])
        y1 = min(start_point[1], current_point[1])

        x2 = max(start_point[0], current_point[0])
        y2 = max(start_point[1], current_point[1])

        selected_rect = (
            x1,
            y1,
            x2 - x1,
            y2 - y1
        )


# ============================================================
# DRAW RECTANGLE
# ============================================================

def draw_selection(frame):

    output = frame.copy()

    if drawing and start_point and current_point:

        x1 = min(start_point[0], current_point[0])
        y1 = min(start_point[1], current_point[1])

        x2 = max(start_point[0], current_point[0])
        y2 = max(start_point[1], current_point[1])

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

    elif selected_rect:

        x, y, w, h = selected_rect

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

    return output


# ============================================================
# GET RECTANGLE FROM USER
# ============================================================

def select_rectangle(frame, title):

    global drawing
    global start_point
    global current_point
    global selected_rect
    global selection_complete

    drawing = False
    start_point = None
    current_point = None
    selected_rect = None
    selection_complete = False

    cv2.namedWindow(WINDOW_NAME)

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print("LEFT CLICK + DRAG to draw rectangle")
    print("ENTER = confirm")
    print("R     = redraw")
    print("Q     = quit")

    while True:

        display = draw_selection(frame)

        cv2.putText(
            display,
            title,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Drag rectangle | ENTER confirm | R redraw | Q quit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 13:

            if selected_rect is not None:

                selection_complete = True

                return selected_rect

        elif key == ord("r"):

            selected_rect = None
            start_point = None
            current_point = None

        elif key == ord("q"):

            cv2.destroyAllWindows()

            exit()


# ============================================================
# SAVE CONFIGURATION
# ============================================================

def save_configuration():

    config = {

        "slider_roi": slider_roi,

        "zone_a": zone_a,

        "zone_b": zone_b

    }

    with open(
        CONFIG_FILE,
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    print()
    print("=" * 60)
    print("CONFIGURATION SAVED")
    print("=" * 60)

    print(
        f"Slider ROI : {slider_roi}"
    )

    print(
        f"Zone A     : {zone_a}"
    )

    print(
        f"Zone B     : {zone_b}"
    )

    print(
        f"Saved to   : {CONFIG_FILE}"
    )


# ============================================================
# LOAD CONFIGURATION
# ============================================================

def load_configuration():

    global slider_roi
    global zone_a
    global zone_b

    if not os.path.exists(CONFIG_FILE):

        return False

    try:

        with open(
            CONFIG_FILE,
            "r"
        ) as f:

            config = json.load(f)

        slider_roi = tuple(
            config["slider_roi"]
        )

        zone_a = tuple(
            config["zone_a"]
        )

        zone_b = tuple(
            config["zone_b"]
        )

        print()
        print("Existing configuration found.")

        return True

    except Exception as e:

        print(
            f"Could not load configuration: {e}"
        )

        return False


# ============================================================
# DRAW FINAL ROI AND ZONES
# ============================================================

def draw_final_overlay(frame):

    output = frame.copy()

    # --------------------------------------------------------
    # Slider ROI
    # --------------------------------------------------------

    sx, sy, sw, sh = slider_roi

    cv2.rectangle(
        output,
        (sx, sy),
        (sx + sw, sy + sh),
        (255, 255, 0),
        3
    )

    cv2.putText(
        output,
        "SLIDER",
        (sx, sy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # --------------------------------------------------------
    # Zone A
    # --------------------------------------------------------

    ax, ay, aw, ah = zone_a

    cv2.rectangle(
        output,
        (sx + ax, sy + ay),
        (
            sx + ax + aw,
            sy + ay + ah
        ),
        (0, 255, 0),
        3
    )

    cv2.putText(
        output,
        "ZONE A",
        (sx + ax, sy + ay - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # Zone B
    # --------------------------------------------------------

    bx, by, bw, bh = zone_b

    cv2.rectangle(
        output,
        (sx + bx, sy + by),
        (
            sx + bx + bw,
            sy + by + bh
        ),
        (0, 0, 255),
        3
    )

    cv2.putText(
        output,
        "ZONE B",
        (sx + bx, sy + by - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    return output


# ============================================================
# CONNECT TO RTSP CAMERA
# ============================================================

def connect_camera():

    print()
    print("[RTSP] Connecting to camera...")

    cap = cv2.VideoCapture(
        RTSP_URL,
        cv2.CAP_FFMPEG
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    # Allow connection to initialize
    time.sleep(1)

    if cap.isOpened():

        print("[RTSP] Camera connected successfully.")

        return cap

    print("[RTSP] Camera connection failed.")

    cap.release()

    return None


# ============================================================
# RECONNECT CAMERA
# ============================================================

def reconnect_camera():

    print()
    print("=" * 60)
    print("[RTSP] CAMERA CONNECTION LOST")
    print("[RTSP] Starting reconnection process...")
    print("=" * 60)

    reconnect_attempt = 0

    while True:

        reconnect_attempt += 1

        print(
            f"[RTSP] Reconnection attempt "
            f"#{reconnect_attempt}"
        )

        cap = connect_camera()

        if cap is not None:

            print("[RTSP] Testing new connection...")

            test_ret, test_frame = cap.read()

            if test_ret and test_frame is not None:

                print(
                    "[RTSP] Reconnection successful."
                )

                return cap, test_frame

            else:

                print(
                    "[RTSP] Connected but frame "
                    "read failed."
                )

                cap.release()

        print(
            f"[RTSP] Waiting "
            f"{RECONNECT_DELAY} seconds..."
        )

        time.sleep(RECONNECT_DELAY)


# ============================================================
# MAIN PROGRAM
# ============================================================

print("=" * 60)
print("       4TH PRESS SLIDER MOTION DETECTOR")
print("=" * 60)

print()
print("A -> B = FORWARD")
print("B -> A = BACKWARD")


# ============================================================
# INITIAL CAMERA CONNECTION
# ============================================================

cap = None

while cap is None:

    cap = connect_camera()

    if cap is None:

        print(
            f"[RTSP] Initial connection failed."
        )

        print(
            f"[RTSP] Retrying in "
            f"{RECONNECT_DELAY} seconds..."
        )

        time.sleep(RECONNECT_DELAY)


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, frame = cap.read()

while not ret:

    print(
        "[RTSP] Initial frame read failed."
    )

    cap.release()

    cap = None

    print(
        f"[RTSP] Reconnecting in "
        f"{RECONNECT_DELAY} seconds..."
    )

    time.sleep(RECONNECT_DELAY)

    cap = connect_camera()

    if cap is not None:

        ret, frame = cap.read()


if not ret:

    print(
        "[RTSP] ERROR: Could not obtain first frame."
    )

    if cap is not None:

        cap.release()

    exit()


original_height, original_width = frame.shape[:2]

print(
    f"Camera resolution: "
    f"{original_width}x{original_height}"
)


# ============================================================
# SCALE FRAME FOR DISPLAY
# ============================================================

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

print(
    f"Display resolution: "
    f"{display_width}x{display_height}"
)


# ============================================================
# EXISTING CONFIGURATION?
# ============================================================

use_existing = False

if load_configuration():

    print()
    print("Press:")
    print("  Y = use existing configuration")
    print("  N = draw new configuration")

    while True:

        key = cv2.waitKey(0) & 0xFF

        if key == ord("y"):

            use_existing = True

            break

        elif key == ord("n"):

            use_existing = False

            break


# ============================================================
# DRAW ROI AND ZONES
# ============================================================

if not use_existing:

    # --------------------------------------------------------
    # SLIDER ROI
    # --------------------------------------------------------

    display_frame = cv2.resize(
        frame,
        (
            display_width,
            display_height
        )
    )

    slider_display = select_rectangle(
        display_frame,
        "STEP 1: DRAW SLIDER ROI"
    )

    # Convert display coordinates
    # back to original image coordinates

    sx, sy, sw, sh = slider_display

    slider_roi = (
        int(sx / scale),
        int(sy / scale),
        int(sw / scale),
        int(sh / scale)
    )

    print(
        "Slider ROI:",
        slider_roi
    )

    # --------------------------------------------------------
    # Extract slider from original frame
    # --------------------------------------------------------

    sx, sy, sw, sh = slider_roi

    slider_original = frame[
        sy:sy + sh,
        sx:sx + sw
    ]

    slider_display = cv2.resize(
        slider_original,
        (
            max(1, int(sw * scale)),
            max(1, int(sh * scale))
        )
    )

    # --------------------------------------------------------
    # ZONE A
    # --------------------------------------------------------

    zone_a_display = select_rectangle(
        slider_display,
        "STEP 2: DRAW ZONE A"
    )

    zax, zay, zaw, zah = zone_a_display

    zone_a = (
        int(zax / scale),
        int(zay / scale),
        int(zaw / scale),
        int(zah / scale)
    )

    print(
        "Zone A:",
        zone_a
    )

    # --------------------------------------------------------
    # ZONE B
    # --------------------------------------------------------

    zone_b_display = select_rectangle(
        slider_display,
        "STEP 3: DRAW ZONE B"
    )

    zbx, zby, zbw, zbh = zone_b_display

    zone_b = (
        int(zbx / scale),
        int(zby / scale),
        int(zbw / scale),
        int(zbh / scale)
    )

    print(
        "Zone B:",
        zone_b
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_configuration()


# ============================================================
# CREATE BACKGROUND SUBTRACTOR
# ============================================================

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=BG_HISTORY,
    varThreshold=BG_VAR_THRESHOLD,
    detectShadows=False
)


kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        KERNEL_SIZE,
        KERNEL_SIZE
    )
)


# ============================================================
# STATE MACHINE
# ============================================================

state = "WAITING"

state_start_time = 0

a_counter = 0
b_counter = 0

last_event = "NONE"
last_event_time = 0


# ============================================================
# BACKGROUND WARM-UP
# ============================================================

print()
print("=" * 60)
print("WARMING UP BACKGROUND MODEL")
print("=" * 60)

warmup_count = 0

while warmup_count < BACKGROUND_WARMUP_FRAMES:

    ret, warmup_frame = cap.read()

    if not ret:

        print(
            "[RTSP] Warm-up frame failed."
        )

        cap.release()

        cap, warmup_frame = reconnect_camera()

        if warmup_frame is None:

            continue

    sx, sy, sw, sh = slider_roi

    slider = warmup_frame[
        sy:sy + sh,
        sx:sx + sw
    ]

    if slider.size > 0:

        bg_subtractor.apply(slider)

        warmup_count += 1

print(
    "[RTSP] Background model ready."
)


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("=" * 60)
print("STARTING SLIDER MOTION DETECTION")
print("=" * 60)

print("A -> B = FORWARD")
print("B -> A = BACKWARD")
print("Q = quit")
print("R = reset state")
print()


failed_reads = 0


while True:

    # ========================================================
    # READ FRAME
    # ========================================================

    ret, frame = cap.read()


    # ========================================================
    # FRAME READ FAILED
    # ========================================================

    if not ret:

        failed_reads += 1

        print(
            f"[RTSP] Frame read failed "
            f"({failed_reads}/{MAX_FAILED_READS})"
        )

        # ----------------------------------------------------
        # Too many failures
        # ----------------------------------------------------

        if failed_reads >= MAX_FAILED_READS:

            # -----------------------------------------------
            # Release and reconnect
            # -----------------------------------------------

            cap.release()

            cap, frame = reconnect_camera()

            if cap is None:

                print(
                    "[RTSP] Reconnection failed."
                )

                continue


            # -----------------------------------------------
            # Reset failed counter
            # -----------------------------------------------

            failed_reads = 0


            # -----------------------------------------------
            # Reset motion state
            # -----------------------------------------------

            state = "WAITING"

            state_start_time = 0

            a_counter = 0
            b_counter = 0


            # -----------------------------------------------
            # Reset background model
            # -----------------------------------------------

            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=BG_HISTORY,
                varThreshold=BG_VAR_THRESHOLD,
                detectShadows=False
            )


            # -----------------------------------------------
            # Warm up background model again
            # -----------------------------------------------

            print()
            print(
                "[RTSP] Rebuilding background model..."
            )

            warmup_count = 0

            while warmup_count < BACKGROUND_WARMUP_FRAMES:

                warm_ret, warm_frame = cap.read()

                if not warm_ret:

                    print(
                        "[RTSP] Warm-up read failed."
                    )

                    cap.release()

                    cap, warm_frame = reconnect_camera()

                    continue


                sx, sy, sw, sh = slider_roi

                slider = warm_frame[
                    sy:sy + sh,
                    sx:sx + sw
                ]


                if slider.size > 0:

                    bg_subtractor.apply(slider)

                    warmup_count += 1


            print(
                "[RTSP] Background model rebuilt."
            )


            # We already have a valid frame
            # from the reconnection function.
            ret = True


        else:

            # ------------------------------------------------
            # Small delay before retry
            # ------------------------------------------------

            time.sleep(0.05)

            continue


    else:

        # Successful frame
        failed_reads = 0


    # ========================================================
    # SLIDER ROI
    # ========================================================

    sx, sy, sw, sh = slider_roi

    slider = frame[
        sy:sy + sh,
        sx:sx + sw
    ]


    if slider.size == 0:

        continue


    # ========================================================
    # BACKGROUND SUBTRACTION
    # ========================================================

    mask = bg_subtractor.apply(
        slider
    )


    # ========================================================
    # MORPHOLOGICAL FILTERING
    # ========================================================

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # ========================================================
    # ZONE A
    # ========================================================

    ax, ay, aw, ah = zone_a

    zone_a_mask = mask[
        ay:ay + ah,
        ax:ax + aw
    ]


    if zone_a_mask.size > 0:

        zone_a_pixels = cv2.countNonZero(
            zone_a_mask
        )

    else:

        zone_a_pixels = 0


    zone_a_active = (
        zone_a_pixels >= ZONE_A_THRESHOLD
    )


    # ========================================================
    # ZONE B
    # ========================================================

    bx, by, bw, bh = zone_b

    zone_b_mask = mask[
        by:by + bh,
        bx:bx + bw
    ]


    if zone_b_mask.size > 0:

        zone_b_pixels = cv2.countNonZero(
            zone_b_mask
        )

    else:

        zone_b_pixels = 0


    zone_b_active = (
        zone_b_pixels >= ZONE_B_THRESHOLD
    )


    # ========================================================
    # TEMPORAL CONFIRMATION
    # ========================================================

    if zone_a_active:

        a_counter += 1

    else:

        a_counter = 0


    if zone_b_active:

        b_counter += 1

    else:

        b_counter = 0


    a_confirmed = (
        a_counter >= REQUIRED_FRAMES
    )


    b_confirmed = (
        b_counter >= REQUIRED_FRAMES
    )


    # ========================================================
    # STATE MACHINE
    # ========================================================

    current_time = time.time()


    # ========================================================
    # WAITING
    # ========================================================

    if state == "WAITING":

        if a_confirmed:

            state = "A_DETECTED"

            state_start_time = current_time

            print(
                "[STATE] Zone A detected"
            )


        elif b_confirmed:

            state = "B_DETECTED"

            state_start_time = current_time

            print(
                "[STATE] Zone B detected"
            )


    # ========================================================
    # A DETECTED
    # ========================================================

    elif state == "A_DETECTED":

        elapsed = (
            current_time -
            state_start_time
        )


        # ----------------------------------------------------
        # A -> B
        # ----------------------------------------------------

        if b_confirmed:

            last_event = "FORWARD"

            last_event_time = current_time

            print(
                ">>> FORWARD: A -> B"
            )

            state = "WAITING"

            a_counter = 0
            b_counter = 0


        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        elif elapsed > MAX_TRANSITION_TIME:

            print(
                "[TIMEOUT] A -> B"
            )

            state = "WAITING"

            a_counter = 0
            b_counter = 0


    # ========================================================
    # B DETECTED
    # ========================================================

    elif state == "B_DETECTED":

        elapsed = (
            current_time -
            state_start_time
        )


        # ----------------------------------------------------
        # B -> A
        # ----------------------------------------------------

        if a_confirmed:

            last_event = "BACKWARD"

            last_event_time = current_time

            print(
                "<<< BACKWARD: B -> A"
            )

            state = "WAITING"

            a_counter = 0
            b_counter = 0


        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        elif elapsed > MAX_TRANSITION_TIME:

            print(
                "[TIMEOUT] B -> A"
            )

            state = "WAITING"

            a_counter = 0
            b_counter = 0


    # ========================================================
    # DISPLAY
    # ========================================================

    display = draw_final_overlay(
        frame
    )


    # ========================================================
    # STATE TEXT
    # ========================================================

    cv2.putText(
        display,
        f"STATE: {state}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )


    # ========================================================
    # ZONE A PIXELS
    # ========================================================

    cv2.putText(
        display,
        f"A: {zone_a_pixels}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ========================================================
    # ZONE B PIXELS
    # ========================================================

    cv2.putText(
        display,
        f"B: {zone_b_pixels}",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    # ========================================================
    # LAST EVENT
    # ========================================================

    if (
        current_time -
        last_event_time
        < 2
    ):

        cv2.putText(
            display,
            f"MOVEMENT: {last_event}",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            3
        )


    # ========================================================
    # RESIZE DISPLAY
    # ========================================================

    display = cv2.resize(
        display,
        (
            display_width,
            display_height
        )
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    elif key == ord("r"):

        state = "WAITING"

        a_counter = 0
        b_counter = 0

        print(
            "[RESET] State reset."
        )


# ============================================================
# CLEANUP
# ============================================================

if cap is not None:

    cap.release()

cv2.destroyAllWindows()

print()
print("Program stopped.")

