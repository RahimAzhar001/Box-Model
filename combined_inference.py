
import cv2
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
import time
import os


# ============================================================
# SETTINGS
# ============================================================

# ---------- RTSP CAMERA ----------
CAMERA_URL = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"


# ---------- MODEL PATHS ----------
BOX_MODEL_PATH = "efficientnet_b0_box_nobox_best.pth"

MATERIAL_MODEL_PATH = "efficientnet_b0_empty_full_final.pth"


# ============================================================
# ORIGINAL CAMERA RESOLUTION
# ============================================================

CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160


# ============================================================
# FIXED ROIs
#
# Format:
# (x, y, width, height)
#
# Coordinates are based on the ORIGINAL
# 3840x2160 camera frame.
# ============================================================

LEFT_ROI = (700, 456, 711, 837)

RIGHT_ROI = (2250, 510, 651, 891)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

RECONNECT_DELAY = 2

WINDOW_NAME = "Box + Material Detection"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD EFFICIENTNET MODEL
# ============================================================

def load_efficientnet_model(
    model_path,
    class_names
):

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nModel not found:\n{model_path}"
        )


    print("\nLoading model:")
    print(model_path)


    checkpoint = torch.load(
        model_path,
        map_location=device
    )


    # --------------------------------------------------------
    # Get model state dictionary
    #
    # Supports:
    #   model_state_dict
    #   state_dict
    #   direct state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # Remove "module." prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value


    # --------------------------------------------------------
    # Create EfficientNet-B0
    # --------------------------------------------------------

    model = efficientnet_b0(
        weights=None
    )


    # --------------------------------------------------------
    # Configure final classification layer
    # --------------------------------------------------------

    num_features = (
        model.classifier[1].in_features
    )


    model.classifier[1] = nn.Linear(
        num_features,
        len(class_names)
    )


    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict
    )


    # --------------------------------------------------------
    # Move model to device
    # --------------------------------------------------------

    model = model.to(device)

    model.eval()


    print(
        "Classes:",
        class_names
    )

    print(
        "Model loaded successfully."
    )


    return model, class_names


# ============================================================
# LOAD BOTH MODELS
# ============================================================

print("\n========================================")
print("LOADING MODELS")
print("========================================")


# IMPORTANT:
#
# The BOX model checkpoint does NOT contain
# class_names, so we explicitly provide them.
#
# 0 = BOX
# 1 = NOBOX
#
box_model, box_classes = load_efficientnet_model(
    BOX_MODEL_PATH,
    ["BOX", "NOBOX"]
)


# The material model was trained as:
#
# 0 = EMPTY
# 1 = FULL
#
material_model, material_classes = load_efficientnet_model(
    MATERIAL_MODEL_PATH,
    ["EMPTY", "FULL"]
)


# ============================================================
# PRINT CLASS MAPPINGS
# ============================================================

print("\n========================================")
print("CLASS MAPPINGS")
print("========================================")

print(
    "Box model classes:",
    box_classes
)

print(
    "Material model classes:",
    material_classes
)


# ============================================================
# FIND CLASS INDEX
# ============================================================

def find_class_index(
    class_names,
    possible_names
):

    normalized = [

        str(name)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")

        for name in class_names

    ]


    for possible_name in possible_names:

        possible_name = (

            possible_name
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")

        )


        if possible_name in normalized:

            return normalized.index(
                possible_name
            )


    return None


# ============================================================
# BOX CLASS MAPPING
# ============================================================

BOX_PRESENT_INDEX = find_class_index(
    box_classes,

    [
        "box",
        "present",
        "boxpresent",
        "boxavailable",
        "available"
    ]
)


BOX_MISSING_INDEX = find_class_index(
    box_classes,

    [
        "nobox",
        "missing",
        "boxmissing",
        "notbox",
        "notpresent"
    ]
)


# ============================================================
# MATERIAL CLASS MAPPING
# ============================================================

EMPTY_INDEX = find_class_index(
    material_classes,

    [
        "empty",
        "materialunavailable",
        "nomaterial"
    ]
)


FULL_INDEX = find_class_index(
    material_classes,

    [
        "full",
        "materialavailable",
        "materialpresent"
    ]
)


# ============================================================
# PRINT CLASS INDICES
# ============================================================

print(
    "\nBox present index:",
    BOX_PRESENT_INDEX
)

print(
    "Box missing index:",
    BOX_MISSING_INDEX
)

print(
    "Empty index:",
    EMPTY_INDEX
)

print(
    "Full index:",
    FULL_INDEX
)


# ============================================================
# CHECK CLASS MAPPING
# ============================================================

if (
    BOX_PRESENT_INDEX is None
    or
    BOX_MISSING_INDEX is None
):

    raise ValueError(
        "\nCould not identify BOX / NOBOX classes.\n"
        f"Classes found: {box_classes}"
    )


if (
    EMPTY_INDEX is None
    or
    FULL_INDEX is None
):

    raise ValueError(
        "\nCould not identify EMPTY / FULL classes.\n"
        f"Classes found: {material_classes}"
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

@torch.no_grad()
def predict(
    model,
    image
):

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    tensor = transform(
        image_rgb
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(0)


    # --------------------------------------------------------
    # Move to GPU / CPU
    # --------------------------------------------------------

    tensor = tensor.to(device)


    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    output = model(
        tensor
    )


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=1
    )


    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )


    return (
        prediction.item(),
        confidence.item()
    )


# ============================================================
# ROI CROP FUNCTION
# ============================================================

def crop_roi(
    frame,
    roi
):

    x, y, w, h = roi


    frame_height, frame_width = (
        frame.shape[:2]
    )


    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )


    x2 = min(
        frame_width,
        x + w
    )

    y2 = min(
        frame_height,
        y + h
    )


    if (
        x1 >= x2
        or
        y1 >= y2
    ):

        return None


    return frame[
        y1:y2,
        x1:x2
    ]


# ============================================================
# DRAW ROI
# ============================================================

def draw_roi(
    frame,
    roi,
    label,
    confidence,
    color
):

    x, y, w, h = roi


    # --------------------------------------------------------
    # Original 3840x2160 -> 1920x1080
    # --------------------------------------------------------

    scale_x = (
        DISPLAY_WIDTH /
        CAMERA_WIDTH
    )

    scale_y = (
        DISPLAY_HEIGHT /
        CAMERA_HEIGHT
    )


    x1 = int(
        x * scale_x
    )

    y1 = int(
        y * scale_y
    )


    x2 = int(
        (x + w) * scale_x
    )

    y2 = int(
        (y + h) * scale_y
    )


    # --------------------------------------------------------
    # ROI rectangle
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )


    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    text = (
        f"{label} "
        f"{confidence * 100:.1f}%"
    )


    cv2.putText(
        frame,
        text,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# CONNECT CAMERA
# ============================================================

def connect_camera():

    print(
        "\nConnecting to camera..."
    )


    cap = cv2.VideoCapture(
        CAMERA_URL,
        cv2.CAP_FFMPEG
    )


    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )


    if cap.isOpened():

        print(
            "Camera connected."
        )

        return cap


    print(
        "Could not connect to camera."
    )


    return None


# ============================================================
# MAIN
# ============================================================

cap = None


print(
    "\n========================================"
)

print(
    "STARTING COMBINED DETECTION"
)

print(
    "========================================"
)


while True:


    # ========================================================
    # CONNECT / RECONNECT
    # ========================================================

    if (
        cap is None
        or
        not cap.isOpened()
    ):

        if cap is not None:

            cap.release()


        cap = connect_camera()


        if cap is None:

            time.sleep(
                RECONNECT_DELAY
            )

            continue


    # ========================================================
    # READ FRAME
    # ========================================================

    ret, frame = cap.read()


    if not ret:

        print(
            "Stream disconnected."
        )


        cap.release()

        cap = None


        time.sleep(
            RECONNECT_DELAY
        )


        continue


    # ========================================================
    # CHECK CAMERA RESOLUTION
    # ========================================================

    frame_height, frame_width = (
        frame.shape[:2]
    )


    # ========================================================
    # CROP LEFT ROI
    #
    # IMPORTANT:
    # Crop BEFORE resizing to 1920x1080.
    # ========================================================

    left_roi = crop_roi(
        frame,
        LEFT_ROI
    )


    # ========================================================
    # CROP RIGHT ROI
    # ========================================================

    right_roi = crop_roi(
        frame,
        RIGHT_ROI
    )


    if (
        left_roi is None
        or
        right_roi is None
    ):

        print(
            "ERROR: ROI is outside camera frame."
        )

        continue


    # ========================================================
    # MODEL 1 - LEFT BOX
    # ========================================================

    left_box_prediction, left_box_confidence = predict(
        box_model,
        left_roi
    )


    # ========================================================
    # MODEL 1 - RIGHT BOX
    # ========================================================

    right_box_prediction, right_box_confidence = predict(
        box_model,
        right_roi
    )


    # ========================================================
    # DEFAULT MATERIAL RESULTS
    # ========================================================

    left_material_prediction = None

    left_material_confidence = 0.0


    right_material_prediction = None

    right_material_confidence = 0.0


    # ========================================================
    # LEFT BOX DECISION
    # ========================================================

    if (
        left_box_prediction
        ==
        BOX_MISSING_INDEX
    ):

        left_status = (
            "BOX MISSING / VIOLATION"
        )


        left_color = (
            0,
            0,
            255
        )


        # ----------------------------------------------------
        # MATERIAL MODEL IS NOT RUN
        # ----------------------------------------------------


    else:

        # ----------------------------------------------------
        # BOX PRESENT
        # Run material model
        # ----------------------------------------------------

        left_material_prediction, left_material_confidence = predict(
            material_model,
            left_roi
        )


        if (
            left_material_prediction
            ==
            FULL_INDEX
        ):

            left_status = (
                "BOX PRESENT / MATERIAL AVAILABLE"
            )


            left_color = (
                0,
                255,
                0
            )


        else:

            left_status = (
                "BOX PRESENT / MATERIAL UNAVAILABLE"
            )


            left_color = (
                0,
                165,
                255
            )


    # ========================================================
    # RIGHT BOX DECISION
    # ========================================================

    if (
        right_box_prediction
        ==
        BOX_MISSING_INDEX
    ):

        right_status = (
            "BOX MISSING / VIOLATION"
        )


        right_color = (
            0,
            0,
            255
        )


        # ----------------------------------------------------
        # MATERIAL MODEL IS NOT RUN
        # ----------------------------------------------------


    else:

        # ----------------------------------------------------
        # BOX PRESENT
        # Run material model
        # ----------------------------------------------------

        right_material_prediction, right_material_confidence = predict(
            material_model,
            right_roi
        )


        if (
            right_material_prediction
            ==
            FULL_INDEX
        ):

            right_status = (
                "BOX PRESENT / MATERIAL AVAILABLE"
            )


            right_color = (
                0,
                255,
                0
            )


        else:

            right_status = (
                "BOX PRESENT / MATERIAL UNAVAILABLE"
            )


            right_color = (
                0,
                165,
                255
            )


    # ========================================================
    # RESIZE FRAME FOR DISPLAY
    #
    # Detection has already been performed on the
    # original 3840x2160 frame.
    # ========================================================

    display_frame = cv2.resize(
        frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        ),
        interpolation=cv2.INTER_AREA
    )


    # ========================================================
    # DRAW LEFT ROI
    # ========================================================

    draw_roi(
        display_frame,
        LEFT_ROI,
        left_status,

        (
            left_material_confidence
            if left_material_prediction is not None
            else left_box_confidence
        ),

        left_color
    )


    # ========================================================
    # DRAW RIGHT ROI
    # ========================================================

    draw_roi(
        display_frame,
        RIGHT_ROI,
        right_status,

        (
            right_material_confidence
            if right_material_prediction is not None
            else right_box_confidence
        ),

        right_color
    )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        display_frame,
        (10, 10),
        (760, 160),
        (0, 0, 0),
        -1
    )


    # --------------------------------------------------------
    # LEFT STATUS
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"LEFT: {left_status}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        left_color,
        2
    )


    # --------------------------------------------------------
    # RIGHT STATUS
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"RIGHT: {right_status}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        right_color,
        2
    )


    # --------------------------------------------------------
    # Box confidences
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        f"Left Box: {left_box_confidence * 100:.1f}%",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display_frame,
        f"Right Box: {right_box_confidence * 100:.1f}%",
        (220, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        "Display: 1920x1080",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

if cap is not None:

    cap.release()


cv2.destroyAllWindows()


print(
    "\nProgram stopped."
)
