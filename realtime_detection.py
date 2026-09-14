
import cv2
import torch
import time
import os
import threading

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# CONFIG
# ============================================================

RTSP_URL = r"rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"

MODEL_PATH = "fasterrcnn_checkpoint.pth"

CONFIDENCE_THRESHOLD = 0.50

# Your model was trained using 800x800
MODEL_SIZE = 800

# Display size
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# ============================================================
# CPU
# ============================================================

device = torch.device("cpu")

cpu_count = os.cpu_count() or 4

torch.set_num_threads(max(1, cpu_count - 1))

print("=" * 65)
print("       LOW LATENCY FASTER R-CNN RTSP")
print("=" * 65)

print("Device:", device)
print("CPU cores:", cpu_count)
print("PyTorch threads:", torch.get_num_threads())


# ============================================================
# MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found:")
    print(os.path.abspath(MODEL_PATH))
    exit()


print("\nLoading Faster R-CNN...")

model = fasterrcnn_resnet50_fpn_v2(
    weights=None,
    min_size=MODEL_SIZE,
    max_size=MODEL_SIZE
)

# 0 = background
# 1 = box
NUM_CLASSES = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    NUM_CLASSES
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()

print("Model loaded.")

if "epoch" in checkpoint:
    print("Epochs:", checkpoint["epoch"])

if "best_loss" in checkpoint:
    print("Best loss:", checkpoint["best_loss"])


# ============================================================
# SHARED CAMERA FRAME
# ============================================================

latest_frame = None

frame_lock = threading.Lock()

camera_running = True

camera_connected = False


# ============================================================
# SHARED DETECTIONS
# ============================================================

latest_detections = []

detection_lock = threading.Lock()

inference_ms = 0.0

inference_fps = 0.0


# ============================================================
# CAMERA CAPTURE THREAD
# ============================================================

def camera_thread():

    global latest_frame
    global camera_running
    global camera_connected

    cap = None

    while camera_running:

        # ----------------------------------------------------
        # Connect
        # ----------------------------------------------------

        if cap is None or not cap.isOpened():

            print("Connecting to camera...")

            if cap is not None:
                cap.release()

            cap = cv2.VideoCapture(
                RTSP_URL,
                cv2.CAP_FFMPEG
            )

            # Minimize buffering
            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

            if not cap.isOpened():

                camera_connected = False

                print("Connection failed. Retrying...")

                time.sleep(0.5)

                continue

            camera_connected = True

            print("Camera connected!")


        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            camera_connected = False

            print("Frame failed. Reconnecting...")

            cap.release()

            cap = None

            time.sleep(0.2)

            continue


        # ----------------------------------------------------
        # CRITICAL
        #
        # Replace old frame.
        #
        # DO NOT queue frames.
        # ----------------------------------------------------

        with frame_lock:

            latest_frame = frame


    if cap is not None:
        cap.release()


# ============================================================
# FASTER R-CNN INFERENCE THREAD
# ============================================================

def inference_thread():

    global latest_detections
    global inference_ms
    global inference_fps
    global camera_running

    last_processed_frame = None

    detection_counter = 0

    fps_start = time.time()


    while camera_running:

        # ----------------------------------------------------
        # Get newest camera frame
        # ----------------------------------------------------

        with frame_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()


        if frame is None:

            time.sleep(0.005)

            continue


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # If this is the same frame we already processed,
        # don't process it again.
        # ----------------------------------------------------

        if last_processed_frame is frame:

            time.sleep(0.001)

            continue


        last_processed_frame = frame


        # ----------------------------------------------------
        # Original dimensions
        # ----------------------------------------------------

        original_height, original_width = frame.shape[:2]


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        resized = cv2.resize(
            frame,
            (MODEL_SIZE, MODEL_SIZE),
            interpolation=cv2.INTER_LINEAR
        )


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # NumPy -> Tensor
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            rgb
        ).permute(
            2, 0, 1
        ).float() / 255.0


        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        start = time.perf_counter()

        with torch.inference_mode():

            prediction = model([
                image_tensor
            ])[0]


        elapsed = time.perf_counter() - start

        inference_ms = elapsed * 1000


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        detection_counter += 1

        fps_elapsed = time.time() - fps_start

        if fps_elapsed >= 1.0:

            inference_fps = (
                detection_counter / fps_elapsed
            )

            detection_counter = 0

            fps_start = time.time()


        # ----------------------------------------------------
        # GET DETECTIONS
        # ----------------------------------------------------

        boxes = prediction["boxes"].cpu()

        scores = prediction["scores"].cpu()

        labels = prediction["labels"].cpu()


        detections = []


        for box, score, label in zip(
            boxes,
            scores,
            labels
        ):

            score = float(score)

            label = int(label)


            # Only BOX
            if label != 1:
                continue


            if score < CONFIDENCE_THRESHOLD:
                continue


            x1, y1, x2, y2 = box.tolist()


            # ------------------------------------------------
            # Convert from 800x800 back to camera resolution
            # ------------------------------------------------

            x1 = int(
                x1 * original_width / MODEL_SIZE
            )

            y1 = int(
                y1 * original_height / MODEL_SIZE
            )

            x2 = int(
                x2 * original_width / MODEL_SIZE
            )

            y2 = int(
                y2 * original_height / MODEL_SIZE
            )


            detections.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    score
                )
            )


        # ----------------------------------------------------
        # Update latest detections
        # ----------------------------------------------------

        with detection_lock:

            latest_detections = detections


# ============================================================
# START THREADS
# ============================================================

threading.Thread(
    target=camera_thread,
    daemon=True
).start()


threading.Thread(
    target=inference_thread,
    daemon=True
).start()


# ============================================================
# DISPLAY LOOP
# ============================================================

print("\n" + "=" * 65)
print("LIVE STREAM STARTED")
print("Press Q to quit")
print("=" * 65)


display_fps = 0

display_counter = 0

fps_start = time.time()


while True:

    # --------------------------------------------------------
    # GET NEWEST FRAME
    # --------------------------------------------------------

    with frame_lock:

        if latest_frame is None:

            frame = None

        else:

            frame = latest_frame.copy()


    if frame is None:

        time.sleep(0.005)

        continue


    display_counter += 1


    # --------------------------------------------------------
    # DISPLAY FPS
    # --------------------------------------------------------

    elapsed = time.time() - fps_start

    if elapsed >= 1.0:

        display_fps = display_counter / elapsed

        display_counter = 0

        fps_start = time.time()


    # --------------------------------------------------------
    # GET LATEST DETECTIONS
    # --------------------------------------------------------

    with detection_lock:

        detections = latest_detections.copy()


    # --------------------------------------------------------
    # DRAW BOXES
    # --------------------------------------------------------

    for (
        x1,
        y1,
        x2,
        y2,
        score
    ) in detections:


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        text = f"BOX {score:.2f}"


        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )


        cv2.rectangle(
            frame,
            (x1, max(0, y1 - th - 10)),
            (x1 + tw + 10, y1),
            (0, 255, 0),
            -1
        )


        cv2.putText(
            frame,
            text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )


    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (430, 125),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"Live FPS: {display_fps:.1f}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Detection FPS: {inference_fps:.2f}",
        (20, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Inference: {inference_ms:.0f} ms",
        (20, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Boxes: {len(detections)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display = cv2.resize(
        frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        ),
        interpolation=cv2.INTER_LINEAR
    )


    cv2.imshow(
        "Faster R-CNN - LOW LATENCY",
        display
    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# STOP
# ============================================================

camera_running = False

cv2.destroyAllWindows()

print("\nStream stopped.")











# import cv2
# import torch
# import time
# import os

# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# # ============================================================
# # CONFIGURATION
# # ============================================================

# VIDEO_PATH = "vid1.mp4"

# MODEL_PATH = "fasterrcnn_checkpoint.pth"

# CONFIDENCE_THRESHOLD = 0.50

# # Your model was trained using 800x800
# MODEL_SIZE = 800


# # ============================================================
# # DEVICE
# # ============================================================

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("=" * 60)
# print("FASTER R-CNN VIDEO TEST")
# print("=" * 60)

# print(f"PyTorch device: {device}")
# print(f"Video: {VIDEO_PATH}")
# print(f"Model: {MODEL_PATH}")
# print()


# # ============================================================
# # CHECK FILES
# # ============================================================

# if not os.path.exists(VIDEO_PATH):
#     print(f"ERROR: Video not found: {VIDEO_PATH}")
#     exit()

# if not os.path.exists(MODEL_PATH):
#     print(f"ERROR: Model not found: {MODEL_PATH}")
#     exit()


# # ============================================================
# # LOAD MODEL
# # ============================================================

# print("Loading Faster R-CNN model...")

# model = fasterrcnn_resnet50_fpn_v2(
#     weights=None,
#     min_size=MODEL_SIZE,
#     max_size=MODEL_SIZE
# )

# # 2 classes:
# # 0 = background
# # 1 = box

# in_features = model.roi_heads.box_predictor.cls_score.in_features

# model.roi_heads.box_predictor = FastRCNNPredictor(
#     in_features,
#     2
# )

# checkpoint = torch.load(
#     MODEL_PATH,
#     map_location=device,
#     weights_only=False
# )

# model.load_state_dict(checkpoint["model_state_dict"])

# model.to(device)
# model.eval()

# print("Model loaded successfully.")
# print()


# # ============================================================
# # OPEN VIDEO
# # ============================================================

# cap = cv2.VideoCapture(VIDEO_PATH)

# if not cap.isOpened():
#     print("ERROR: Could not open video.")
#     exit()


# video_fps = cap.get(cv2.CAP_PROP_FPS)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# video_duration = (
#     total_frames / video_fps
#     if video_fps > 0
#     else 0
# )

# print("=" * 60)
# print("VIDEO INFORMATION")
# print("=" * 60)

# print(f"Resolution     : {video_width} x {video_height}")
# print(f"Video FPS      : {video_fps:.2f}")
# print(f"Total frames   : {total_frames}")
# print(f"Duration       : {video_duration:.2f} seconds")

# print()


# # ============================================================
# # STATISTICS
# # ============================================================

# frame_count = 0
# detection_count = 0

# inference_times = []

# start_time = time.time()


# # ============================================================
# # MAIN LOOP
# # ============================================================

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     frame_count += 1

#     # --------------------------------------------------------
#     # Convert BGR -> RGB
#     # --------------------------------------------------------

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # --------------------------------------------------------
#     # Resize to model input size
#     # --------------------------------------------------------

#     resized = cv2.resize(
#         rgb,
#         (MODEL_SIZE, MODEL_SIZE)
#     )

#     # --------------------------------------------------------
#     # Convert to tensor
#     # --------------------------------------------------------

#     image_tensor = (
#         torch.from_numpy(resized)
#         .permute(2, 0, 1)
#         .float()
#         / 255.0
#     )

#     image_tensor = image_tensor.to(device)

#     # --------------------------------------------------------
#     # RUN INFERENCE
#     # --------------------------------------------------------

#     inference_start = time.time()

#     with torch.no_grad():

#         outputs = model([
#             image_tensor
#         ])

#     inference_time = time.time() - inference_start

#     inference_times.append(inference_time)

#     # --------------------------------------------------------
#     # GET DETECTIONS
#     # --------------------------------------------------------

#     boxes = outputs[0]["boxes"].detach().cpu()
#     scores = outputs[0]["scores"].detach().cpu()
#     labels = outputs[0]["labels"].detach().cpu()

#     frame_detections = 0

#     # --------------------------------------------------------
#     # DRAW DETECTIONS
#     # --------------------------------------------------------

#     for box, score, label in zip(
#         boxes,
#         scores,
#         labels
#     ):

#         score = float(score)

#         if score < CONFIDENCE_THRESHOLD:
#             continue

#         if int(label) != 1:
#             continue

#         frame_detections += 1
#         detection_count += 1

#         x1, y1, x2, y2 = box.tolist()

#         # Convert model coordinates back to original frame
#         x1 = int(x1 * video_width / MODEL_SIZE)
#         x2 = int(x2 * video_width / MODEL_SIZE)

#         y1 = int(y1 * video_height / MODEL_SIZE)
#         y2 = int(y2 * video_height / MODEL_SIZE)

#         cv2.rectangle(
#             frame,
#             (x1, y1),
#             (x2, y2),
#             (0, 255, 0),
#             3
#         )

#         cv2.putText(
#             frame,
#             f"BOX {score:.2f}",
#             (x1, max(y1 - 10, 30)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 0),
#             2
#         )

#     # --------------------------------------------------------
#     # CURRENT PERFORMANCE
#     # --------------------------------------------------------

#     elapsed = time.time() - start_time

#     processing_fps = (
#         frame_count / elapsed
#         if elapsed > 0
#         else 0
#     )

#     current_inference_fps = (
#         1.0 / inference_time
#         if inference_time > 0
#         else 0
#     )

#     # --------------------------------------------------------
#     # DISPLAY INFORMATION
#     # --------------------------------------------------------

#     cv2.putText(
#         frame,
#         f"Video FPS: {video_fps:.1f}",
#         (20, 35),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 255),
#         2
#     )

#     cv2.putText(
#         frame,
#         f"Inference FPS: {current_inference_fps:.2f}",
#         (20, 70),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 255),
#         2
#     )

#     cv2.putText(
#         frame,
#         f"Processing FPS: {processing_fps:.2f}",
#         (20, 105),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 255),
#         2
#     )

#     cv2.putText(
#         frame,
#         f"Boxes: {frame_detections}",
#         (20, 140),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 255),
#         2
#     )

#     cv2.putText(
#         frame,
#         f"Frame: {frame_count}/{total_frames}",
#         (20, 175),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 255),
#         2
#     )

#     # --------------------------------------------------------
#     # SHOW FRAME
#     # --------------------------------------------------------

#     display_frame = cv2.resize(
#         frame,
#         (1280, 720)
#     )

#     cv2.imshow(
#         "Faster R-CNN Video Test",
#         display_frame
#     )

#     # Press Q to quit
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord("q"):
#         break


# # ============================================================
# # FINAL STATISTICS
# # ============================================================

# cap.release()
# cv2.destroyAllWindows()

# total_time = time.time() - start_time

# average_inference_time = (
#     sum(inference_times) / len(inference_times)
#     if inference_times
#     else 0
# )

# average_inference_fps = (
#     1.0 / average_inference_time
#     if average_inference_time > 0
#     else 0
# )

# overall_fps = (
#     frame_count / total_time
#     if total_time > 0
#     else 0
# )


# print()
# print("=" * 60)
# print("FINAL RESULTS")
# print("=" * 60)

# print(f"Video FPS              : {video_fps:.2f}")
# print(f"Frames processed       : {frame_count}")
# print(f"Total detections       : {detection_count}")
# print(f"Total processing time  : {total_time:.2f} sec")

# print()
# print(f"Average inference time : {average_inference_time * 1000:.2f} ms")
# print(f"Average inference FPS  : {average_inference_fps:.2f}")
# print(f"Overall processing FPS : {overall_fps:.2f}")

# print("=" * 60)
# print("TEST FINISHED")
# print("=" * 60)