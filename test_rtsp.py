import cv2
import time

RTSP_URL = "rtsp://AI:Procon@1122@172.16.86.52/ch1/main/av_stream"

cap = cv2.VideoCapture(
    RTSP_URL,
    cv2.CAP_FFMPEG
)

if not cap.isOpened():
    print("ERROR: Could not open RTSP stream")
    exit()

print("RTSP connected")

frame_count = 0
failed_count = 0
start = time.time()

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        failed_count += 1

        print(
            f"FAILED FRAME #{failed_count}"
        )

        time.sleep(0.05)

        if failed_count >= 20:
            print("Too many failed frames.")
            break

        continue

    failed_count = 0
    frame_count += 1

    if frame_count % 30 == 0:

        elapsed = time.time() - start

        fps = frame_count / elapsed

        print(
            f"Frames: {frame_count} | "
            f"FPS: {fps:.2f} | "
            f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
        )

    display = cv2.resize(
        frame,
        (1280, 720)
    )

    cv2.imshow(
        "RTSP Test",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()