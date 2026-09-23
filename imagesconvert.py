import cv2
import os

# ============================================================
# SETTINGS
# ============================================================

# Your original images
INPUT_FOLDER = r"E:\datacollection\dataset\newfull"

# Cropped images will be saved here
OUTPUT_FOLDER = r"E:\datacollection\dataset\newsplit"


# ============================================================
# ROI COORDINATES
# ============================================================
# previous 969
LEFT_ROI = (1100, 456, 711, 837)

RIGHT_ROI = (2200, 510, 651, 891)



# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

LEFT_OUTPUT = os.path.join(OUTPUT_FOLDER, "left")
RIGHT_OUTPUT = os.path.join(OUTPUT_FOLDER, "right")

os.makedirs(LEFT_OUTPUT, exist_ok=True)
os.makedirs(RIGHT_OUTPUT, exist_ok=True)


# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# FIND IMAGES
# ============================================================

image_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]

image_files.sort()

print("==============================================")
print("             ROI IMAGE CROPPER")
print("==============================================")

print(f"Input folder : {INPUT_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Images found : {len(image_files)}")

print()
print("LEFT ROI :", LEFT_ROI)
print("RIGHT ROI:", RIGHT_ROI)
print()


# ============================================================
# PROCESS IMAGES
# ============================================================

processed = 0
failed = 0

for filename in image_files:

    input_path = os.path.join(INPUT_FOLDER, filename)

    # Read image
    image = cv2.imread(input_path)

    if image is None:
        print(f"❌ Could not read: {filename}")
        failed += 1
        continue

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # LEFT ROI
    # --------------------------------------------------------

    x, y, w, h = LEFT_ROI

    if (
        x < 0 or
        y < 0 or
        x + w > image_width or
        y + h > image_height
    ):
        print(f"❌ LEFT ROI outside image: {filename}")
        failed += 1
        continue

    left_crop = image[
        y:y + h,
        x:x + w
    ]


    # --------------------------------------------------------
    # RIGHT ROI
    # --------------------------------------------------------

    x, y, w, h = RIGHT_ROI

    if (
        x < 0 or
        y < 0 or
        x + w > image_width or
        y + h > image_height
    ):
        print(f"❌ RIGHT ROI outside image: {filename}")
        failed += 1
        continue

    right_crop = image[
        y:y + h,
        x:x + w
    ]


    # --------------------------------------------------------
    # SAVE CROPS
    # --------------------------------------------------------

    left_output_path = os.path.join(
        LEFT_OUTPUT,
        filename
    )

    right_output_path = os.path.join(
        RIGHT_OUTPUT,
        filename
    )

    cv2.imwrite(left_output_path, left_crop)
    cv2.imwrite(right_output_path, right_crop)

    processed += 1

    print(
        f"✅ {filename} → "
        f"LEFT {left_crop.shape[1]}x{left_crop.shape[0]} | "
        f"RIGHT {right_crop.shape[1]}x{right_crop.shape[0]}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("==============================================")
print("                  COMPLETE")
print("==============================================")

print(f"Total images : {len(image_files)}")
print(f"Processed    : {processed}")
print(f"Failed       : {failed}")

print()
print("LEFT crops:")
print(LEFT_OUTPUT)

print()
print("RIGHT crops:")
print(RIGHT_OUTPUT)