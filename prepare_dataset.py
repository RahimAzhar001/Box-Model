# import os
# import shutil
# import random

# # ============================================================
# # SETTINGS
# # ============================================================

# SOURCE_DIR = r"E:\datacollection\dataset\emptyfull"

# OUTPUT_DIR = r"E:\datacollection\dataset\emptyfull_split"

# TRAIN_RATIO = 0.70
# VAL_RATIO = 0.20
# TEST_RATIO = 0.10

# random.seed(42)

# # ============================================================
# # CREATE OUTPUT DIRECTORIES
# # ============================================================

# for split in ["train", "val", "test"]:
#     for class_name in ["box", "nobox"]:
#         os.makedirs(
#             os.path.join(
#                 OUTPUT_DIR,
#                 split,
#                 class_name
#             ),
#             exist_ok=True
#         )

# # ============================================================
# # COLLECT IMAGES FROM LEFT + RIGHT
# # ============================================================

# all_images = {
#     "box": [],
#     "nobox": []
# }

# for side in ["left", "right"]:

#     for class_name in ["box", "nobox"]:

#         folder = os.path.join(
#             SOURCE_DIR,
#             side,
#             class_name
#         )

#         if not os.path.exists(folder):
#             print(f"❌ Folder not found: {folder}")
#             continue

#         for filename in os.listdir(folder):

#             if filename.lower().endswith(
#                 (".jpg", ".jpeg", ".png", ".bmp", ".webp")
#             ):
#                 all_images[class_name].append(
#                     (
#                         os.path.join(folder, filename),
#                         side
#                     )
#                 )

# # ============================================================
# # SPLIT AND COPY
# # ============================================================

# for class_name in ["box", "nobox"]:

#     images = all_images[class_name]

#     random.shuffle(images)

#     total = len(images)

#     train_end = int(total * TRAIN_RATIO)

#     val_end = train_end + int(total * VAL_RATIO)

#     train_images = images[:train_end]
#     val_images = images[train_end:val_end]
#     test_images = images[val_end:]

#     splits = {
#         "train": train_images,
#         "val": val_images,
#         "test": test_images
#     }

#     print()
#     print(f"Class: {class_name}")
#     print(f"Total: {total}")
#     print(f"Train: {len(train_images)}")
#     print(f"Val:   {len(val_images)}")
#     print(f"Test:  {len(test_images)}")

#     # --------------------------------------------------------
#     # COPY FILES
#     # --------------------------------------------------------

#     for split_name, split_images in splits.items():

#         for index, (source_path, side) in enumerate(split_images):

#             original_filename = os.path.basename(source_path)

#             # Add side to filename to avoid collisions
#             new_filename = f"{side}_{original_filename}"

#             destination = os.path.join(
#                 OUTPUT_DIR,
#                 split_name,
#                 class_name,
#                 new_filename
#             )

#             shutil.copy2(
#                 source_path,
#                 destination
#             )

# print()
# print("==============================================")
# print("DATASET PREPARATION COMPLETE")
# print("==============================================")

# print(f"Dataset saved at:")
# print(OUTPUT_DIR)


import os
import random
import shutil

# =========================================================
# CONFIGURATION
# =========================================================

SOURCE_DIR = r"E:\datacollection\dataset\emptyfull"
OUTPUT_DIR = r"E:\datacollection\dataset\updatedempty_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# =========================================================
# CHECK RATIOS
# =========================================================

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

random.seed(RANDOM_SEED)

# =========================================================
# SUPPORTED IMAGE FORMATS
# =========================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

classes = ["empty", "full"]

for split in ["train", "val", "test"]:
    for class_name in classes:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )

# =========================================================
# PROCESS EACH CLASS
# =========================================================

for class_name in classes:

    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    if not os.path.exists(source_class_dir):
        print(f"ERROR: Folder not found: {source_class_dir}")
        continue

    images = [
        f for f in os.listdir(source_class_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    print("\n======================================")
    print(f"Class: {class_name}")
    print("======================================")
    print(f"Total : {total}")
    print(f"Train : {len(train_images)}")
    print(f"Val   : {len(val_images)}")
    print(f"Test  : {len(test_images)}")

    # -----------------------------------------------------
    # COPY FILES
    # -----------------------------------------------------

    for filename in train_images:

        src = os.path.join(
            source_class_dir,
            filename
        )

        dst = os.path.join(
            OUTPUT_DIR,
            "train",
            class_name,
            filename
        )

        shutil.copy2(src, dst)

    for filename in val_images:

        src = os.path.join(
            source_class_dir,
            filename
        )

        dst = os.path.join(
            OUTPUT_DIR,
            "val",
            class_name,
            filename
        )

        shutil.copy2(src, dst)

    for filename in test_images:

        src = os.path.join(
            source_class_dir,
            filename
        )

        dst = os.path.join(
            OUTPUT_DIR,
            "test",
            class_name,
            filename
        )

        shutil.copy2(src, dst)

# =========================================================
# DONE
# =========================================================

print("\n======================================")
print("DATASET SPLIT COMPLETE")
print("======================================")

print(f"\nDataset created at:")
print(os.path.abspath(OUTPUT_DIR))