"""
Merges the original training images with their augmented copies into a
single folder used for training.

Input:  train/{benign,malignant}
        augmented_images/{benign,malignant}
Output: augmented_train/{benign,malignant}
"""

import os
import shutil

# Paths
TRAIN_PATH = 'train'
AUGMENTED_PATH = 'augmented_images'
MERGED_PATH = 'augmented_train'

CLASSES = ['benign', 'malignant']


def merge_folder(source_folder):
    """Copy every class folder from the source into the merged folder."""
    for class_name in CLASSES:
        source_class = os.path.join(source_folder, class_name)
        target_class = os.path.join(MERGED_PATH, class_name)
        os.makedirs(target_class, exist_ok=True)

        files = [f for f in os.listdir(source_class) if not f.startswith('.')]
        print(f"Copying {len(files)} files from {source_class}")

        for filename in files:
            shutil.copy(
                os.path.join(source_class, filename),
                os.path.join(target_class, filename)
            )


if __name__ == '__main__':
    os.makedirs(MERGED_PATH, exist_ok=True)

    merge_folder(TRAIN_PATH)
    merge_folder(AUGMENTED_PATH)

    print("\nMerge complete:")
    for class_name in CLASSES:
        count = len(os.listdir(os.path.join(MERGED_PATH, class_name)))
        print(f"  {class_name}: {count} images")
