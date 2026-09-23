"""
Generates one augmented copy of each training image using random
geometric transformations.

Input:  data/train/{benign,malignant}
Output: data/augmented_images/{benign,malignant}
"""

import os
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, img_to_array, array_to_img, load_img
)

# Paths (relative to the repository root)
TRAIN_DATA_DIR = os.path.join('data', 'train')
AUGMENTED_DATA_DIR = os.path.join('data', 'augmented_images')

# Number of augmented copies to generate per original image
AUGMENTATIONS_PER_IMAGE = 1

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


def augment_class(class_name):
    """Generate augmented copies for every image in one class folder."""
    class_path = os.path.join(TRAIN_DATA_DIR, class_name)
    output_path = os.path.join(AUGMENTED_DATA_DIR, class_name)
    os.makedirs(output_path, exist_ok=True)

    images = [
        os.path.join(class_path, f)
        for f in os.listdir(class_path)
        if f.lower().endswith('.jpg')
    ]

    print(f"Augmenting {len(images)} images in '{class_name}'...")

    for img_path in images:
        x = img_to_array(load_img(img_path))

        for i in range(AUGMENTATIONS_PER_IMAGE):
            augmented = datagen.random_transform(x)
            filename = f'aug_{i}_{os.path.basename(img_path)}'
            array_to_img(augmented).save(os.path.join(output_path, filename))

    print(f"Saved augmented images to {output_path}")


if __name__ == '__main__':
    augment_class('malignant')
    augment_class('benign')
