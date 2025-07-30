import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mask_segmentation import face_masks_robo
from detectron2_mask_segmentation import face_masks_detectron

IMAGE_FOLDER = "test_images"
MIN_CONFIDENCE = 0.8
TARGET_CMAP_ROBOFLOW = "pink"
TARGET_CMAP_DETECTRON2 = "pink"

def display_image_with_overlays(image_path: str):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Roboflow
    mask_robo_tf, label_robo, _ = face_masks_robo(image_path, minimum_confidence=MIN_CONFIDENCE)
    mask_robo = mask_robo_tf.numpy() if mask_robo_tf is not None else None

    # Detectron2
    mask_det_tf, label_det, _ = face_masks_detectron(image_path, minimum_confidence=MIN_CONFIDENCE)
    mask_det = mask_det_tf.numpy() if mask_det_tf is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Roboflow overlay
    axes[1].imshow(image_rgb)
    if mask_robo is not None:
        axes[1].imshow(mask_robo, cmap=TARGET_CMAP_ROBOFLOW, alpha=0.5)
    axes[1].set_title(f"Roboflow\n{label_robo}")
    axes[1].axis("off")

    # Detectron2 overlay
    axes[2].imshow(image_rgb)
    if mask_det is not None:
        axes[2].imshow(mask_det, cmap=TARGET_CMAP_DETECTRON2, alpha=0.5)
    axes[2].set_title(f"Detectron2\n{label_det}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    for filename in sorted(os.listdir(IMAGE_FOLDER)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(IMAGE_FOLDER, filename)
            print(f"Processing {filename}...")
            display_image_with_overlays(path)

if __name__ == "__main__":
    main()