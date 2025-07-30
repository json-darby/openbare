import os
import sys
import cv2
import dlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config
import mask_utilities
from mask_segmentation import face_masks_robo
from detectron2_mask_segmentation import face_masks_detectron

"""
Two‑stage face‑inpainting pipeline:
  • Stage 1:  four‑channel “no_mapping” model hallucinates a full face.
  • Stage 2:  twelve‑channel “mapping” model refines the result using
              region masks + expanded face‑segmentation from full‑res.
Produces three variants: default, bigger‑nose, and smaller‑mouth.
"""

IMAGE_HEIGHT = 256
IMAGE_WIDTH  = 256

NO_MAPPING_MODEL_PATH = "inference models/no_mapping_missy/model_epoch300_20250417-031956.h5"
MAPPING_MODEL_PATH = "inference models/mapping/hinge_5k_mapping_occlusion_model_ep300.keras"
# all_mapping_occlusion_inpainting_model_final.keras

OUTPUT_DIRECTORY      = "regenerated_images"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

MINIMUM_CONFIDENCE = 0.7
INPUT_IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/I_NEE/Desktop/28346_surgical.jpg"

no_mapping_model = tf.keras.models.load_model(NO_MAPPING_MODEL_PATH, compile=False,
                                              custom_objects={"InputLayer": tf.keras.layers.InputLayer})
mapping_model = tf.keras.models.load_model(MAPPING_MODEL_PATH,   compile=False,
                                              custom_objects={"InputLayer": tf.keras.layers.InputLayer})

face_detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Region index sets
REGIONS = [
    list(range(0, 17)),   # jaw
    list(range(17, 22)),  # right eyebrow
    list(range(22, 27)),  # left eyebrow
    list(range(27, 36)),  # nose
    list(range(36, 42)),  # right eye
    list(range(42, 48)),  # left eye
    list(range(48, 68)),  # mouth
]
NUM_REGIONS = len(REGIONS) + 1  # plus background


def load_and_resize_rgb(path):
    """Load an image and resize to network resolution (returns float32 RGB 0‑1)."""
    data = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    data = tf.image.resize(data, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return data / 255.0

def generate_face_segmentation_mask(path):
    """Full‑res segmentation → expansion/clean → down‑sample to 256×256."""
    if config.FACE_MASK_METHOD == 1:
        raw, desc, _ = face_masks_detectron(path, minimum_confidence=MINIMUM_CONFIDENCE)
    else:
        raw, desc, _ = face_masks_robo(path, minimum_confidence=MINIMUM_CONFIDENCE)
    print(f"Segmentation: {desc}")

    fullres = raw.numpy() if hasattr(raw, "numpy") else np.array(raw)
    binary  = (fullres > 0.5).astype(np.uint8)

    if config.MASK_EXPAND_METHOD == "distance":
        expanded = mask_utilities.expand_mask_using_distance_transform(binary)
    else:
        expanded = mask_utilities.expand_mask_using_morphological_dilation(binary)

    cleaned = mask_utilities.clean_mask_using_morphology(expanded)
    mask256 = cv2.resize(cleaned, (IMAGE_WIDTH, IMAGE_HEIGHT),
                         interpolation=cv2.INTER_NEAREST)[..., None]
    return mask256.astype(np.float32)

def adjust_nose(points, scale=0.8, shift=(0, 0)):
    indices = REGIONS[3]
    centre  = points[indices].mean(axis=0)
    points[indices] = ((points[indices] - centre) * scale + centre + shift).round()
    return points

def adjust_mouth(points, scale=1.2, shift=(0, 0)):
    indices = REGIONS[6]
    centre  = points[indices].mean(axis=0)
    points[indices] = ((points[indices] - centre) * scale + centre + shift).round()
    return points

def generate_region_masks(rgb_uint8, adjust_fn=None):
    """Return (H,W,NUM_REGIONS) float32 region masks for a 256‑sized face."""
    grey  = cv2.cvtColor(rgb_uint8, cv2.COLOR_BGR2GRAY)
    faces = face_detector(grey, 1)
    masks = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_REGIONS), np.float32)
    if not faces:
        return masks

    shape  = landmark_predict(grey, faces[0])
    points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)],
                      dtype=np.float32)
    if adjust_fn:
        points = adjust_fn(points)

    points = points.astype(np.int32)
    for idx, region_indices in enumerate(REGIONS):
        coords = points[region_indices]
        if coords.shape[0] < 3:
            continue
        if idx == 3:  # nose convex hull
            coords = cv2.convexHull(coords)
        mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
        cv2.fillPoly(mask, [coords], 1)
        masks[..., idx] = mask
    masks[..., -1] = 1.0 - np.clip(masks[..., :-1].sum(-1), 0, 1)
    return masks

def main(image_path):
    # Stage 1 – rough inpaint
    occluded_image = load_and_resize_rgb(image_path)
    face_mask_1    = generate_face_segmentation_mask(image_path)
    stage1_input   = tf.concat([occluded_image, face_mask_1], axis=-1)[None]
    stage1_output  = no_mapping_model.predict(stage1_input)[0]
    stage1_uint8   = (np.clip(stage1_output, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, "echo_stage1.png"),
                cv2.cvtColor(stage1_uint8, cv2.COLOR_RGB2BGR))

    # Prepare original occluded RGB for Stage 2 mapping
    occluded_rgb_256 = load_and_resize_rgb(image_path)

    variants = {
        "default":      lambda pts: adjust_mouth(adjust_nose(pts, scale=0.8), scale=1.2),
        "nose_bigger":  lambda pts: adjust_nose(pts, scale=1.2),
        "mouth_smaller":lambda pts: adjust_mouth(pts, scale=0.8),
    }

    results = {}
    for name, adjust_fn in variants.items():
        face_mask_2 = generate_face_segmentation_mask(image_path)
        region_masks = generate_region_masks(stage1_uint8, adjust_fn)

        twelve_channel_input = np.concatenate([
            occluded_rgb_256.numpy()[None],
            face_mask_2[None],
            region_masks[None]
        ], axis=-1)

        stage2_output = mapping_model.predict(twelve_channel_input)[0]
        final_uint8   = (np.clip(stage2_output, 0, 1) * 255).astype(np.uint8)
        out_path      = os.path.join(OUTPUT_DIRECTORY, f"echo_reconstructed_output_{name}.png")
        cv2.imwrite(out_path, cv2.cvtColor(final_uint8, cv2.COLOR_RGB2BGR))
        results[name] = final_uint8

    # Display variants side‑by‑side
    plt.figure(figsize=(12, 4))
    for index, (name, image_bgr) in enumerate(results.items(), start=1):
        plt.subplot(1, 3, index)
        plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        plt.title(name.replace("_", " ").title())
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(INPUT_IMAGE_PATH)
