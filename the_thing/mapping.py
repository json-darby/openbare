import sys
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import cv2
import dlib
import matplotlib.pyplot as plt
import config
import mask_utilities

from mask_segmentation import face_masks_robo
from detectron2_mask_segmentation import face_masks_detectron

IMAGE_HEIGHT_PIXELS = 256
IMAGE_WIDTH_PIXELS = 256
# INPAINTING_MODEL_FILE_PATH = "inference models/mapping/occlusion_model_ep300.keras"  # 45

INPAINTING_MODEL_FILE_PATH = "inference models/mapping/all_mapping_occlusion_inpainting_model_final.keras"
# INPAINTING_MODEL_FILE_PATH = "inference models/mapping/hinge_5k_mapping_occlusion_model_ep300.keras"

RECONSTRUCTED_IMAGE_SAVE_FILE_PATH = "regenerated_images/reconstructed_output.png"
MINIMUM_SEGMENTATION_CONFIDENCE = 0.7

PRE_OCCLUDED_IMAGE_FILE_PATH = "C:/Users/I_NEE/Desktop/00018_surgical.jpg"  # Still only for testing

# Dlib landmark regions for mask
LANDMARK_REGIONS = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "mouth": list(range(48, 68)),
}
DETECTION_REGION_NAMES = list(LANDMARK_REGIONS.keys()) + ["background"]
NUM_REGIONS = len(DETECTION_REGION_NAMES)

REGION_COLOURS = {
    "jaw":            (255, 255, 255),
    "right_eyebrow":  (0, 255, 0),
    "left_eyebrow":   (0, 255, 0),
    "nose":           (255, 255, 0),
    "right_eye":      (255, 0, 0),
    "left_eye":       (255, 0, 0),
    "mouth":          (0, 0, 255),
    "background":     (64, 64, 64),
}
OVERLAY_ALPHA = 0.5

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def load_and_preprocess_image(image_file_path):
    raw_data = tf.io.read_file(image_file_path)
    decoded = tf.image.decode_jpeg(raw_data, channels=3)
    resized = tf.image.resize(decoded, [IMAGE_HEIGHT_PIXELS, IMAGE_WIDTH_PIXELS])
    return resized / 255.0

def generate_face_segmentation_mask(image_file_path):
    # 1) Full‑res segmentation
    if config.FACE_MASK_METHOD == 1:
        raw_fullres, description, _ = face_masks_detectron(image_file_path, minimum_confidence=MINIMUM_SEGMENTATION_CONFIDENCE)
    else:
        raw_fullres, description, _ = face_masks_robo(image_file_path, minimum_confidence=MINIMUM_SEGMENTATION_CONFIDENCE)
    print(f"Face mask generation description: {description}")

    if hasattr(raw_fullres, 'numpy'):
        fullres_array = raw_fullres.numpy()
    else:
        fullres_array = np.array(raw_fullres)

    bin_mask = (fullres_array > 0.5).astype(np.uint8)

    if config.MASK_EXPAND_METHOD == 'distance':
        expanded_mask = mask_utilities.expand_mask_using_distance_transform(bin_mask)
    else:
        expanded_mask = mask_utilities.expand_mask_using_morphological_dilation(bin_mask)

    cleaned_mask = mask_utilities.clean_mask_using_morphology(expanded_mask)
    mask_256 = cv2.resize(cleaned_mask, (IMAGE_WIDTH_PIXELS, IMAGE_HEIGHT_PIXELS), interpolation=cv2.INTER_NEAREST)
    return mask_256[..., None].astype(np.float32)

def generate_region_masks(image_uint8):
    height, width = IMAGE_HEIGHT_PIXELS, IMAGE_WIDTH_PIXELS
    output = np.zeros((height, width, NUM_REGIONS), np.float32)
    detections = detector(image_uint8, 1)
    if detections:
        shape = predictor(image_uint8, detections[0])
        points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], np.int32)
        for idx, region in enumerate(LANDMARK_REGIONS):
            coords = points[LANDMARK_REGIONS[region]]
            polygon = cv2.convexHull(coords) if region == "nose" else coords
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            output[..., idx] = mask
    union_mask = np.clip(output[..., :-1].sum(-1), 0, 1)
    output[..., -1] = 1.0 - union_mask
    return output

def build_12_channel_input_from_occluded(image_file_path):
    occluded = load_and_preprocess_image(image_file_path)
    face_mask = generate_face_segmentation_mask(image_file_path)

    rgb8 = (occluded.numpy() * 255).astype(np.uint8)
    regions = generate_region_masks(rgb8)

    input_tensor = tf.concat([occluded, face_mask], axis=-1).numpy()
    input_12ch = np.concatenate([input_tensor, regions], axis=-1)
    return input_12ch[None], face_mask

def main(image_path):
    print("TensorFlow version:", tf.__version__)
    print("GPU(s) available:", len(tf.config.list_physical_devices('GPU')))

    occluded_image = load_and_preprocess_image(image_path)
    input_batch, face_mask = build_12_channel_input_from_occluded(image_path)

    try:
        model = tf.keras.models.load_model(
            INPAINTING_MODEL_FILE_PATH,
            compile=False,
            custom_objects={"InputLayer": tf.keras.layers.InputLayer}
        )
        print("Inpainting model loaded successfully.")
    except Exception as load_error:
        print(f"ERROR: Could not load inpainting model: {load_error}")
        sys.exit(1)

    try:
        output_tensor = model.predict(input_batch)[0]
        print(f"Generated inpainted output shape: {output_tensor.shape}")
    except Exception as inference_error:
        print(f"ERROR: Model inference failed: {inference_error}")
        sys.exit(1)

    clipped = tf.clip_by_value(output_tensor, 0.0, 1.0)
    output_uint8 = (clipped * 255).numpy().astype(np.uint8)
    output_bgr = cv2.cvtColor(output_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(RECONSTRUCTED_IMAGE_SAVE_FILE_PATH, output_bgr)
    print(f"Reconstructed image saved to: {RECONSTRUCTED_IMAGE_SAVE_FILE_PATH}")

    region_array = input_batch[0, ..., 4:]
    overlay = occluded_image.numpy().copy()
    colour_overlay = np.zeros_like(overlay)
    for idx, region in enumerate(DETECTION_REGION_NAMES):
        colour = np.array(REGION_COLOURS[region]) / 255.0
        mask = region_array[..., idx]
        colour_overlay[mask > 0] = colour
    combined_overlay = (1 - OVERLAY_ALPHA) * overlay + OVERLAY_ALPHA * colour_overlay

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(occluded_image.numpy())
    plt.title("Occluded RGB Input")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(face_mask.squeeze(), cmap="grey")
    plt.title("Face‑Segmentation Mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(combined_overlay)
    plt.title("Region Masks Overlay")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(clipped.numpy())
    plt.title("Inpainted Reconstruction")
    plt.axis("off")

    plt.tight_layout()
plt.show()

if __name__ == "__main__":
    main(PRE_OCCLUDED_IMAGE_FILE_PATH)
