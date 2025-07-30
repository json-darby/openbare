import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import config
import mask_utilities
from mask_segmentation import face_masks_robo
from detectron2_mask_segmentation import face_masks_detectron


MODEL_SOURCE = "inference models/no_mapping/all_no_mapping_model_epoch220_20250422-154917.h5"

OCCLUDED_IMAGE_PATH = "C:/Users/I_NEE/Desktop/00018_surgical.jpg"
OUTPUT_IMAGE_PATH = "regenerated_images/reconstructed_output.png"
IMAGE_HEIGHT = 256
IMAGE_WIDTH  = 256

def build_attention_unet():
    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        return x

    def residual_block(x, filters):
        skip = tf.keras.layers.Conv2D(filters, 1, padding='same')(x)
        y    = conv_block(x, filters)
        return tf.keras.layers.Activation('relu')(y + skip)

    def attention_gate(x, g, inter_channels):
        θ = tf.keras.layers.Conv2D(inter_channels, 1, padding='same')(x)
        φ = tf.keras.layers.Conv2D(inter_channels, 1, padding='same')(g)
        a = tf.keras.layers.Activation('relu')(θ + φ)
        ψ = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(a)
        return x * ψ

    inp = tf.keras.Input((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    # Encoding
    c1 = conv_block(inp, 64);   p1 = tf.keras.layers.MaxPool2D()(c1)
    c2 = conv_block(p1, 128);  p2 = tf.keras.layers.MaxPool2D()(c2)
    c3 = conv_block(p2, 256);  p3 = tf.keras.layers.MaxPool2D()(c3)
    # Bottleneck
    b  = residual_block(p3, 512)
    # Decoding with attention
    u3 = tf.keras.layers.UpSampling2D()(b)
    c5 = conv_block(tf.keras.layers.Concatenate()([u3, attention_gate(c3, u3, 128)]), 256)
    u2 = tf.keras.layers.UpSampling2D()(c5)
    c6 = conv_block(tf.keras.layers.Concatenate()([u2, attention_gate(c2, u2, 64)]), 128)
    u1 = tf.keras.layers.UpSampling2D()(c6)
    c7 = conv_block(tf.keras.layers.Concatenate()([u1, attention_gate(c1, u1, 32)]), 64)
    out = tf.keras.layers.Conv2D(3, 1, padding='same', activation='sigmoid')(c7)
    return tf.keras.Model(inp, out)

def locate_model():
    # If MODEL_SOURCE is a directory, find the one .h5 inside it.
    if os.path.isdir(MODEL_SOURCE):
        files = [f for f in os.listdir(MODEL_SOURCE) if f.lower().endswith('.h5')]
        if len(files) != 1:
            print("ERROR: Please keep exactly one .h5 in", MODEL_SOURCE)
            sys.exit(1)
        return os.path.join(MODEL_SOURCE, files[0])
    # Otherwise assume it's a file
    if os.path.isfile(MODEL_SOURCE) and MODEL_SOURCE.lower().endswith('.h5'):
        return MODEL_SOURCE
    print("ERROR: MODEL_SOURCE must be a .h5 file or folder containing exactly one .h5")
    sys.exit(1)

def load_inpainting_model(path):
    # Try full Keras load first
    try:
        model = tf.keras.models.load_model(
            path, compile=False,
            custom_objects={"InputLayer": tf.keras.layers.InputLayer}
        )
        print("Loaded full Keras model.")
        return model, "keras_full"
    except Exception:
        # Fallback to attention-UNet weights
        print("Not a standalone Keras model; assuming attention-UNet weights.")
        model = build_attention_unet()
        model.load_weights(path)
        return model, "attention_unet"

def load_image(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    return tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH]) / 255.0

def make_face_mask(path, confidence=0.7):
    if config.FACE_MASK_METHOD == 1:
        fullres, desc, _ = face_masks_detectron(path, minimum_confidence=confidence)
    else:
        fullres, desc, _ = face_masks_robo(path, minimum_confidence=confidence)
    print("Mask method:", desc)

    arr = fullres.numpy() if hasattr(fullres, "numpy") else np.array(fullres)
    binm = (arr > 0.5).astype(np.uint8)
    if config.MASK_EXPAND_METHOD == 'distance':
        binm = mask_utilities.expand_mask_using_distance_transform(binm)
    else:
        binm = mask_utilities.expand_mask_using_morphological_dilation(binm)
    binm = mask_utilities.clean_mask_using_morphology(binm)
    mask256 = cv2.resize(binm, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return tf.convert_to_tensor(mask256[..., None], dtype=tf.float32)

def main(image_path):
    model_path = locate_model()
    model, kind = load_inpainting_model(model_path)
    # print(f"Using model type: {kind}")

    img  = load_image(image_path)
    mask = make_face_mask(image_path)
    inp  = tf.concat([img, mask], axis=-1)[None, ...]

    if kind == "keras_full":
        out = model.predict(inp)[0]
    else:
        out = model(inp, training=False)[0]
    out = tf.clip_by_value(out, 0.0, 1.0).numpy()

    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    bgr = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_IMAGE_PATH, bgr)
    print("Saved inpainted image to", OUTPUT_IMAGE_PATH)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img.numpy());            plt.title("Input");  plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(mask[...,0].numpy(), cmap='gray'); plt.title("Mask");   plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(out);                    plt.title("Output"); plt.axis("off")
    plt.tight_layout(); plt.show()

if __name__=="__main__":
    main(OCCLUDED_IMAGE_PATH)