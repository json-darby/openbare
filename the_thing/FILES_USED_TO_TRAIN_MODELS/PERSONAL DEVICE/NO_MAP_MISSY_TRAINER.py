"""
Created on Sat Apr 19 00:43:06 2025

@author: I_NEE
"""

'''
  ░▒▓██████▓▒░ ░▒▓███████▓▒░ ░▒▓████████▓▒░░▒▓███████▓▒░ ░▒▓███████▓▒░  ░▒▓██████▓▒░ ░▒▓███████▓▒░ ░▒▓████████▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░       ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░       ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░  ░▒▓█▓▒░░▒▓█▓▒░░▒▓███████▓▒░ ░▒▓████████▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░   
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░       ░▒▓█▓▒░       ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░       ░▒▓█▓▒░       ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░        
  ░▒▓██████▓▒░ ░▒▓█▓▒░       ░▒▓████████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓███████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓████████▓▒░
'''

# pairs = load_pairs_from_text(ORIGINAL_TXT, OCCLUDED_TXT, num_pairs=100)

import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import cv2

##############################
# CONFIGURATION
##############################
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 250
LEARNING_RATE = 2e-4  # Learning rate

NUMBER_OF_PAIRS = None  # None for all

# Loss weighting parameters:
ADVERSARIAL_WEIGHT = 5e-2      # Weight for adversarial loss
LAMBDA_PERCEPTUAL_LOSS_WEIGHT = 0.3   # Weight for perceptual loss
EDGE_LOSS_WEIGHT = 0.05        # Penalty for errors outside the occlusion area
LAMBDA_ADVANCED_LOSS_WEIGHT = 0.1     # Additional hyperparameter for advanced loss term

# Choose adversarial loss type: "hinge", "lsgan", or "bce"
ADVERSARIAL_LOSS_TYPE = "lsgan"

# Paths to the text files containing image paths (one per line)
ORIGINAL_TEXT_FILE = "original.txt"
OCCLUDED_TEXT_FILE = "occluded.txt"

##############################
# UTILITY FUNCTIONS
##############################
def load_image_for_training(image_path):
    """Loads an image from file, resises it to the specified dimensions, and normalises to [0,1]."""
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return image / 255.0

def compute_difference_mask(original_image, occluded_image, threshold=0.05):
    """
    Computes a binary difference mask from the absolute difference between the original and occluded images.
    Returns a tensor of shape (IMAGE_HEIGHT, IMAGE_WIDTH, 1) with values of 0 or 1.
    """
    difference = tf.abs(original_image - occluded_image)
    grey_difference = tf.image.rgb_to_grayscale(difference)
    mask = tf.cast(grey_difference > threshold, tf.float32)
    return mask

def load_pairs_from_text(original_text_file, occluded_text_file, num_pairs=None):
    """Loads the file paths from the provided text files and returns a list of (original, occluded) pairs."""
    with open(original_text_file, "r") as f:
        original_paths = [line.strip() for line in f if line.strip()]
    with open(occluded_text_file, "r") as f:
        occluded_paths = [line.strip() for line in f if line.strip()]
    pair_count = min(len(original_paths), len(occluded_paths))
    if num_pairs is not None:
        pair_count = min(pair_count, num_pairs)
    pairs = list(zip(original_paths[:pair_count], occluded_paths[:pair_count]))
    print(f"Loaded {len(pairs)} pairs from text files.")
    return pairs

def sample_generator(pairs):
    """
    For each pair, loads the images and computes the difference mask.
    Yields a tuple (input_four_channel, target) where:
      - input_four_channel is a 4-channel tensor obtained by concatenating the occluded image (3 channels)
        with the difference mask (1 channel),
      - target is the original image (3 channels).
    """
    for original_path, occluded_path in pairs:
        original_image = load_image_for_training(original_path)
        occluded_image = load_image_for_training(occluded_path)
        difference_mask = compute_difference_mask(original_image, occluded_image, threshold=0.05)
        input_four_channel = tf.concat([occluded_image, difference_mask], axis=-1)
        yield input_four_channel, original_image

##############################
# DATA AUGMENTATION FUNCTION (missy_elliott)
##############################
def missy_elliott(
    dataset,
    double_dataset=True,
    flip=True,
    brightness=True,
    brightness_delta=0.1,
    contrast=True,
    contrast_lower=0.7,
    contrast_upper=1.3,
    saturation=True,
    saturation_lower=0.8,
    saturation_upper=1.2,
    hue=True,
    hue_delta=0.05
):
    """
    Data augmentation function.
    This function applies random horizontal flips and colour jitters (brightness, contrast, saturation, and hue adjustments)
    to the occluded image portion. If double_dataset is True, an augmented version is concatenated with the original
    dataset, effectively doubling the number of training samples.
    """
    def augment_sample(input_four_channel, target):
        # Separate the occluded image (first 3 channels) from the mask (4th channel)
        occluded_image = input_four_channel[..., :3]
        mask_channel = input_four_channel[..., 3:]
        
        if flip:
            coin = tf.random.uniform(())
            occluded_image = tf.cond(coin < 0.5,
                                     lambda: tf.image.flip_left_right(occluded_image),
                                     lambda: occluded_image)
            target = tf.cond(coin < 0.5,
                             lambda: tf.image.flip_left_right(target),
                             lambda: target)
            mask_channel = tf.cond(coin < 0.5,
                                   lambda: tf.image.flip_left_right(mask_channel),
                                   lambda: mask_channel)
        if brightness:
            occluded_image = tf.image.random_brightness(occluded_image, max_delta=brightness_delta)
        if contrast:
            occluded_image = tf.image.random_contrast(occluded_image, contrast_lower, contrast_upper)
        if saturation:
            occluded_image = tf.image.random_saturation(occluded_image, saturation_lower, saturation_upper)
        if hue:
            occluded_image = tf.image.random_hue(occluded_image, hue_delta)
        
        augmented_input = tf.concat([occluded_image, mask_channel], axis=-1)
        return augmented_input, target

    augmented_dataset = dataset.map(augment_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if double_dataset:
        dataset = dataset.concatenate(augmented_dataset)
        print("Dataset has been effectively doubled via data augmentation!")
        return dataset
    else:
        return augmented_dataset

##############################
# MODEL DEFINITIONS
##############################
def conv_block(x, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, activation='relu', padding='same')(x)
    return x

def residual_block(x, num_filters):
    """A simple residual block: adds a shortcut connection and applies ReLU activation."""
    shortcut = tf.keras.layers.Conv2D(num_filters, 1, padding='same')(x)
    conv = conv_block(x, num_filters)
    output = tf.keras.layers.Add()([conv, shortcut])
    output = tf.keras.layers.Activation('relu')(output)
    return output

def attention_gate(x, gating, inter_channels):
    """
    Computes an attention mask using the gating signal to modulate the encoder features.
    """
    theta_x = tf.keras.layers.Conv2D(inter_channels, 1, strides=1, padding='same')(x)
    phi_g = tf.keras.layers.Conv2D(inter_channels, 1, strides=1, padding='same')(gating)
    added = tf.keras.layers.Add()([theta_x, phi_g])
    activated = tf.keras.layers.Activation('relu')(added)
    psi = tf.keras.layers.Conv2D(1, 1, strides=1, padding='same')(activated)
    sigmoid = tf.keras.layers.Activation('sigmoid')(psi)
    output = tf.keras.layers.Multiply()([x, sigmoid])
    return output

def build_generator(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 4)):
    """
    Builds a U-Net–like generator with residual blocks and attention gates.
    The input is a 4-channel tensor: 3 channels for the occluded image and 1 channel for the difference mask.
    """
    inputs = tf.keras.Input(shape=input_shape)
    # Encoder path:
    conv1 = conv_block(inputs, 64)
    pool1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)
    
    conv2 = conv_block(pool1, 128)
    pool2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)
    
    conv3 = conv_block(pool2, 256)
    pool3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)
    
    conv4 = conv_block(pool3, 512)
    bottleneck = residual_block(conv4, 512)
    
    # Decoder path with attention skip connections:
    up3 = tf.keras.layers.UpSampling2D((2,2))(bottleneck)
    attn3 = attention_gate(conv3, up3, inter_channels=128)
    concat3 = tf.keras.layers.Concatenate()([up3, attn3])
    conv5 = conv_block(concat3, 256)
    
    up2 = tf.keras.layers.UpSampling2D((2,2))(conv5)
    attn2 = attention_gate(conv2, up2, inter_channels=64)
    concat2 = tf.keras.layers.Concatenate()([up2, attn2])
    conv6 = conv_block(concat2, 128)
    
    up1 = tf.keras.layers.UpSampling2D((2,2))(conv6)
    attn1 = attention_gate(conv1, up1, inter_channels=32)
    concat1 = tf.keras.layers.Concatenate()([up1, attn1])
    conv7 = conv_block(concat1, 64)
    
    outputs = tf.keras.layers.Conv2D(3, 1, activation='sigmoid')(conv7)
    model = tf.keras.Model(inputs, outputs)
    return model

def build_discriminator(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)):
    """
    Builds a PatchGAN discriminator.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    
    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    
    x = tf.keras.layers.Conv2D(512, 4, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    
    outputs = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

##############################
# VGG FEATURE EXTRACTOR FOR PERCEPTUAL LOSS
##############################
vgg_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
feature_extractor = tf.keras.Model(inputs=vgg_model.input,
                                   outputs=vgg_model.get_layer('block3_conv3').output)
feature_extractor.trainable = False

def perceptual_loss(fake, real):
    fake_scaled = fake * 255.0
    real_scaled = real * 255.0
    fake_preprocessed = tf.keras.applications.vgg19.preprocess_input(fake_scaled)
    real_preprocessed = tf.keras.applications.vgg19.preprocess_input(real_scaled)
    fake_features = feature_extractor(fake_preprocessed)
    real_features = feature_extractor(real_preprocessed)
    return tf.reduce_mean(tf.abs(fake_features - real_features))

##############################
# DATA PIPELINE SETUP
##############################
pairs = load_pairs_from_text(ORIGINAL_TEXT_FILE, OCCLUDED_TEXT_FILE, NUMBER_OF_PAIRS)
output_signature = (
    tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=tf.float32),
    tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32)
)
dataset = tf.data.Dataset.from_generator(lambda: sample_generator(pairs),
                                           output_signature=output_signature)
dataset = dataset.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Apply data augmentation using missy_elliott.
dataset = missy_elliott(dataset, double_dataset=True, flip=True, brightness=True, contrast=True, saturation=True, hue=True)

##############################
# BUILD MODELS
##############################
# We are using a 4-channel input.
generator = build_generator(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 4))
discriminator = build_discriminator()

##############################
# LOSSES AND OPTIMISERS
##############################
generator_optimiser = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
discriminator_optimiser = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)

##############################
# GAN TRAINING LOOP WITH EXTRA LOSS TERMS
##############################
@tf.function
def train_step(input_tensor, target):
    adv_weight = tf.constant(ADVERSARIAL_WEIGHT, dtype=tf.float32)
    lambda_perceptual = tf.constant(LAMBDA_PERCEPTUAL_LOSS_WEIGHT, dtype=tf.float32)
    lambda_edge = tf.constant(EDGE_LOSS_WEIGHT, dtype=tf.float32)
    lambda_advanced = tf.constant(LAMBDA_ADVANCED_LOSS_WEIGHT, dtype=tf.float32)
    
    # Update discriminator:
    with tf.GradientTape() as disc_tape:
        fake = generator(input_tensor, training=True)
        disc_real = discriminator(target, training=True)
        disc_fake = discriminator(fake, training=True)
        if ADVERSARIAL_LOSS_TYPE == "hinge":
            disc_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - disc_real))
            disc_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + disc_fake))
            disc_loss = disc_loss_real + disc_loss_fake
        elif ADVERSARIAL_LOSS_TYPE == "lsgan":
            disc_loss_real = tf.reduce_mean(tf.square(disc_real - 1))
            disc_loss_fake = tf.reduce_mean(tf.square(disc_fake))
            disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
        else:
            bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            disc_loss_real = bce_loss(tf.ones_like(disc_real), disc_real)
            disc_loss_fake = bce_loss(tf.zeros_like(disc_fake), disc_fake)
            disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimiser.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    # Compute reconstruction loss using the difference mask (channel index 3).
    mask = tf.expand_dims(input_tensor[..., 3], axis=-1)
    generated = generator(input_tensor, training=True)
    mask_loss = tf.reduce_mean(mask * tf.abs(target - generated))
    non_mask_loss = tf.reduce_mean((1 - mask) * tf.abs(target - generated))
    reconstruction_loss = mask_loss + lambda_edge * non_mask_loss
    
    with tf.GradientTape() as gen_tape:
        fake = generator(input_tensor, training=True)
        disc_fake = discriminator(fake, training=True)
        if ADVERSARIAL_LOSS_TYPE == "hinge":
            adversarial_loss = -tf.reduce_mean(disc_fake)
        elif ADVERSARIAL_LOSS_TYPE == "lsgan":
            adversarial_loss = tf.reduce_mean(tf.square(disc_fake - 1))
        else:
            bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            adversarial_loss = bce_loss(tf.ones_like(disc_fake), disc_fake)
        
        perceptualLoss = perceptual_loss(fake, target)
        generator_loss = reconstruction_loss + (adv_weight + lambda_advanced) * adversarial_loss + lambda_perceptual * perceptualLoss
    gen_gradients = gen_tape.gradient(generator_loss, generator.trainable_variables)
    generator_optimiser.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    
    return disc_loss, generator_loss

##############################
# TRAINING LOOP WITH VISUALISATION
##############################
loss_history = {"disc_loss": [], "gen_loss": []}
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    total_disc_loss = 0.0
    total_gen_loss = 0.0
    batch_count = 0
    
    for input_tensor, target in dataset:
        d_loss, g_loss = train_step(input_tensor, target)
        total_disc_loss += d_loss
        total_gen_loss += g_loss
        batch_count += 1
    
    avg_disc_loss = total_disc_loss / batch_count
    avg_gen_loss = total_gen_loss / batch_count
    loss_history["disc_loss"].append(avg_disc_loss)
    loss_history["gen_loss"].append(avg_gen_loss)
    print(f"  Discriminator Loss: {avg_disc_loss:.4f}, Generator Loss: {avg_gen_loss:.4f}")
    
    # Visualise one batch (first sample) from the dataset.
    for input_tensor, target in dataset.take(1):
        fake_output = generator(input_tensor, training=False)
        break
    occluded_image = input_tensor[0, :, :, :3].numpy()
    difference_mask = input_tensor[0, :, :, 3].numpy()  # Single-channel difference mask
    real_image = target[0].numpy()
    fake_image = fake_output[0].numpy()
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(occluded_image)
    axs[0].set_title("Occluded Input")
    axs[0].axis("off")
    axs[1].imshow(difference_mask, cmap="gray")
    axs[1].set_title("Difference Mask")
    axs[1].axis("off")
    axs[2].imshow(real_image)
    axs[2].set_title("Real")
    axs[2].axis("off")
    axs[3].imshow(fake_image)
    axs[3].set_title("Fake")
    axs[3].axis("off")
    plt.suptitle(f"Epoch {epoch+1}")
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)
    plt.close(fig)

##############################
# SAVE THE GENERATOR MODEL
##############################
time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = f"garbage_occlusion_gan_{time_stamp}.h5"
generator.save(model_save_path)
print(f"Saved occlusion GAN model to {model_save_path}")

# Plot loss curves.
plt.figure(figsize=(8, 5))
plt.plot(loss_history["disc_loss"], label="Discriminator Loss")
plt.plot(loss_history["gen_loss"], label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
plt.show()
