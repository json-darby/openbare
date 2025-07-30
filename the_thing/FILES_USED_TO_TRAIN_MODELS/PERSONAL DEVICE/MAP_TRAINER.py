# ────────────────────────────────────────────────────────────────
#  Face‑in‑painting trainer – region one‑hot masks (from original)
# ────────────────────────────────────────────────────────────────
import os, cv2, dlib, datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────
IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
BATCH_SIZE                = 4
NUM_EPOCHS                = 50
LEARNING_RATE             = 2e-4
ADVERSARIAL_WEIGHT        = 5e-2
PERCEPTUAL_LOSS_WEIGHT    = 0.3
EDGE_LOSS_WEIGHT          = 0.05
ADVANCED_LOSS_WEIGHT      = 0.1
ADVERSARIAL_LOSS_TYPE     = "lsgan"          # "hinge" | "lsgan" | "bce"
PLOT_INTERVAL             = 5                # epochs

# local files
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
ORIGINAL_TEXT        = "original.txt"
OCCLUDED_TEXT        = "occluded.txt"
NUMBER_OF_PAIRS      = 20                     # None = all

# landmark groups (68‑pt dlib)
REGIONS = {
    "jaw"           : list(range(0 , 17)),
    "right_eyebrow" : list(range(17, 22)),
    "left_eyebrow"  : list(range(22, 27)),
    "nose"          : list(range(27, 36)),      # ridge + nostrils
    "right_eye"     : list(range(36, 42)),
    "left_eye"      : list(range(42, 48)),
    "mouth"         : list(range(48, 68)),      # outer & inner lips
}
REGION_NAMES  = list(REGIONS.keys()) + ["background"]
NUM_REGION_CH = len(REGION_NAMES)               # 8

# colours (B,G,R) for overlay
REGION_COLORS = {
    "jaw":            (255,255,255),
    "right_eyebrow":  (  0,255,  0),
    "left_eyebrow":   (  0,255,  0),
    "nose":           (255,255,  0),
    "right_eye":      (255,  0,  0),
    "left_eye":       (255,  0,  0),
    "mouth":          (  0,  0,255),
    "background":     ( 64, 64, 64),
}
OVERLAY_ALPHA = 0.5

# ── I/O ---------------------------------------------------------
def load_image(path: str) -> tf.Tensor:
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return img / 255.0

def diff_mask(orig, occ, thr=0.05):
    g = tf.image.rgb_to_grayscale(tf.abs(orig - occ))
    return tf.cast(g > thr, tf.float32)

def load_pairs(a_txt, b_txt, n=None):
    with open(a_txt) as f: A = [l.strip() for l in f if l.strip()]
    with open(b_txt) as f: B = [l.strip() for l in f if l.strip()]
    m = min(len(A), len(B)); m = m if n is None else min(m, n)
    pairs = list(zip(A[:m], B[:m]))
    print(f"Loaded {len(pairs)} pairs")
    return pairs

# generator yields occluded‑4ch, original‑rgb, original‑rgb
def sample_gen(pairs):
    for p_orig, p_occ in pairs:
        orig = load_image(p_orig)
        occ  = load_image(p_occ)
        dmask = diff_mask(orig, occ)
        yield tf.concat([occ, dmask], -1), orig, orig

# ── dlib one‑hot mask builder -----------------------------------
det  = dlib.get_frontal_face_detector()
pred = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

def make_region_masks(img_uint8: np.ndarray) -> np.ndarray:
    H, W = IMAGE_HEIGHT, IMAGE_WIDTH
    out  = np.zeros((H, W, NUM_REGION_CH), np.float32)

    dets = det(img_uint8, 1)
    if dets:
        shp = pred(img_uint8, dets[0])
        pts = np.array([[shp.part(i).x, shp.part(i).y] for i in range(68)],
                       np.int32)

        for k, name in enumerate(REGIONS):
            poly = cv2.convexHull(pts[REGIONS[name]])  # full outline
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [poly], 1)
            out[..., k] = mask

    union = np.clip(out[..., :-1].sum(-1), 0., 1.)
    out[..., -1] = 1.0 - union                         # background
    return out

def tf_add_regions(inp4, tgt_rgb, tgt):               # tgt_rgb == original
    rgb8  = tf.cast(tgt_rgb * 255.0, tf.uint8)        # build from **clean**
    masks = tf.numpy_function(make_region_masks, [rgb8], tf.float32)
    masks.set_shape((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_REGION_CH))
    return tf.concat([inp4, masks], -1), tgt          # (H,W,4+8) , (H,W,3)

# ── MODEL HELPERS ───────────────────────────────────────────────
def conv_blk(x, f):
    x = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    return x

def res_blk(x, f):
    skip = tf.keras.layers.Conv2D(f, 1, padding='same')(x)     # <─ fixed
    y    = conv_blk(x, f)
    return tf.keras.layers.Activation('relu')(y + skip)

def attn_gate(x, g, it):
    θ = tf.keras.layers.Conv2D(it, 1, padding='same')(x)
    φ = tf.keras.layers.Conv2D(it, 1, padding='same')(g)
    a = tf.keras.layers.Activation('relu')(θ + φ)
    ψ = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(a)
    return x * ψ

def build_generator():
    inp = tf.keras.Input((IMAGE_HEIGHT, IMAGE_WIDTH, 4 + NUM_REGION_CH))
    c1 = conv_blk(inp, 64); p1 = tf.keras.layers.MaxPool2D()(c1)
    c2 = conv_blk(p1, 128); p2 = tf.keras.layers.MaxPool2D()(c2)
    c3 = conv_blk(p2, 256); p3 = tf.keras.layers.MaxPool2D()(c3)
    b  = res_blk(p3, 512)
    u3 = tf.keras.layers.UpSampling2D()(b)
    c5 = conv_blk(tf.keras.layers.Concatenate()([u3, attn_gate(c3, u3, 128)]), 256)
    u2 = tf.keras.layers.UpSampling2D()(c5)
    c6 = conv_blk(tf.keras.layers.Concatenate()([u2, attn_gate(c2, u2, 64)]), 128)
    u1 = tf.keras.layers.UpSampling2D()(c6)
    c7 = conv_blk(tf.keras.layers.Concatenate()([u1, attn_gate(c1, u1, 32)]), 64)
    out = tf.keras.layers.Conv2D(3, 1, padding='same', activation='sigmoid')(c7)
    return tf.keras.Model(inp, out)

def build_discriminator():
    inp = tf.keras.Input((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    x   = tf.keras.layers.Conv2D(64, 4, 2, padding='same')(inp); x = tf.keras.layers.LeakyReLU(0.2)(x)
    for f in (128, 256):
        x = tf.keras.layers.Conv2D(f, 4, 2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    x   = tf.keras.layers.Conv2D(512, 4, 1, padding='same')(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.LeakyReLU(0.2)(x)
    out = tf.keras.layers.Conv2D(1, 4, 1, padding='same')(x)
    return tf.keras.Model(inp, out)

# perceptual
vgg  = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
feat = tf.keras.Model(vgg.input, vgg.get_layer('block3_conv3').output)
feat.trainable = False
def perceptual(fake, real):
    f = tf.keras.applications.vgg19.preprocess_input(fake * 255.)
    r = tf.keras.applications.vgg19.preprocess_input(real * 255.)
    return tf.reduce_mean(tf.abs(feat(f) - feat(r)))

# ── DATASET -----------------------------------------------------
pairs = load_pairs(ORIGINAL_TEXT, OCCLUDED_TEXT, NUMBER_OF_PAIRS)
ds = (tf.data.Dataset.from_generator(lambda: sample_gen(pairs),
      output_signature=(tf.TensorSpec((IMAGE_HEIGHT, IMAGE_WIDTH, 4), tf.float32),
                        tf.TensorSpec((IMAGE_HEIGHT, IMAGE_WIDTH, 3), tf.float32),
                        tf.TensorSpec((IMAGE_HEIGHT, IMAGE_WIDTH, 3), tf.float32)))
      .map(tf_add_regions, num_parallel_calls=tf.data.AUTOTUNE)
      .shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# ── OPTS, MODELS -----------------------------------------------
gen, disc = build_generator(), build_discriminator()
g_opt = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
d_opt = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)

@tf.function
def train_step(inp, tgt):
    aw, pw, ew, adw = [tf.constant(v, tf.float32) for v in
                       (ADVERSARIAL_WEIGHT, PERCEPTUAL_LOSS_WEIGHT,
                        EDGE_LOSS_WEIGHT, ADVANCED_LOSS_WEIGHT)]
    # --- discriminator ---
    with tf.GradientTape() as t:
        fake = gen(inp, training=True)
        r, f = disc(tgt, training=True), disc(fake, training=True)
        if ADVERSARIAL_LOSS_TYPE == 'lsgan':
            d_loss = 0.5 * (tf.reduce_mean((r - 1) ** 2) + tf.reduce_mean(f ** 2))
        elif ADVERSARIAL_LOSS_TYPE == 'hinge':
            d_loss = tf.reduce_mean(tf.nn.relu(1 - r)) + tf.reduce_mean(tf.nn.relu(1 + f))
        else:
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            d_loss = 0.5 * (bce(tf.ones_like(r), r) + bce(tf.zeros_like(f), f))
    d_opt.apply_gradients(zip(t.gradient(d_loss, disc.trainable_variables),
                              disc.trainable_variables))

    # --- generator ---
    mask = tf.expand_dims(inp[..., 3], -1)
    with tf.GradientTape() as t:
        fake = gen(inp, training=True)
        f    = disc(fake, training=True)
        if ADVERSARIAL_LOSS_TYPE == 'lsgan':
            adv = tf.reduce_mean((f - 1) ** 2)
        elif ADVERSARIAL_LOSS_TYPE == 'hinge':
            adv = -tf.reduce_mean(f)
        else:
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            adv = bce(tf.ones_like(f), f)
        rec  = tf.reduce_mean(mask * tf.abs(tgt - fake))
        nrec = tf.reduce_mean((1 - mask) * tf.abs(tgt - fake))
        g_loss = rec + ew * nrec + (aw + adw) * adv + pw * perceptual(fake, tgt)
    g_opt.apply_gradients(zip(t.gradient(g_loss, gen.trainable_variables),
                              gen.trainable_variables))
    return d_loss, g_loss

# ── TRAIN -------------------------------------------------------
hist = {'d': [], 'g': []}
for ep in range(1, NUM_EPOCHS+1):
    td = tg = 0.0; n = 0
    for inp, tgt in ds:                          # <-- only two items now!
        dl, gl = train_step(inp, tgt)
        td += dl; tg += gl; n += 1
    hist['d'].append(td/n); hist['g'].append(tg/n)
    print(f"Epoch {ep}/{NUM_EPOCHS}  D={td/n:.4f}  G={tg/n:.4f}")

    if ep % PLOT_INTERVAL == 0:
        inp, tgt_sample = next(iter(ds))         # <-- two items here too
        fake = gen(inp, training=False)[0].numpy()
        rgb  = (inp[0,...,:3].numpy() * 255).astype(np.uint8)
        overlay = rgb.copy()

        dets = det(rgb, 1)
        if dets:
            shp = pred(rgb, dets[0])
            pts = np.array([[shp.part(i).x, shp.part(i).y] for i in range(68)],
                           np.int32)
            for name in REGION_NAMES[:-1]:              # skip background
                poly = cv2.convexHull(pts[REGIONS[name]]) if name == "nose" else pts[REGIONS[name]]
                mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
                cv2.fillPoly(mask, [poly], 1)
                col = np.array(REGION_COLORS[name], np.uint8)
                overlay[mask == 1] = ((1 - OVERLAY_ALPHA) * overlay[mask == 1]
                                      + OVERLAY_ALPHA * col).astype(np.uint8)

        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        axs[0].imshow(inp[0, ..., :3]);          axs[0].set_title("Occluded");  axs[0].axis('off')
        axs[1].imshow(inp[0, ..., 3], cmap='gray'); axs[1].set_title("Diff‑mask"); axs[1].axis('off')
        axs[2].imshow(overlay);                  axs[2].set_title("Region overlay"); axs[2].axis('off')
        axs[3].imshow(tgt_sample[0]);            axs[3].set_title("Real"); axs[3].axis('off')
        axs[4].imshow(fake);                     axs[4].set_title("Fake"); axs[4].axis('off')
        plt.suptitle(f"Epoch {ep}")
        plt.tight_layout(); plt.show()

# ── SAVE & CURVES ----------------------------------------------
# to get a single-file Keras format
gen.save("occlusion_inpainting_model.keras")

# or to HDF5
gen.save("occlusion_inpainting_model.h5")


plt.figure(figsize=(8, 5))
plt.plot(hist['d'], label='D loss'); plt.plot(hist['g'], label='G loss')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training curves")
plt.show()
