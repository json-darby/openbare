import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def mask_detection_detectron(image_path, minimum_confidence=0.5):
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found.", False
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    scores = instances.scores.numpy()
    if len(scores) > 0:
        idx = int(np.argmax(scores))
        conf = float(scores[idx])
        if conf >= minimum_confidence:
            return f"Mask detected with {conf*100:.0f}% confidence.", True
    return "No mask detected.", False

def centre_expansion(mask_uint8, kernel_x=9, kernel_y=9):
    """
    Expand the mask from the centre using dilation.
    Default kernel size is (5, 5).
    """
    kernel = np.ones((kernel_x, kernel_y), np.uint8)
    return cv2.dilate(mask_uint8, kernel, iterations=1)


base_path = "m666sk.v1i.coco"
train_json = os.path.join(base_path, "train/_annotations.coco.json")
train_imgs = os.path.join(base_path, "train/images")

if "mask_train" not in DatasetCatalog.list():
    register_coco_instances("mask_train", {}, train_json, train_imgs)

with open(train_json) as f:
    train_data = json.load(f)
thing_classes = [cat["name"] for cat in train_data["categories"]]

MetadataCatalog.get("mask_train").set(thing_classes=thing_classes)
metadata = MetadataCatalog.get("mask_train")
print("mask_train thing_classes:", metadata.thing_classes)
# (from your JSON): ['objects', 'mask']


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TEST = ("mask_train",)

# Set the number of classes to match the training JSON.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
cfg.MODEL.WEIGHTS = os.path.join("mask_detectron2_modelV2", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"  # Model trained using an A100 GPU

predictor = DefaultPredictor(cfg)


def face_masks_detectron(image_path, minimum_confidence=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Test image not found at {image_path}")
    
    # Perform inference.
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    
    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()  # Confidence scores
    
    overlay = img.copy()
    alpha = 0.5
    fill_colour = np.array([203, 192, 255])  # Pink fill
    border_colour = (128, 0, 128)  # Purple border
    
    dilated_masks_list = []

    for i in range(len(masks)):
        mask = masks[i]
        mask_uint8 = (mask.astype(np.uint8)) * 255
        
        dilated_mask_uint8 = centre_expansion(mask_uint8)
        dilated_mask = dilated_mask_uint8 > 0
        dilated_masks_list.append(dilated_mask)
        
        overlay[dilated_mask] = (1 - alpha) * overlay[dilated_mask] + alpha * fill_colour
        
        contours, _ = cv2.findContours(dilated_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(overlay, [largest_contour], -1, border_colour, 2)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cX, cY = x + w // 2, y + h // 2
            score_text = f"{scores[i]:.2f}"
            cv2.putText(overlay, score_text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_colour, 2, cv2.LINE_AA)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Custom Mask R-CNN Inference Result")
    plt.show()
    
    combined_mask = np.zeros_like(dilated_masks_list[0], dtype=bool)
    for m in dilated_masks_list:
        combined_mask = np.logical_or(combined_mask, m)
    mask_tf = tf.convert_to_tensor(combined_mask)
    
    combined_mask_np = mask_tf.numpy().astype(np.uint8) * 255
    plt.figure(figsize=(15, 15))
    plt.imshow(combined_mask_np, cmap='grey')
    plt.axis("off")
    plt.title("Combined Predicted Mask (TensorFlow Tensor)")
    plt.show()
    
    if len(scores) > 0:
        idx = int(np.argmax(scores))
        best_detection = {
            "class": metadata.thing_classes[int(instances.pred_classes.numpy()[idx])] if len(instances.pred_classes) > 0 else "Mask",
            "confidence": float(scores[idx])
        }
        description = f"Mask – {best_detection['confidence'] * 100:.0f}% confidence"
    else:
        best_detection = None
        description = "No valid detection"
    
    return mask_tf, description, best_detection

# mask_tf, description, best_detection = face_masks_detectron("C:/Users/I_NEE/Desktop/00018_surgical.jpg", minimum_confidence=0.7)
# print(description)
