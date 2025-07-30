import base64
import requests
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf


def mask_detection_roboflow(image_path, minimum_confidence=0.5):
    mask_tf, description, best_detection = face_masks_robo(image_path, minimum_confidence)
    if best_detection and best_detection.get("confidence", 0) >= minimum_confidence:
        conf = best_detection["confidence"]
        return f"Mask detected with {conf*100:.0f}% confidence.", True
    return "No mask detected.", False

def meets_confidence(detection, minimum_confidence):
    return detection.get("confidence", 0) >= minimum_confidence

def is_sunglasses(detection):
    return detection.get("class", "").lower() == "sunglasses"

def process_segmentation(api_url, image_path, filter_func=None):
    # Process segmentation; returns (mask_tf, description, best_detection)
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(api_url, data=encoded_image, headers=headers)
    
    if "application/json" in response.headers.get("Content-Type", ""):
        result = response.json()
        print("Received JSON result:")
        print(json.dumps(result, indent=2))
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(image_rgb)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        predictions = result.get("predictions", [])
        if filter_func is not None:
            filtered_preds = [detection for detection in predictions if filter_func(detection)]
        else:
            filtered_preds = predictions
        
        if filtered_preds:
            best_detection = max(filtered_preds, key=lambda detection: detection.get("confidence", 0))
            # Default description; may be overridden later if needed.
            description = f"{best_detection['class'].capitalize()} ({best_detection['confidence']:.2f} confidence)"
        else:
            best_detection = None
            description = "No valid detection"
        
        for detection in filtered_preds:
            points = detection.get("points", [])
            if points:
                polygon_points = [(pt["x"], pt["y"]) for pt in points]
                patch = patches.Polygon(polygon_points, closed=True, linewidth=3,
                                        edgecolor="purple", facecolor="pink")
                patch.set_alpha(0.5)
                ax.add_patch(patch)
                pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
        
        plt.axis("off")
        plt.show()
        
        mask_tf = tf.convert_to_tensor(mask, dtype=tf.float32) / 255.0
        print("TensorFlow mask shape:", mask_tf.shape)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(mask, cmap="grey")
        plt.axis("off")
        plt.title("Mask")
        plt.show()
        plt.close(fig)

        return mask_tf, description, best_detection
    else:
        print("Response did not return JSON data.")
        return None, "No valid detection", None

def face_masks_robo(image_path, minimum_confidence=0.5):
    mask_api = "https://outline.roboflow.com/m666sk/1?api_key=dtPOWUN60iGi0OxQ1nkZ"
    def filter_face(detection):
        return meets_confidence(detection, minimum_confidence)
    
    mask_tf, _, best_detection = process_segmentation(mask_api, image_path, filter_face)
    
    # Override the description for face masks to display "Mask – XX% confidence"
    if best_detection:
        description = f"Mask – {best_detection['confidence'] * 100:.0f}% confidence"
    else:
        description = "No valid detection"
    
    return mask_tf, description, best_detection

# def eye_wear(image_path):
#     eyewear_api = "https://outline.roboflow.com/eyewear-prototype/1?api_key=your_api_key_here"
#     def filter_eyewear(detection):
#         return detection.get("class", "").lower() in ["glasses", "sunglasses"]
#     mask_tf, description, best_detection = process_segmentation(eyewear_api, image_path, filter_eyewear)
#     if best_detection:
#         if best_detection.get("class", "").lower() != "sunglasses":
#             description = f"Eyewear ({best_detection['confidence']:.2f} confidence) - not processing"
#     return mask_tf, description, best_detection

# mask_tf, description, detection = face_masks_robo("C:/Users/I_NEE/Desktop/00018_surgical.jpg", minimum_confidence=0.7)
# # print(description)

