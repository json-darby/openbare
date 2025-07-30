import cv2
import numpy as np
import config

''' 
Patch 1.1: Fixed mask-to-skin gap by expanding boundaries and sealing holes for a perfect face fit.
'''


# trying to keep the shape the same
def expand_mask_using_distance_transform(binary_mask):
    height, width = binary_mask.shape
    expansion_radius = int(max(height, width) * config.MASK_EXPAND_RATIO)
    inverted_mask = (binary_mask == 0).astype(np.uint8)
    distance_map = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
    expanded_mask = ((distance_map <= expansion_radius) | (binary_mask == 1)).astype(np.uint8)
    return expanded_mask


def expand_mask_using_morphological_dilation(binary_mask):
    height, width = binary_mask.shape
    expansion_radius = int(max(height, width) * config.MASK_EXPAND_RATIO)
    kernel_size = 2 * expansion_radius + 1
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(binary_mask, structuring_element)
    return dilated_mask


def clean_mask_using_morphology(binary_mask):
    close_kernel_size = config.MASK_CLOSE_KERNEL_SIZE
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, structuring_element)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, structuring_element)
    return opened_mask
