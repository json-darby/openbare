'''
import the whole module when trying to use 
mutating as opposed to binding!

OTHER MODELS AVAILABLE IN FOLDERS
MAYBE SOME OVERFITTING...

'''

INFERENCE_MODEL = 1  # This needs to be changed....

FACE_MASK_METHOD = 2  # 1 = Detectron2, 2 = Roboflow

MASK_EXPAND_METHOD = "distance"  # morphological


if FACE_MASK_METHOD == 1:
    MASK_EXPAND_RATIO = 0.0028  # Expansion radius - 1 = 100%
    MASK_CLOSE_KERNEL_SIZE = 31  # ODD ONLY
else:
    MASK_EXPAND_RATIO = 0.0044
    MASK_CLOSE_KERNEL_SIZE = 11  # ODD ONLY