
import numpy as np

def one_hot(cls_idx, CLASSES):
    vector = np.zeros(CLASSES, dtype = np.float32)
    vector[cls_idx] = 1.

    return vector