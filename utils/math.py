import numpy as np


def sign(x, threshold=1e-2):
    return np.where(x > threshold, 1, np.where(x < -threshold, -1, 0))
