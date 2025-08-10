# Module: io.py
import numpy as np
import matplotlib.image as mpimg
from skimage.util import random_noise

def load_image(filepath):
    """Load and normalize image to [0, 1]"""
    img = mpimg.imread(filepath)
    return img/img.max()

def add_gaussian_noise(image, mean = 0, sigma=0.075):
    """Add Gaussian noise to an image."""
    noisy_image = image + np.random.normal(loc=mean, scale=sigma, size=image.shape)
    return np.clip(noisy_image, 0, 1)

