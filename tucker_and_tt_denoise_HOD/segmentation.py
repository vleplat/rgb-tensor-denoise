# Module: segmentation.py
from skimage.segmentation import slic

def superpixel_segmentation(image, n_segments=10, compactness=20):
    """Apply SLIC superpixel segmentation to an image."""
    return slic(image, n_segments=n_segments, compactness=compactness)