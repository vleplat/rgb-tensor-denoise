# Module: patches.py
import numpy as np

def extract_clustered_patches(image, segments, patch_size=6):
    """Extract patches grouped by segment labels."""
    margin = patch_size // 2
    H, W, _ = image.shape
    Xs = []
    for segidx in sorted(np.unique(segments)):
        seg_array = []
        for (i, j) in zip(*np.where(segments == segidx)):
            if i < margin or i > H - margin - 1 or j < margin or j > W - margin - 1:
                continue
            patch = image[i - margin:i + margin, j - margin:j + margin]
            seg_array.append(np.expand_dims(patch, axis=-1))
        if seg_array:
            X = np.concatenate(seg_array, axis=-1)
            Xs.append((segidx, X))
    return Xs
