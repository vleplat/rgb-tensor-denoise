# Module: reconstruction.py
import numpy as np
from scipy import ndimage

def reconstruct_image_from_patches(Xs_hat, segments, image_shape, patch_size=6):
    """Reconstruct image from denoised patches using averaging."""
    H, W, C = image_shape
    margin = patch_size // 2
    recon_image = np.zeros((H, W, C))
    denom_matrix = np.zeros((H, W))

    for (segidx, X) in Xs_hat:
        K = 0
        for (i, j) in zip(*np.where(segments == segidx)):
            if i < margin or i > H - margin - 1 or j < margin or j > W - margin - 1:
                continue
            recon_image[i - margin:i + margin, j - margin:j + margin] += X[:, :, :, K]
            denom_matrix[i - margin:i + margin, j - margin:j + margin] += 1
            K += 1

    # Avoid division by zero
    denom_matrix = np.maximum(denom_matrix, 1)
    return recon_image / np.expand_dims(denom_matrix, -1)



def reconstruct_image_from_patches_median(Xs_hat, segments, image_shape, patch_size=6):
    """Reconstruct image using marginal median (per-channel) of overlapping patches."""
    H, W, C = image_shape
    margin = patch_size // 2
    
    # Initialize buffers to store all overlapping patches
    patch_buffer = np.zeros((H, W, C, 0))  # Will grow dynamically
    counts = np.zeros((H, W), dtype=int)    # Track overlaps

    # Accumulate all overlapping patches
    for segidx, X in Xs_hat:
        for (i, j) in zip(*np.where(segments == segidx)):
            if i < margin or i >= H - margin or j < margin or j >= W - margin:
                continue
            
            # Extract patch region and append to buffer
            patch = X[:, :, :, counts[i, j]]  # Shape: (h, w, C)
            if patch_buffer.shape[-1] <= counts[i, j]:
                # Dynamically expand buffer if needed
                patch_buffer = np.pad(patch_buffer, ((0,0),(0,0),(0,0),(0,1)), mode='constant')
            
            patch_buffer[i-margin:i+margin, j-margin:j+margin, :, counts[i,j]] = patch
            counts[i, j] += 1

    # Compute median (ignore zero-padded unused slots)
    recon_image = np.zeros((H, W, C))
    for i in range(H):
        for j in range(W):
            if counts[i, j] > 0:
                recon_image[i, j] = np.median(
                    patch_buffer[i, j, :, :counts[i, j]], 
                    axis=1
                )
    
    return recon_image