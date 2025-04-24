# Module: metrics.py
from skimage import metrics

def compute_quality_metrics(original, reconstructed):
    """Compute PSNR and SSIM for floating point images in [0, 1]."""
    psnr = metrics.peak_signal_noise_ratio(original, reconstructed, data_range=1.0)
    ssim = metrics.structural_similarity(original, reconstructed, data_range=1.0, channel_axis=2)
    return psnr, ssim