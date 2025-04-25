import matplotlib.pyplot as plt
import numpy as np
from skimage import metrics
from skimage.segmentation import mark_boundaries

from tucker_and_tt_denoise_HOD.io import load_image, add_gaussian_noise
from tucker_and_tt_denoise_HOD.segmentation import superpixel_segmentation
from tucker_and_tt_denoise_HOD.patches import extract_clustered_patches
from tucker_and_tt_denoise_HOD.decomposition import decompose_and_denoise_patch
from tucker_and_tt_denoise_HOD.reconstruction import reconstruct_image_from_patches
from tucker_and_tt_denoise_HOD.metrics import compute_quality_metrics


def main(image_path="Datasets/parrot_small.jpg", sigma=0.075, method="tt_adaptive", reshape_for_tt=True, chosen_order = 6, n_segments=50, compactness=30, patch_size = 6):
    # Load and display image
    image = load_image(image_path)
    plt.figure()
    plt.imshow(image)
    plt.title("Original Image")
    plt.show()

    # Add noise
    image_noise = add_gaussian_noise(image, sigma=sigma)

    # Segment
    segments = superpixel_segmentation(image_noise, n_segments=n_segments, compactness = compactness)
    plt.figure()
    plt.imshow(mark_boundaries(image_noise, segments))
    plt.title("Superpixel Segmentation")
    plt.show()

    # Extract patches
    Xs = extract_clustered_patches(image_noise, segments, patch_size)

    # Denoise
    Xs_hat = []
    for segidx, X in Xs:
        print(f'--------------------------------------------------------------------')
        print(f"Denoising segment {segidx} with shape {X.shape}")
        X_hat, _, _ = decompose_and_denoise_patch(X, method=method, sigma=sigma, reshape_for_tt=reshape_for_tt, target_order=chosen_order)
        Xs_hat.append((segidx, X_hat))
        print(f'--------------------------------------------------------------------')

    # Reconstruct
    recon_image = reconstruct_image_from_patches(Xs_hat, segments, image.shape, patch_size)
    reconstructed_image = np.clip(recon_image, 0, 1)

    # Show result
    plt.figure()
    plt.imshow(reconstructed_image)
    plt.title("Reconstructed Image")
    plt.show()

    # Metrics
    psnr, ssim = compute_quality_metrics(image, reconstructed_image)
    print(f"PSNR = {psnr:.4f} dB, SSIM = {ssim:.4f}")


if __name__ == "__main__":
    main(reshape_for_tt=True, chosen_order=8 )
