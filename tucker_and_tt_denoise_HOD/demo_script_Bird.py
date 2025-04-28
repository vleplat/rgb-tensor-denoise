import matplotlib.pyplot as plt
import numpy as np
from skimage import metrics
from skimage.segmentation import mark_boundaries
import os
import sys
from datetime import datetime
import time

from tucker_and_tt_denoise_HOD.io import load_image, add_gaussian_noise
from tucker_and_tt_denoise_HOD.segmentation import superpixel_segmentation
from tucker_and_tt_denoise_HOD.patches import extract_clustered_patches
from tucker_and_tt_denoise_HOD.decomposition import decompose_and_denoise_patch
from tucker_and_tt_denoise_HOD.reconstruction import reconstruct_image_from_patches
from tucker_and_tt_denoise_HOD.metrics import compute_quality_metrics


def main(image_path="Datasets/parrot_small.jpg", sigma=0.075, method="tt_adaptive", reshape_for_tt=True, chosen_order = 6, n_segments=50, compactness=30, patch_size = 6, save_figures=True):
    
    try:
        start_time = time.time()
        # Create output directory for figures
        output_dir = "output_figures"
        if save_figures:
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nSaving figures to: {output_dir}/")
        
        print("\nStarting image denoising process...")
        print(f"Parameters:")
        print(f"- Image: {image_path}")
        print(f"- Method: {method}")
        print(f"- Sigma: {sigma}")
        print(f"- Number of segments: {n_segments}")
        print(f"{'-'*80}\n")
        
        # Load and display image
        load_start = time.time()
        image = load_image(image_path)
        load_time = time.time() - load_start
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        if save_figures:
            plt.savefig(os.path.join(output_dir, 'original_image.png'), bbox_inches='tight', dpi=300)
        plt.show()

        # Add noise
        noise_start = time.time()
        image_noise = add_gaussian_noise(image, sigma=sigma)
        noise_time = time.time() - noise_start
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image_noise)
        plt.title("Noisy Image")
        plt.axis('off')
        if save_figures:
            plt.savefig(os.path.join(output_dir, 'noisy_image.png'), bbox_inches='tight', dpi=300)
        plt.show()

        # Segment
        segment_start = time.time()
        segments = superpixel_segmentation(image_noise, n_segments=n_segments, compactness=compactness)
        segment_time = time.time() - segment_start
        
        plt.figure(figsize=(10, 10))
        plt.imshow(mark_boundaries(image_noise, segments))
        plt.title("Superpixel Segmentation")
        plt.axis('off')
        if save_figures:
            plt.savefig(os.path.join(output_dir, 'segmentation.png'), bbox_inches='tight', dpi=300)
        plt.show()

        # Extract patches
        patch_start = time.time()
        Xs = extract_clustered_patches(image_noise, segments, patch_size)
        patch_time = time.time() - patch_start

        # Denoise using parallel processing at segment level
        denoise_start = time.time()
        Xs_hat = []
        for segidx, X in Xs:
            print(f'--------------------------------------------------------------------')
            print(f"Denoising segment {segidx} with shape {X.shape}")
            X_hat, _, _ = decompose_and_denoise_patch(X, method=method, sigma=sigma, reshape_for_tt=reshape_for_tt, target_order=chosen_order)
            Xs_hat.append((segidx, X_hat))
            print(f'--------------------------------------------------------------------')

        denoise_time = time.time() - denoise_start

            # Reconstruct
        recon_start = time.time()
        recon_image = reconstruct_image_from_patches(Xs_hat, segments, image.shape, patch_size)
        reconstructed_image = np.clip(recon_image, 0, 1)
        recon_time = time.time() - recon_start

        # Show result
        plt.figure(figsize=(10, 10))
        plt.imshow(reconstructed_image)
        plt.title("Reconstructed Image")
        plt.axis('off')
        if save_figures:
            plt.savefig(os.path.join(output_dir, 'reconstructed_image.png'), bbox_inches='tight', dpi=300)
        plt.show()

        # Create side-by-side comparison
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(image_noise)
        plt.title("Noisy")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(reconstructed_image)
        plt.title("Reconstructed")
        plt.axis('off')
        
        plt.tight_layout()
        if save_figures:
            plt.savefig(os.path.join(output_dir, 'comparison.png'), bbox_inches='tight', dpi=300)
        plt.show()

        # Metrics
        psnr, ssim = compute_quality_metrics(image, reconstructed_image)
        # print(f"PSNR = {psnr:.4f} dB, SSIM = {ssim:.4f}")
        # Print timing information
        total_time = time.time() - start_time
        print("\nProcessing Complete!")
        print(f"{'-'*80}")
        print("\nTiming Information:")
        print(f"Image loading: {load_time:.2f} seconds")
        print(f"Noise addition: {noise_time:.2f} seconds")
        print(f"Segmentation: {segment_time:.2f} seconds")
        print(f"Patch extraction: {patch_time:.2f} seconds")
        print(f"Denoising: {denoise_time:.2f} seconds")
        print(f"Reconstruction: {recon_time:.2f} seconds")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"\nQuality Metrics:")
        print(f"PSNR = {psnr:.4f} dB")
        print(f"SSIM = {ssim:.4f}")
        print(f"{'-'*80}\n")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main(reshape_for_tt=True, chosen_order=8 )
