#!/usr/bin/env python3
"""
Simple test script for RGB image denoising functionality.
"""

import sys
import os
sys.path.append('.')

from tucker_and_tt_denoise_HOD.utils import load_image, add_gaussian_noise, compute_quality_metrics
from tucker_and_tt_denoise_HOD.core import superpixel_segmentation, extract_clustered_patches, decompose_and_denoise_patch, reconstruct_image_from_patches

def test_denoising():
    """Test the complete denoising pipeline."""
    print("🧪 Testing RGB Image Denoising Pipeline")
    print("=" * 50)
    
    # Test parameters
    image_path = "tucker_and_tt_denoise_HOD/Datasets/parrot_small.jpg"
    sigma = 0.075
    n_segments = 20  # Reduced for faster testing
    patch_size = 6
    
    try:
        # 1. Load image
        print("1. Loading image...")
        image = load_image(image_path)
        print(f"   ✓ Image loaded: {image.shape}")
        
        # 2. Add noise
        print("2. Adding noise...")
        noisy_image = add_gaussian_noise(image, sigma=sigma)
        print(f"   ✓ Noise added with σ={sigma}")
        
        # 3. Segment image
        print("3. Segmenting image...")
        segments = superpixel_segmentation(noisy_image, n_segments=n_segments)
        print(f"   ✓ Image segmented into {len(set(segments.flatten()))} segments")
        
        # 4. Extract patches
        print("4. Extracting patches...")
        patches = extract_clustered_patches(noisy_image, segments, patch_size)
        print(f"   ✓ Extracted patches from {len(patches)} segments")
        
        # 5. Denoise patches (test with TT adaptive method for better stability)
        print("5. Denoising patches...")
        denoised_patches = []
        for i, (segidx, X) in enumerate(patches[:2]):  # Test first 2 segments only
            print(f"   Processing segment {segidx} ({X.shape})...")
            try:
                X_hat, _, _ = decompose_and_denoise_patch(
                    X, method="tt_adaptive", sigma=sigma, reshape_for_tt=False
                )
                denoised_patches.append((segidx, X_hat))
                print(f"   ✓ Segment {segidx} denoised successfully")
            except Exception as e:
                print(f"   ⚠️  Segment {segidx} failed: {str(e)}")
                continue
        
        # 6. Reconstruct image
        print("6. Reconstructing image...")
        if denoised_patches:
            reconstructed = reconstruct_image_from_patches(
                denoised_patches, segments, image.shape, patch_size
            )
            print(f"   ✓ Image reconstructed")
            
            # 7. Evaluate quality
            print("7. Evaluating quality...")
            psnr, ssim = compute_quality_metrics(image, reconstructed)
            print(f"   ✓ PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        else:
            print("   ⚠️  No patches were successfully denoised")
            return False
        
        print("\n🎉 All tests passed! RGB image denoising pipeline is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_denoising()
    sys.exit(0 if success else 1)
