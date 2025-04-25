# SR_Project
A lightweight post-processing toolbox to smooth super-resolution (SR) images with patch-based denoising using Tucker and Tensor Train (TT) decompositions. Designed to reduce tiling artifacts via high-order tensor modeling and superpixel-aware segmentation. Includes adaptive TT compression and support for higher-order reshaping.


## 🧩 Image Reconstruction

This task focuses on **image denoising and reconstruction** using local low-rank **tensor decomposition** (Tucker or Tensor Train) within segmented image regions.

---

### 🖼️ Problem Setup

You are given a **color image** $ Y \in \mathbb{R}^{I \times J \times 3}$ that is degraded by **Gaussian noise** with:
- Mean = 0  
- Standard deviation = $ \sigma $

---

### 🧱 Step-by-Step Instructions

#### **(a) Superpixel Clustering / Segmentation**

- Segment the noisy image $ Y $ using a superpixel clustering algorithm (e.g., SLIC).
- This produces clusters (segments) of pixels with similar appearance.

#### **(b) Patch Tensor Construction + Low-Rank Approximation**

For **each segment**:

1. **Extract patches**:
   - For each pixel in the segment, extract a centered patch of size $ d \times d \times 3 $.
   - Stack all patches into a tensor:
     $
     X \in \mathbb{R}^{d \times d \times 3 \times K}
     $
     where $K $ is the number of pixels (patches) in the segment.

2. **Denoise using a low-rank tensor model**:
   - Approximate $ X $ using a **Tucker** or **Tensor Train (TT)** decomposition:
     $
     \hat{X} \approx X
     $
   - Subject to a reconstruction error bound:
     $
     \| X - \hat{X} \|_F^2 \leq \epsilon^2 = 3\sigma^2 d^2 K
     $
   - Goal: minimize the **number of parameters** in $ \hat{X} $.

2. **Notes**:
  - For TT: two methods can be used
    - TT decomposition with fixed ranks
    - TT decomposition with error bounded constraints -> it computes the best TT ranks so that the error of approximation is bounded (for each patch)
  - For each method, we have the choice of the shape
    - One may keep the orginal 4-th order format of $X$
    - One can reshape it into a 6-th order tensor format, then compute the TT decomposition. The motivation is to take advantage of TT compression capabilities for high-order tensors, in other words to obtain higher compression
  - Parameters to play with:
    - the value of $\sigma$ - for real life noisy signals, we have to try different values
    - the value of $d$ - this has an impact on the smoothing since higher value for $d$ leads to higher overlapping for the aggregation steps. 
    - activate or not the 6-th order reshaping: set parameter "reshape_for_tt" to False or True 
    - n_segments for superpixel method 

---

#### **(c) Reconstruct the Image**

1. Place denoised patches back into their original image locations.
2. Average overlapping regions.
3. Compute reconstruction quality metrics:
   - **PSNR** (Peak Signal-to-Noise Ratio)
   - **SSIM** (Structural Similarity Index)

---

### 🎯 Goal

Reconstruct a **high-quality, denoised image** by leveraging **local patch redundancy** through **low-rank tensor decompositions** on segments of the image.

