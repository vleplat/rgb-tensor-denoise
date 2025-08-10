# 🧠 RGB Image Denoising with Tensor Decompositions

A lightweight toolbox for **RGB image denoising** using **patch-based tensor decomposition techniques**.  
The core idea is to **leverage advanced image denoising methods** — based on **low-rank tensor decompositions (Tucker and Tensor Train)** — to remove noise from RGB images while preserving important image details.

This package implements recent developments in **image denoising with tensor models** and provides a complete pipeline for RGB image denoising. It supports:
- **Superpixel-based segmentation** to capture local spatial coherence,
- **Patch grouping and higher-order tensor reshaping**,
- **Tucker and TT-based denoising with adaptive or fixed compression**,
- Full end-to-end pipeline including reconstruction and evaluation.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SR_Project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import numpy, matplotlib, skimage, tensorly; print('✅ All packages installed successfully!')"
   ```

---

## 🔍 Motivation: Tensor-Based Image Denoising

Traditional image denoising methods often struggle with **preserving fine details** while removing noise. Our approach uses **tensor decompositions** to exploit the **redundancy in local image patches** and achieve better denoising results.

The method treats **noise as a perturbation** to the underlying low-rank structure of image patches and applies **tensor-based denoising** methods to suppress it in a spatially-aware, patch-wise manner.

---

## 🧩 Image Denoising Pipeline

### 🖼️ Problem Setup

We are given a **color image** $Y \in \mathbb{R}^{I \times J \times 3}$ affected by **Gaussian noise**. Our goal is to reconstruct a cleaner version using local low-rank tensor approximations.

---

### 🧱 Step-by-Step Overview

#### **(a) Superpixel Clustering / Segmentation**

- Segment the noisy image using a superpixel algorithm (e.g., SLIC),
- Each segment captures spatially coherent pixel clusters.

#### **(b) Patch Tensor Construction + Low-Rank Denoising**

For **each segment**:

1. **Extract patches**:
   - Centered patches of size $d \times d \times 3$ around each pixel.
   - Grouped into a 4D tensor:
     \[
     X \in \mathbb{R}^{d \times d \times 3 \times K}
     \]
     where $K$ is the number of patches in the segment.

2. **Denoise using low-rank tensor approximation**:
   - Either **Tucker** or **Tensor Train (TT)**:
     \[
     \hat{X} \approx X
     \quad \text{subject to} \quad \| X - \hat{X} \|_F^2 \leq \epsilon^2 = 3\sigma^2 d^2 K
     \]
   - Goal: minimize the **number of parameters** in $\hat{X}$ (i.e., achieve compression).

3. **Options & Configurations**:
   - **TT mode**:
     - Fixed-rank TT decomposition
     - Adaptive TT decomposition (via approximation error bound)
   - **Reshaping**:
     - Keep $X$ as a 4th-order tensor, or
     - Reshape into higher-order tensors (e.g., 6th or 10th order) to enhance TT compression
   - **Tunable Parameters**:
     - $\sigma$: estimated noise level
     - $d$: patch size — affects smoothing and overlap
     - `reshape_for_tt`: whether to activate high-order reshaping
     - `n_segments`: number of superpixels (impacts local grouping granularity)

---

#### **(c) Reconstruct the Image**

1. Place denoised patches back into their spatial locations.
2. Average overlapping pixels.
3. Evaluate quality with:
   - **PSNR** (Peak Signal-to-Noise Ratio)
   - **SSIM** (Structural Similarity Index)

---

## 🎯 Usage Examples

### Basic Usage

```python
from tucker_and_tt_denoise_HOD.io import load_image, add_gaussian_noise
from tucker_and_tt_denoise_HOD.segmentation import superpixel_segmentation
from tucker_and_tt_denoise_HOD.patches import extract_clustered_patches
from tucker_and_tt_denoise_HOD.decomposition import decompose_and_denoise_patch
from tucker_and_tt_denoise_HOD.reconstruction import reconstruct_image_from_patches
from tucker_and_tt_denoise_HOD.metrics import compute_quality_metrics

# Load and add noise to image
image = load_image("path/to/image.jpg")
noisy_image = add_gaussian_noise(image, sigma=0.075)

# Segment image
segments = superpixel_segmentation(noisy_image, n_segments=50)

# Extract patches
patches = extract_clustered_patches(noisy_image, segments, patch_size=6)

# Denoise patches
denoised_patches = []
for segidx, X in patches:
    X_hat, _, _ = decompose_and_denoise_patch(
        X, method="tt_adaptive", sigma=0.075, reshape_for_tt=True
    )
    denoised_patches.append((segidx, X_hat))

# Reconstruct image
reconstructed = reconstruct_image_from_patches(
    denoised_patches, segments, image.shape, patch_size=6
)

# Evaluate quality
psnr, ssim = compute_quality_metrics(image, reconstructed)
print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
```

### Running Demo Scripts

```bash
# Run the bird image denoising demo
python tucker_and_tt_denoise_HOD/demo_script_Bird.py
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open and run the demo notebook:
# - demo_notebook_tt_Bird.ipynb
```

### Testing

```bash
# Run the test script to verify installation
python test_denoising.py
```

---

## 📊 Performance

The method achieves competitive denoising results by:
- **Preserving fine details** through local patch processing
- **Exploiting redundancy** in image patches via tensor decompositions
- **Adaptive compression** based on noise levels
- **Spatially-aware processing** through superpixel segmentation

Typical results show:
- **PSNR improvements** of 2-5 dB over noisy images
- **SSIM improvements** of 0.1-0.3
- **Effective noise suppression** while preserving edges and textures

---

## 🔧 Configuration Options

### Key Parameters

- **`sigma`**: Noise standard deviation (default: 0.075)
- **`patch_size`**: Size of extracted patches (default: 6)
- **`n_segments`**: Number of superpixels (default: 50)
- **`method`**: Denoising method (`"tucker"`, `"tt_fixed"`, `"tt_adaptive"`)
- **`reshape_for_tt`**: Enable higher-order reshaping for TT (default: True)
- **`target_order`**: Target order for reshaping (default: 6-10)

### Method Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Tucker** | Fast, good compression | Limited to 4D tensors |
| **TT Fixed** | Predictable ranks | May over/under-compress |
| **TT Adaptive** | Optimal compression | Slower, more complex |

---

## 🚀 Future Work

We are actively exploring enhancements including:

- **Parallelized Patch Denoising**:  
  Denoising across segments is **embarrassingly parallel** — a parallel implementation using `joblib`, `multiprocessing`, or GPU-backed tensor libraries is in progress.

- **TT-based smoothing with Total Variation regularization** (experimental).

- **Integration with deep learning methods** for hybrid approaches.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📚 References

This implementation is based on research in tensor decompositions for image processing:

- Tucker decomposition for image denoising
- Tensor Train (TT) decompositions for high-dimensional data
- Superpixel-based image segmentation
- Patch-based image processing techniques

