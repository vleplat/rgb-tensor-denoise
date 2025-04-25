# 🧠 Super Resolution Image Project

A lightweight post-processing toolbox to **reduce tiling artifacts in Super-Resolution (SR) images** using **patch-based denoising techniques**.  
The core idea is to **leverage advanced image denoising methods** — based on **low-rank tensor decompositions (Tucker and Tensor Train)** — to smooth out local inconsistencies in SR outputs.

This package adapts recent developments in **image denoising with tensor models** and repurposes them for **artifact removal** in SR images. It supports:
- **Superpixel-based segmentation** to capture local spatial coherence,
- **Patch grouping and higher-order tensor reshaping**,
- **Tucker and TT-based denoising with adaptive or fixed compression**,
- Full end-to-end pipeline including reconstruction and evaluation.

---

## 🔍 Motivation: Denoising for SR Artifact Smoothing

While SR networks aim to enhance resolution, they can produce **tiling artifacts** — local grid-like discontinuities — due to patch-based processing or architectural choices.  
Our strategy is to **treat these artifacts as a kind of noise**, and apply **tensor-based denoising** methods to suppress them in a spatially-aware, patch-wise manner.

The approach builds on a classic **image denoising pipeline**, adapted with this goal in mind.

---

## 🧩 Image Denoising Pipeline (used as SR Artifact Smoother)

This procedure also works as a general-purpose **denoiser** — but here, it is primarily **a tool for post-processing SR images**.

### 🖼️ Problem Setup

We are given a **color image** $Y \\in \\mathbb{R}^{I \\times J \\times 3}$ affected by **local visual artifacts** or **Gaussian noise**. Our goal is to reconstruct a cleaner version using local low-rank tensor approximations.

---

### 🧱 Step-by-Step Overview

#### **(a) Superpixel Clustering / Segmentation**

- Segment the image using a superpixel algorithm (e.g., SLIC),
- Each segment captures spatially coherent pixel clusters.

#### **(b) Patch Tensor Construction + Low-Rank Denoising**

For **each segment**:

1. **Extract patches**:
   - Centered patches of size $d \\times d \\times 3$ around each pixel.
   - Grouped into a 4D tensor:
     \[
     X \\in \\mathbb{R}^{d \\times d \\times 3 \\times K}
     \]
     where $K$ is the number of patches in the segment.

2. **Denoise using low-rank tensor approximation**:
   - Either **Tucker** or **Tensor Train (TT)**:
     \[
     \\hat{X} \\approx X
     \quad \\text{subject to} \\quad \\| X - \\hat{X} \\|_F^2 \\leq \\epsilon^2 = 3\\sigma^2 d^2 K
     \]
   - Goal: minimize the **number of parameters** in $\\hat{X}$ (i.e., achieve compression).

3. **Options & Configurations**:
   - **TT mode**:
     - Fixed-rank TT decomposition
     - Adaptive TT decomposition (via approximation error bound)
   - **Reshaping**:
     - Keep $X$ as a 4th-order tensor, or
     - Reshape into higher-order tensors (e.g., 6th or 10th order) to enhance TT compression
   - **Tunable Parameters**:
     - $\\sigma$: estimated noise/artifact level
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

### 🎯 Goal

Achieve **artifact-free, visually smooth reconstructions** of SR images by leveraging **local patch redundancy** and **tensor decompositions**.

This method can also serve as a standalone **denoising tool** — but its primary value here is in **refining SR results**.

---

### 🚀 Future Work

We are actively exploring enhancements including:

- **Parallelized Patch Denoising**:  
  Denoising across segments is **embarrassingly parallel** — a parallel implementation using `joblib`, `multiprocessing`, or GPU-backed tensor libraries is in progress.

- **TT-based smoothing with Total Variation regularization** (experimental).

