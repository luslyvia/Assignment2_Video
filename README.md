# Perceptual Quantization Optimization System

This project implements a **Perceptual Quantization Optimization** system for multimedia data compression. It features a live interactive system that dynamically adjusts the Quantization Parameter (QP) for individual image/frame blocks based on the **Human Visual System (HVS)** characteristics, specifically leveraging **Texture Masking** principles.

---

## System Overview & Architecture

The framework consists of a pipeline that performs spatial complexity analysis to optimize bits allocation without degrading the perceptual quality:

1. **Spatial Complexity Analysis:** Input frames are partitioned into $16 \times 16$ macroblocks to calculate local spatial variance.
2. **Adaptive QP Mapping:** A non-linear mapping function assigns lower QP values to smooth/critical regions (preserving details) and higher QP values to highly textured regions (compressing aggressively).
3. **Quantization Engine:** Simulates the compression pipeline via Discrete Cosine Transform (DCT) and block-wise adaptive quantization matrix scaling.

---

## Installation & Environment Setup

Follow these steps to configure the environment and run the live demo system:

### 1. Prerequisites
Ensure you have **Python 3.8 or higher** installed on your system.

### 2. Install Required Dependencies
Open your terminal or command prompt in the project root directory and execute the following command to install all necessary libraries:

```bash
pip install opencv-python numpy streamlit matplotlib scikit-image
