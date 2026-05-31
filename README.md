# Perceptual Quantization Optimization System

## Overview

The **Perceptual Quantization Optimization System** is a multimedia compression framework that dynamically adjusts the Quantization Parameter (QP) based on Human Visual System (HVS) characteristics.

Instead of applying a fixed quantization level to all image regions, the system analyzes local spatial complexity and allocates bits more efficiently using **Texture Masking** principles. Smooth and visually sensitive regions are preserved with lower QP values, while highly textured regions are compressed more aggressively with higher QP values.

The project provides an interactive visualization platform built with Streamlit, allowing users to observe the adaptive quantization process in real time.

---

# System Architecture

The processing pipeline consists of the following stages:

```
Input Image / Video Frame
            │
            ▼
Spatial Complexity Analysis
(16×16 Macroblock Variance)
            │
            ▼
Adaptive QP Mapping
(HVS-Based Optimization)
            │
            ▼
DCT Transformation
            │
            ▼
Adaptive Quantization
            │
            ▼
Inverse Quantization
            │
            ▼
Inverse DCT
            │
            ▼
Reconstructed Frame
            │
            ▼
Quality Evaluation
(PSNR, SSIM, Compression Statistics)
```

---

# Main Features

- Human Visual System (HVS) based optimization
- Texture masking driven QP adaptation
- Macroblock-level complexity analysis
- Adaptive quantization matrix scaling
- DCT-based compression simulation
- Interactive Streamlit dashboard
- Visualization of:
  - Original image/frame
  - Variance heatmap
  - QP allocation map
  - Reconstructed output
  - Compression metrics

---

# Project Structure

```text
Assignment2_Video/
│
├── app.py
├── main.py
├── src/
│   ├── complexity_analysis.py
│   ├── adaptive_qp.py
│   ├── dct_engine.py
│   ├── quantization.py
│   ├── metrics.py
│   └── utils.py
│
├── assets/
│   ├── sample_images/
│   └── sample_videos/
│
├── output/
│   ├── reconstructed/
│   ├── heatmaps/
│   └── reports/
│
├── requirements.txt
└── README.md
```

> Actual filenames may vary depending on the final implementation.

---

# System Requirements

## Operating System

- Ubuntu 20.04 LTS or newer
- Ubuntu 22.04 LTS recommended

## Python

Python version:

```bash
Python >= 3.8
```

Check installed version:

```bash
python3 --version
```

---

# Installation Guide (Ubuntu)

## Step 1: Update Ubuntu Packages

```bash
sudo apt update
sudo apt upgrade -y
```

---

## Step 2: Install Python and Pip

```bash
sudo apt install python3 python3-pip -y
```

Verify installation:

```bash
python3 --version
pip3 --version
```

---

## Step 3: Clone the Repository

```bash
git clone https://github.com/luslyvia/Assignment2_Video.git
```

Navigate into the project directory:

```bash
cd Assignment2_Video
```

---

## Step 4: Create Virtual Environment (Recommended)

Install virtual environment package:

```bash
sudo apt install python3-venv -y
```

Create environment:

```bash
python3 -m venv venv
```

Activate environment:

```bash
source venv/bin/activate
```

You should see:

```bash
(venv)
```

at the beginning of your terminal prompt.

---

## Step 5: Install Required Dependencies

Install all required libraries:

```bash
pip install opencv-python numpy streamlit matplotlib scikit-image
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

---

# Running the Application

## Option 1: Launch Interactive Streamlit Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

Expected output:

```bash
Local URL: http://localhost:8501
```

Open your browser and navigate to:

```text
http://localhost:8501
```

---

## Option 2: Run Compression Pipeline Directly

If the project contains a standalone processing script:

```bash
python3 main.py
```

This mode processes input images/videos and generates outputs automatically.

---

# Using the Interactive Dashboard

## Upload Input

The system accepts:

- JPG
- JPEG
- PNG
- BMP

and optionally:

- MP4
- AVI

depending on implementation.

---

## Processing Workflow

### 1. Upload an Image

Select an image from your local machine.

### 2. Complexity Analysis

The system divides the image into:

```text
16 × 16 macroblocks
```

and computes local variance values.

### 3. Adaptive QP Assignment

Each macroblock receives a QP value according to:

- Low variance → Lower QP
- High variance → Higher QP

This follows Texture Masking principles.

### 4. DCT Compression Simulation

The system applies:

- Forward DCT
- Quantization
- Inverse Quantization
- Inverse DCT

to reconstruct the image.

### 5. Visualization

The dashboard displays:

- Original Image
- Variance Map
- QP Heatmap
- Reconstructed Image
- Error Map

---

# Evaluation Metrics

## Peak Signal-to-Noise Ratio (PSNR)

Measures reconstruction quality.

Formula:

\[
PSNR = 10 \log_{10}
\left(
\frac{255^2}{MSE}
\right)
\]

Higher PSNR indicates better quality.

---

## Structural Similarity Index (SSIM)

Measures perceptual similarity between images.

Range:

```text
0 → Completely different
1 → Identical
```

---

## Compression Ratio

Measures storage reduction effectiveness.

Formula:

\[
Compression\ Ratio =
\frac{Original\ Size}
{Compressed\ Size}
\]

---

# Example Output

The system generates:

### Original Frame

Input image before processing.

### Variance Heatmap

Visualization of spatial complexity.

### QP Allocation Map

Block-wise adaptive quantization parameter distribution.

### Reconstructed Frame

Image after compression simulation.

### Compression Report

Contains:

- Average QP
- PSNR
- SSIM
- Compression Ratio

---

# Troubleshooting

## Streamlit Command Not Found

Install Streamlit:

```bash
pip install streamlit
```

Verify:

```bash
streamlit --version
```

---

## OpenCV Import Error

Install OpenCV:

```bash
pip install opencv-python
```

Test:

```bash
python3
```

```python
import cv2
print(cv2.__version__)
```

---

## Permission Denied

Grant execution permission:

```bash
chmod +x app.py
```

or

```bash
chmod +x main.py
```

---

# Future Improvements

- Temporal masking integration
- Motion-compensated adaptive QP allocation
- Video-level optimization
- Real-time streaming support
- JPEG/H.264 compatible quantization matrices
- Machine learning based perceptual quality prediction

---

# Authors

Multimedia Compression Project

Department of Computer Engineering

Hanoi University of Science and Technology

---

# License

This project is developed for educational and research purposes.
