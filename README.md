# Perceptual Quantization Optimization for Video Compression

## Overview

The **Perceptual Quantization Optimization System** is a video compression framework that dynamically adjusts the Quantization Parameter (QP) according to Human Visual System (HVS) characteristics.

Instead of applying a fixed quantization level to every region of a video frame, the system analyzes spatial complexity on a macroblock basis and allocates bits more efficiently using **Texture Masking** principles. Smooth and visually sensitive regions receive lower QP values to preserve important visual details, while highly textured regions are compressed more aggressively.

The system processes videos frame-by-frame and provides an interactive visualization platform that demonstrates adaptive quantization behavior and compression quality metrics.

> **Note:** This project focuses on spatial perceptual optimization. Temporal redundancy reduction techniques such as motion estimation and motion compensation are not included in the current implementation.

---

# System Architecture

The processing pipeline consists of the following stages:

```text
Input Video
     │
     ▼
Frame Extraction
     │
     ▼
Spatial Complexity Analysis
(16×16 Macroblocks)
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
Frame Reconstruction
     │
     ▼
Output Video
     │
     ▼
Quality Evaluation
(PSNR, SSIM)
```

---

# Main Features

- Human Visual System (HVS) based perceptual optimization
- Texture masking driven adaptive quantization
- Frame-by-frame video processing
- 16×16 macroblock spatial complexity analysis
- Dynamic Quantization Parameter (QP) allocation
- DCT-based compression simulation
- Interactive Streamlit visualization
- Original vs reconstructed video comparison
- Quality evaluation using PSNR and SSIM
- Compression statistics and performance reporting

---

# Project Structure

```text
Assignment2_Video/
│
├── app.py
├── main.py
├── src/
│   ├── video_loader.py
│   ├── complexity_analysis.py
│   ├── adaptive_qp.py
│   ├── dct_engine.py
│   ├── quantization.py
│   ├── metrics.py
│   └── utils.py
│
├── assets/
│   └── sample_videos/
│
├── output/
│   ├── compressed_videos/
│   ├── reconstructed_frames/
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

## Python Version

```bash
Python >= 3.8
```

Check your Python version:

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

Navigate to the project directory:

```bash
cd Assignment2_Video
```

---

## Step 4: Create a Virtual Environment (Recommended)

Install virtual environment support:

```bash
sudo apt install python3-venv -y
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the environment:

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

Or install directly from the requirements file:

```bash
pip install -r requirements.txt
```

---

# Running the Application

## Launch the Interactive Streamlit Dashboard

Start the application:

```bash
streamlit run app.py
```

Expected output:

```text
Local URL: http://localhost:8501
```

Open your web browser and navigate to:

```text
http://localhost:8501
```

---

## Alternative: Run the Processing Pipeline Directly

If the repository includes a standalone processing script:

```bash
python3 main.py
```

This mode processes the selected video and generates output files automatically.

---

# Supported Input Formats

The system currently supports video files only:

- MP4
- AVI
- MOV
- MKV

Input videos are processed frame-by-frame through the adaptive quantization pipeline.

---

# Using the Interactive Dashboard

## 1. Upload a Video

Upload a supported video file from your local machine.

---

## 2. Frame Extraction

The system extracts individual frames from the uploaded video.

---

## 3. Spatial Complexity Analysis

Each frame is divided into:

```text
16 × 16 Macroblocks
```

The local variance of each macroblock is calculated to estimate spatial complexity.

---

## 4. Adaptive QP Assignment

Based on the calculated variance:

- Low variance → Lower QP
- High variance → Higher QP

This follows Human Visual System texture masking principles.

---

## 5. Compression Simulation

For every frame, the system performs:

1. Forward DCT
2. Adaptive Quantization
3. Inverse Quantization
4. Inverse DCT

---

## 6. Video Reconstruction

The reconstructed frames are combined to generate the final output video.

---

## 7. Visualization

The dashboard displays:

- Original Video
- Reconstructed Video
- Variance Heatmaps
- QP Distribution Maps
- Compression Statistics
- Quality Metrics

---

# Compression Methodology

## Spatial Complexity Analysis

Each frame is partitioned into 16×16 macroblocks.

For every macroblock, the local variance is computed:

\[
Variance = \frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
\]

where:

- \(x_i\) is a pixel value
- \(\mu\) is the block mean
- \(N\) is the number of pixels in the block

Higher variance indicates more texture and visual complexity.

---

## Adaptive Quantization Parameter Mapping

The computed variance is mapped to Quantization Parameters (QP):

- Smooth regions → Smaller QP
- Textured regions → Larger QP

This strategy reduces bitrate while maintaining perceptual quality.

---

## DCT-Based Compression

The system applies the Discrete Cosine Transform (DCT) to each block:

\[
F(u,v)
\]

The transformed coefficients are then quantized using adaptive quantization matrices scaled according to the assigned QP.

---

# Evaluation Metrics

## Peak Signal-to-Noise Ratio (PSNR)

PSNR measures reconstruction quality.

\[
PSNR = 10 \log_{10}
\left(
\frac{255^2}{MSE}
\right)
\]

Higher PSNR values indicate better reconstruction quality.

---

## Structural Similarity Index (SSIM)

SSIM measures perceptual similarity between the original and reconstructed frames.

Range:

```text
0 → Completely Different
1 → Identical
```

Higher SSIM values indicate better perceptual quality.

---

## Compression Ratio

Compression efficiency is measured using:

\[
Compression\ Ratio=
\frac{Original\ Size}
{Compressed\ Size}
\]

Higher ratios indicate better compression efficiency.

---

# Output

After processing, the system generates:

## Reconstructed Video

The reconstructed video produced by adaptive quantization.

## Variance Heatmaps

Visualization of frame spatial complexity.

## QP Allocation Maps

Visualization of adaptive QP distribution across macroblocks.

## Quality Metrics

- Average PSNR
- Average SSIM
- Average QP
- Processing Time

## Compression Statistics

- Original Video Size
- Estimated Compressed Size
- Compression Ratio

---

# Example Workflow

```text
sample.mp4
    │
    ▼
Frame Extraction
    │
    ▼
Spatial Complexity Analysis
    │
    ▼
Adaptive QP Optimization
    │
    ▼
DCT Quantization
    │
    ▼
Frame Reconstruction
    │
    ▼
output.mp4
```

---

# Troubleshooting

## Streamlit Command Not Found

Install Streamlit:

```bash
pip install streamlit
```

Verify installation:

```bash
streamlit --version
```

---

## OpenCV Import Error

Reinstall OpenCV:

```bash
pip install opencv-python
```

Test installation:

```bash
python3
```

```python
import cv2
print(cv2.__version__)
```

---

## Permission Denied Error

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
- Motion estimation and compensation
- GOP-based processing
- H.264/H.265 inspired rate control
- Real-time streaming support
- Machine learning based perceptual quality prediction
- Content-aware quantization matrices

