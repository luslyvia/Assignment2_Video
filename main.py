import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import module của thành viên còn lại
from block_analyzer import analyze_frame_adaptive, debug_visualize_blocks

# ==========================================
# 1. THUẬT TOÁN CỐT LÕI (Linh - Member A)
# ==========================================

def calculate_block_variance(image, block_size=16):
    """
    Chia ảnh thành các khối và tính phương sai (Variance) của từng khối.
    Phương sai cao = Vùng nhiều chi tiết (Texture)
    Phương sai thấp = Vùng phẳng (Smooth)
    """
    h, w = image.shape[:2]
    grid_h = h // block_size
    grid_w = w // block_size
    variance_map = np.zeros((grid_h, grid_w))

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    for i in range(grid_h):
        for j in range(grid_w):
            block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            variance_map[i, j] = np.var(block)

    return variance_map

def generate_perceptual_qp_map(variance_map, base_qp=24, sensitivity=1.0):
    """
    Thuật toán Lượng tử hóa thích nghi (Adaptive QP):
    - Vùng phẳng (Variance thấp): QP thấp → giữ chất lượng tránh blocking artifact
    - Vùng chi tiết (Variance cao): QP cao → tiết kiệm bitrate (Texture Masking)
    """
    max_var = np.max(variance_map) if np.max(variance_map) > 0 else 1
    norm_variance = variance_map / max_var
    qp_delta = (norm_variance * 12 * sensitivity).astype(int)
    qp_map = base_qp + qp_delta
    qp_map = np.clip(qp_map, 1, 51)
    return qp_map

# ==========================================
# BƯỚC 3: Tích hợp block_analyzer vào QP map
# ==========================================

def generate_hybrid_qp_map(image, base_qp=24, sensitivity=1.0, threshold=100, block_size=16):
    """
    Kết hợp 2 thuật toán:
    - generate_perceptual_qp_map (Member A): tính QP liên tục từ variance
    - analyze_frame_adaptive (Member B): bổ sung delta ±3 theo ngưỡng nhị phân
    Kết quả: QP map chính xác hơn, kế thừa công sức cả hai thành viên
    """
    # Bước A: QP map liên tục từ thuật toán variance
    var_map = calculate_block_variance(image, block_size)
    qp_map_continuous = generate_perceptual_qp_map(var_map, base_qp, sensitivity)

    # Bước B: Delta QP nhị phân từ block_analyzer (chỉ hoạt động với block_size=16)
    if block_size == 16:
        delta_matrix = analyze_frame_adaptive(image, threshold=threshold)
        delta_np = np.array(delta_matrix)

        # Đảm bảo kích thước khớp (crop nếu cần)
        h_min = min(qp_map_continuous.shape[0], delta_np.shape[0])
        w_min = min(qp_map_continuous.shape[1], delta_np.shape[1])
        qp_map_continuous = qp_map_continuous[:h_min, :w_min]
        delta_np = delta_np[:h_min, :w_min]

        # Cộng delta từ Member B vào QP map của Member A
        qp_map_hybrid = qp_map_continuous + delta_np
    else:
        qp_map_hybrid = qp_map_continuous

    return np.clip(qp_map_hybrid, 1, 51).astype(int), var_map

def simulate_quantization(image, qp_map, block_size=16):
    """
    Mô phỏng nén: DCT → Quantize → IDCT theo từng block với QP riêng biệt
    """
    h, w, c = image.shape
    grid_h = h // block_size
    grid_w = w // block_size
    compressed_image = np.zeros_like(image, dtype=np.float32)

    q_matrix_base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    if block_size != 8:
        q_matrix_base = cv2.resize(q_matrix_base, (block_size, block_size))

    for channel in range(c):
        for i in range(grid_h):
            for j in range(grid_w):
                block = image[i*block_size:(i+1)*block_size,
                              j*block_size:(j+1)*block_size, channel].astype(np.float32)
                dct_block = cv2.dct(block)
                qp = qp_map[i, j]
                scale_factor = (qp / 24.0)
                q_matrix = q_matrix_base * scale_factor
                q_matrix[q_matrix == 0] = 1
                quantized = np.round(dct_block / q_matrix)
                dequantized = quantized * q_matrix
                idct_block = cv2.idct(dequantized)
                compressed_image[i*block_size:(i+1)*block_size,
                                 j*block_size:(j+1)*block_size, channel] = idct_block

    return np.clip(compressed_image, 0, 255).astype(np.uint8)

# ==========================================
# BƯỚC 5: Đọc CSV baseline và vẽ RD-Curve
# ==========================================

def load_baseline_csv(csv_path="results/baseline_metrics.csv"):
    """Đọc file CSV từ run_baseline.py, trả về DataFrame hoặc None nếu chưa có"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception:
            return None
    return None

def plot_rd_curve(df):
    """
    Vẽ biểu đồ Rate-Distortion Curve: Bitrate (kbps) vs PSNR (dB)
    Mỗi video 1 đường, càng lên trên - sang trái càng tốt
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    videos = df["Video"].unique()

    for i, video in enumerate(videos):
        subset = df[df["Video"] == video].sort_values("Bitrate (kbps)")
        ax.plot(
            subset["Bitrate (kbps)"],
            subset["PSNR (dB)"],
            marker='o', linewidth=2,
            color=colors[i % len(colors)],
            label=video
        )
        # Gán nhãn QP lên từng điểm
        for _, row in subset.iterrows():
            ax.annotate(
                f"QP={int(row['QP'])}",
                xy=(row["Bitrate (kbps)"], row["PSNR (dB)"]),
                xytext=(6, 2), textcoords="offset points",
                fontsize=8, color=colors[i % len(colors)]
            )

    ax.set_xlabel("Bitrate (kbps)", fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=11)
    ax.set_title("Rate-Distortion Curve — Baseline H.264 (libx264)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def estimate_adaptive_bitrate_saving(df, base_qp, avg_qp_adaptive):
    """
    Ước tính bitrate thật của phương pháp Adaptive dựa trên nội suy RD-curve.
    Tìm 2 mức QP baseline gần nhất với avg_qp_adaptive, nội suy bitrate.
    Trả về bitrate ước tính (kbps) và % tiết kiệm so với baseline cùng QP.
    """
    try:
        # Gộp tất cả video, lấy bitrate trung bình theo QP
        avg_by_qp = df.groupby("QP")["Bitrate (kbps)"].mean().reset_index()
        avg_by_qp = avg_by_qp.sort_values("QP")

        qp_vals = avg_by_qp["QP"].values.astype(float)
        br_vals = avg_by_qp["Bitrate (kbps)"].values.astype(float)

        # Nội suy tuyến tính bitrate tại base_qp và avg_qp_adaptive
        bitrate_at_base = float(np.interp(base_qp, qp_vals, br_vals))
        bitrate_at_adaptive = float(np.interp(avg_qp_adaptive, qp_vals, br_vals))

        saving_pct = max(0.0, (bitrate_at_base - bitrate_at_adaptive) / bitrate_at_base * 100)
        return bitrate_at_base, bitrate_at_adaptive, saving_pct
    except Exception:
        return None, None, None

# ==========================================
# GIAO DIỆN STREAMLIT
# ==========================================

st.set_page_config(layout="wide", page_title="Perceptual QP Optimization")
st.title("Perceptual Quantization Optimization System")

# --- SIDEBAR ---
st.sidebar.header("System Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload sample image/frame", type=["jpg", "png", "jpeg"])
base_qp = st.sidebar.slider("Base Quantization Parameter (Base QP)", min_value=10, max_value=40, value=24)
sensitivity = st.sidebar.slider("Perceptual Sensitivity", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
block_size = st.sidebar.selectbox("Quantization Block Size", [8, 16], index=1)
threshold = st.sidebar.slider("Region Calssification Threshold (Threshold T)", min_value=50, max_value=500, value=100, step=50,
                               help="Variance threshold for block_analyzer to classify flat vs. textured regions")

st.sidebar.markdown("---")
st.sidebar.markdown("**QP Map Color Legend:**")
st.sidebar.markdown("Red = High QP → Heavy compression (textured region)")
st.sidebar.markdown("Blue = Low QP → Preserve details (flat region)")

# --- IMAGE LOADING ---
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
else:
    img_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img_bgr, (128, 128), 60, (255, 255, 255), -1)
    cv2.putText(img_bgr, "Hay upload anh", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Ensure dimensions are divisible by block_size
h, w = img_bgr.shape[:2]
h_new = (h // block_size) * block_size
w_new = (w // block_size) * block_size
img_bgr = cv2.resize(img_bgr, (w_new, h_new))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- CHẠY THUẬT TOÁN ---
with st.spinner("Processing algorithm..."):
    # BƯỚC 3: Dùng hybrid QP map (kết hợp cả 2 module)
    qp_map, var_map = generate_hybrid_qp_map(img_bgr, base_qp, sensitivity, threshold, block_size)

    # Nén Adaptive (Proposed)
    img_compressed_adaptive = simulate_quantization(img_bgr, qp_map, block_size)
    img_compressed_adaptive_rgb = cv2.cvtColor(img_compressed_adaptive, cv2.COLOR_BGR2RGB)

    # Nén Baseline (QP cố định)
    fixed_qp_map = np.ones_like(qp_map) * base_qp
    img_compressed_baseline = simulate_quantization(img_bgr, fixed_qp_map.astype(int), block_size)
    img_compressed_baseline_rgb = cv2.cvtColor(img_compressed_baseline, cv2.COLOR_BGR2RGB)

    # Metrics
    psnr_base = psnr(img_rgb, img_compressed_baseline_rgb)
    ssim_base = ssim(img_rgb, img_compressed_baseline_rgb, channel_axis=2)
    psnr_adapt = psnr(img_rgb, img_compressed_adaptive_rgb)
    ssim_adapt = ssim(img_rgb, img_compressed_adaptive_rgb, channel_axis=2)

    avg_qp_baseline = float(np.mean(fixed_qp_map))
    avg_qp_adaptive = float(np.mean(qp_map))

    # BƯỚC 4: Difference map (Before/After visual)
    diff_map = cv2.absdiff(img_compressed_baseline_rgb, img_compressed_adaptive_rgb)
    # Khuếch đại để dễ nhìn (×5)
    diff_amplified = np.clip(diff_map.astype(np.float32) * 5, 0, 255).astype(np.uint8)

    # Visualization từ block_analyzer (debug grid)
    if block_size == 16:
        delta_matrix = analyze_frame_adaptive(img_bgr, threshold=threshold)
        block_viz_bgr = debug_visualize_blocks(img_bgr, delta_matrix)
        block_viz_rgb = cv2.cvtColor(block_viz_bgr, cv2.COLOR_BGR2RGB)
    else:
        block_viz_rgb = img_rgb.copy()

# ==========================================
# PHẦN HIỂN THỊ 1: So sánh ảnh gốc / QP map / Adaptive
# ==========================================
st.header("Part 1 - Adaptive Compression Results")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Original Frame (Raw)")
    st.image(img_rgb, use_container_width=True)
    st.info("Status: Uncompressed")

with col2:
    st.subheader("2. QP Map (Hybrid)")
    fig_qp, ax_qp = plt.subplots()
    im = ax_qp.imshow(qp_map, cmap='jet', vmin=1, vmax=51)
    ax_qp.axis('off')
    fig_qp.colorbar(im, ax=ax_qp, orientation='horizontal', label='QP Value')
    st.pyplot(fig_qp)
    plt.close(fig_qp)

with col3:
    st.subheader("3. After Adaptive Compression (Proposed)")
    st.image(img_compressed_adaptive_rgb, use_container_width=True)
    st.success(f"Average QP: {avg_qp_adaptive:.1f} (Baseline: {avg_qp_baseline:.1f})")

# ==========================================
# BƯỚC 4: BEFORE / AFTER VISUAL QUALITY
# ==========================================
st.write("---")
st.header("🔍 Phần 2 — Before/After Visual Quality")

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.subheader("Baseline")
    st.image(img_compressed_baseline_rgb, use_container_width=True)
    st.caption(f"Fixed QP = {base_qp} | PSNR: {psnr_base:.2f} dB | SSIM: {ssim_base:.4f}")

with col_b:
    st.subheader("Adaptive (Proposed)")
    st.image(img_compressed_adaptive_rgb, use_container_width=True)
    st.caption(f"Average QP = {avg_qp_adaptive:.1f} | PSNR: {psnr_adapt:.2f} dB | SSIM: {ssim_adapt:.4f}")

with col_c:
    st.subheader("Difference Map (×5)")
    st.image(diff_amplified, use_container_width=True)
    st.caption("Bright regions = high discrepancy between methods. Dark regions = high similarity.")

with col_d:
    st.subheader("Block Classification")
    st.image(block_viz_rgb, use_container_width=True)
    st.caption("Green = flat region (QP increased). Red = textured region (QP decreased)")

# ==========================================
# BẢNG SO SÁNH METRICS
# ==========================================
st.write("---")
st.header("Part 3 - Performance Analysis Table")

# Đọc CSV để lấy bitrate thật (Bước 5)
df_baseline = load_baseline_csv("results/baseline_metrics.csv")

if df_baseline is not None:
    br_base_real, br_adapt_est, saving_pct = estimate_adaptive_bitrate_saving(df_baseline, base_qp, avg_qp_adaptive)
    bitrate_base_str = f"{br_base_real:.0f} kbps (thực đo)" if br_base_real else "N/A"
    bitrate_adapt_str = f"{br_adapt_est:.0f} kbps (nội suy)" if br_adapt_est else "N/A"
    saving_str = f"{saving_pct:.1f}%" if saving_pct is not None else "N/A"
else:
    bitrate_base_str = "CSV not available"
    bitrate_adapt_str = "CCSV not available"
    saving_str = f"~{max(0.0, (avg_qp_adaptive - avg_qp_baseline) * 3.5):.1f}% (estimated)"

df_metrics = pd.DataFrame({
    "Method": ["Fixed compression (Baseline)", "Perceptual compression (Proposed)"],
    "Average QP": [f"{avg_qp_baseline:.1f}", f"{avg_qp_adaptive:.1f}"],
    "Bitrate": [bitrate_base_str, bitrate_adapt_str],
    "PSNR (dB)": [f"{psnr_base:.2f}", f"{psnr_adapt:.2f}"],
    "SSIM": [f"{ssim_base:.4f}", f"{ssim_adapt:.4f}"],
    "Bitrate savings": ["—", saving_str],
})
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

# ==========================================
# BƯỚC 5: RD-CURVE TỪ CSV THỰC ĐO
# ==========================================
st.write("---")
st.header("Part 4 - Rate-Distortion Curve (Measured Baseline)")

if df_baseline is not None:
    st.success(f"Loaded baseline data: {len(df_baseline)} rows from results/baseline_metrics.csv")
    
    col_rd1, col_rd2 = st.columns([2, 1])
    with col_rd1:
        fig_rd = plot_rd_curve(df_baseline)
        st.pyplot(fig_rd)
        plt.close(fig_rd)
    with col_rd2:
        st.subheader("Raw data")
        st.dataframe(df_baseline, use_container_width=True, hide_index=True)
    
else:
    st.warning("Warning: The file `results/baseline_metrics.csv` was not found. Please run `python run_baseline.py` first to generate actual data.")
