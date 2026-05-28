import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess
import math
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from block_analyzer import analyze_frame_adaptive, debug_visualize_blocks

# ==========================================
# CORE ALGORITHMS
# ==========================================

def calculate_block_variance(image, block_size=16):
    """Tính phương sai từng block để đo độ phức tạp không gian."""
    h, w = image.shape[:2]
    grid_h = h // block_size
    grid_w = w // block_size
    variance_map = np.zeros((grid_h, grid_w))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    for i in range(grid_h):
        for j in range(grid_w):
            block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            variance_map[i, j] = np.var(block)
    return variance_map


def generate_perceptual_qp_map(variance_map, base_qp=24, sensitivity=1.0):
    """
    Member A — Adaptive QP từ variance liên tục.
    Chiều: vùng chi tiết (variance cao) → QP tăng (texture masking, tiết kiệm bitrate).
            vùng phẳng (variance thấp)  → QP gần base (mắt người nhạy với artifact ở vùng trơn).
    """
    max_var = np.max(variance_map) if np.max(variance_map) > 0 else 1.0
    norm_var = variance_map / max_var           # [0..1]
    qp_delta = norm_var * 12.0 * sensitivity   # range [0..+12]
    return np.clip(base_qp + qp_delta, 1, 51).astype(int)


def generate_hybrid_qp_map(image, base_qp=24, sensitivity=1.0, threshold=100, block_size=16):
    """
    Kết hợp Member A (variance liên tục) + Member B (delta ±3 nhị phân).
    Cả hai cùng chiều:
        vùng phẳng   → QP tăng  (+delta từ A, +3 từ B)
        vùng chi tiết → QP thấp hơn (delta nhỏ từ A, −3 từ B)
    """
    var_map  = calculate_block_variance(image, block_size)
    qp_map_a = generate_perceptual_qp_map(var_map, base_qp, sensitivity)

    if block_size == 16:
        delta_matrix = analyze_frame_adaptive(image, threshold=threshold)
        delta_np     = np.array(delta_matrix)
        h_min = min(qp_map_a.shape[0], delta_np.shape[0])
        w_min = min(qp_map_a.shape[1], delta_np.shape[1])
        qp_map_hybrid = qp_map_a[:h_min, :w_min] + delta_np[:h_min, :w_min]
    else:
        qp_map_hybrid = qp_map_a

    return np.clip(qp_map_hybrid, 1, 51).astype(int), var_map


def simulate_quantization(image, qp_map, block_size=16):
    """Mô phỏng nén: DCT → Quantize → IDCT theo từng block với QP riêng biệt."""
    h, w, c = image.shape
    grid_h = h // block_size
    grid_w = w // block_size
    compressed = np.zeros_like(image, dtype=np.float32)

    q_base = np.array([
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
        q_base = cv2.resize(q_base, (block_size, block_size))

    for ch in range(c):
        for i in range(grid_h):
            for j in range(grid_w):
                blk   = image[i*block_size:(i+1)*block_size,
                              j*block_size:(j+1)*block_size, ch].astype(np.float32)
                q_mat = np.maximum(q_base * (qp_map[i, j] / 24.0), 1)
                recon = cv2.idct(np.round(cv2.dct(blk) / q_mat) * q_mat)
                compressed[i*block_size:(i+1)*block_size,
                           j*block_size:(j+1)*block_size, ch] = recon

    return np.clip(compressed, 0, 255).astype(np.uint8)


# ==========================================
# BITRATE (FFmpeg thực đo)
# ==========================================

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps > 0 else 1.0


def run_real_compression_and_get_saving(input_video_path, base_qp, avg_qp_adaptive):
    """
    Gọi FFmpeg để nén video thật với 2 mức QP, đo bitrate thực tế.
    Trả về: br_base (kbps), br_adapt (kbps), saving_pct (%), path_base, path_adapt
    """
    try:
        os.makedirs("results/temp_videos", exist_ok=True)
        duration        = get_video_duration(input_video_path)
        qp_adaptive_int = int(np.clip(math.ceil(avg_qp_adaptive), 1, 51))

        # ĐƯA TIMESTAMP LÊN ĐẦU: Tạo tên file độc nhất trước khi gọi FFmpeg
        timestamp = int(time.time())
        out_base  = f"results/temp_videos/base_{timestamp}.mp4"
        out_adapt = f"results/temp_videos/adapt_{timestamp}.mp4"

        # FFmpeg sẽ ghi trực tiếp vào các file chứa timestamp
        for out, qp in [(out_base, int(base_qp)), (out_adapt, qp_adaptive_int)]:
            subprocess.run(
                ['ffmpeg', '-y', '-i', input_video_path,
                 '-an', '-c:v', 'libx264', '-preset', 'fast', '-qp', str(qp), out],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        br_base  = (os.path.getsize(out_base)  * 8) / duration / 1000
        br_adapt = (os.path.getsize(out_adapt) * 8) / duration / 1000
        saving   = max(0.0, (br_base - br_adapt) / br_base * 100)

        return br_base, br_adapt, saving, out_base, out_adapt
    except Exception:
        return None, None, None, None, None


# ==========================================
# RD-CURVE (từ baseline_metrics.csv)
# ==========================================

def load_baseline_csv(csv_path="results/baseline_metrics.csv"):
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return None
    return None


def plot_rd_curve(df):
    """Rate-Distortion Curve: Bitrate (kbps) vs PSNR (dB), mỗi video 1 đường."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, video in enumerate(df["Video"].unique()):
        sub = df[df["Video"] == video].sort_values("Bitrate (kbps)")
        ax.plot(sub["Bitrate (kbps)"], sub["PSNR (dB)"],
                marker='o', linewidth=2, color=colors[i % len(colors)], label=video)
        for _, row in sub.iterrows():
            ax.annotate(f"QP={int(row['QP'])}",
                        xy=(row["Bitrate (kbps)"], row["PSNR (dB)"]),
                        xytext=(6, 2), textcoords="offset points",
                        fontsize=8, color=colors[i % len(colors)])

    ax.set_xlabel("Bitrate (kbps)", fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=11)
    ax.set_title("Rate-Distortion Curve — Baseline H.264 (libx264)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(layout="wide", page_title="Perceptual QP Optimization")
st.title("Perceptual Quantization Optimization System")

# --- Sidebar ---
st.sidebar.header("System Control Panel")
uploaded_video = st.sidebar.file_uploader("Upload video file (.mp4)", type=["mp4"])
base_qp     = st.sidebar.slider("Base Quantization Parameter (Base QP)", 10, 40, 24)
sensitivity = st.sidebar.slider("Perceptual Sensitivity", 0.0, 2.0, 1.0, 0.1)
block_size  = st.sidebar.selectbox("Quantization Block Size", [8, 16], index=1)
threshold   = st.sidebar.slider("Region Classification Threshold (T)", 50, 500, 100, 50,
                                 help="Variance threshold: classify flat vs. textured blocks")
st.sidebar.markdown("---")
st.sidebar.markdown("**QP Map Color Legend:**")
st.sidebar.markdown("Red = High QP → Heavy compression (textured region)")
st.sidebar.markdown("Blue = Low QP → Preserve quality (flat region)")

# --- Load video & extract frame ---
if uploaded_video is not None:
    os.makedirs("results/temp_videos", exist_ok=True)
    video_path = f"results/temp_videos/{uploaded_video.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 30:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)   # skip fade-in
    ret, img_bgr = cap.read()
    if not ret or img_bgr is None:              # fallback to frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, img_bgr = cap.read()
    cap.release()

    if not ret or img_bgr is None:
        st.error("Cannot read frames from the uploaded video. Please check file format.")
        st.stop()
else:
    video_path = None
    img_bgr    = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.putText(img_bgr, "Upload a video to begin", (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Ensure divisible by block_size
h, w    = img_bgr.shape[:2]
img_bgr = cv2.resize(img_bgr, ((w // block_size) * block_size, (h // block_size) * block_size))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- Run algorithms ---
with st.spinner("Processing algorithm..."):
    qp_map, var_map = generate_hybrid_qp_map(img_bgr, base_qp, sensitivity, threshold, block_size)

    img_adaptive_bgr = simulate_quantization(img_bgr, qp_map, block_size)
    img_adaptive_rgb = cv2.cvtColor(img_adaptive_bgr, cv2.COLOR_BGR2RGB)

    fixed_qp_map     = (np.ones_like(qp_map) * base_qp).astype(int)
    img_baseline_bgr = simulate_quantization(img_bgr, fixed_qp_map, block_size)
    img_baseline_rgb = cv2.cvtColor(img_baseline_bgr, cv2.COLOR_BGR2RGB)

    # PSNR/SSIM tính từ simulate_quantization thật (không dùng noise giả)
    psnr_base  = psnr(img_rgb, img_baseline_rgb)
    ssim_base  = ssim(img_rgb, img_baseline_rgb, channel_axis=2)
    psnr_adapt = psnr(img_rgb, img_adaptive_rgb)
    ssim_adapt = ssim(img_rgb, img_adaptive_rgb, channel_axis=2)

    avg_qp_baseline = float(np.mean(fixed_qp_map))
    avg_qp_adaptive = float(np.mean(qp_map))

    # Difference map (khuếch đại ×5 để dễ thấy)
    diff_amplified = np.clip(
        cv2.absdiff(img_baseline_rgb, img_adaptive_rgb).astype(np.float32) * 5, 0, 255
    ).astype(np.uint8)

    # Block classification từ Member B
    if block_size == 16:
        delta_matrix  = analyze_frame_adaptive(img_bgr, threshold=threshold)
        block_viz_rgb = cv2.cvtColor(
            debug_visualize_blocks(img_bgr, delta_matrix), cv2.COLOR_BGR2RGB
        )
    else:
        block_viz_rgb = img_rgb.copy()

# ── Part 1: Adaptive compression results ─────────────────────────────────────
st.header("Part 1 — Adaptive Compression Results")
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Original Frame (Raw)")
    st.image(img_rgb, use_container_width=True)
    st.info("Status: Uncompressed")

with c2:
    st.subheader("QP Map (Hybrid)")
    fig_qp, ax_qp = plt.subplots()
    im = ax_qp.imshow(qp_map, cmap='jet', vmin=1, vmax=51)
    ax_qp.axis('off')
    fig_qp.colorbar(im, ax=ax_qp, orientation='horizontal', label='QP Value')
    st.pyplot(fig_qp);  plt.close(fig_qp)
    st.caption("Red = high QP (textured, compress more) | Blue = low QP (flat, preserve quality)")

with c3:
    st.subheader("After Adaptive Compression (Proposed)")
    st.image(img_adaptive_rgb, use_container_width=True)
    st.success(f"Average QP: {avg_qp_adaptive:.1f}  (Baseline: {avg_qp_baseline:.1f})")

# ── Part 2: Before / After Visual Quality ────────────────────────────────────
st.write("---")
st.header("Part 2 — Before / After Visual Quality")
ca, cb, cc, cd = st.columns(4)

with ca:
    st.subheader("Baseline")
    st.image(img_baseline_rgb, use_container_width=True)
    st.caption(f"Fixed QP = {base_qp} | PSNR: {psnr_base:.2f} dB | SSIM: {ssim_base:.4f}")

with cb:
    st.subheader("Adaptive (Proposed)")
    st.image(img_adaptive_rgb, use_container_width=True)
    st.caption(f"Avg QP = {avg_qp_adaptive:.1f} | PSNR: {psnr_adapt:.2f} dB | SSIM: {ssim_adapt:.4f}")

with cc:
    st.subheader("Difference Map (×5)")
    st.image(diff_amplified, use_container_width=True)
    st.caption("Bright = high difference between methods. Dark = similar.")

with cd:
    st.subheader("Block Classification (Member B)")
    st.image(block_viz_rgb, use_container_width=True)
    st.caption("Green = flat → QP+3  |  Red = textured → QP−3")

# ── Part 3: Performance table + download ─────────────────────────────────────
st.write("---")
st.header("Part 3 — Performance Analysis Table")

if video_path is not None:
    with st.spinner("Running FFmpeg compression to measure actual bitrate..."):
        br_base, br_adapt, saving_pct, path_base, path_adapt = \
            run_real_compression_and_get_saving(video_path, base_qp, avg_qp_adaptive)
else:
    br_base, br_adapt, saving_pct, path_base, path_adapt = None, None, None, None, None

bitrate_base_str  = f"{br_base:.0f} kbps"  if br_base  is not None else "Upload video first"
bitrate_adapt_str = f"{br_adapt:.0f} kbps" if br_adapt is not None else "Upload video first"
saving_str        = f"{saving_pct:.1f}%"   if saving_pct is not None else "N/A"

df_metrics = pd.DataFrame({
    "Method":         ["Fixed compression (Baseline)", "Perceptual compression (Proposed)"],
    "Average QP":     [f"{avg_qp_baseline:.1f}", f"{avg_qp_adaptive:.1f}"],
    "Bitrate":        [bitrate_base_str, bitrate_adapt_str],
    "PSNR (dB)":      [f"{psnr_base:.2f}", f"{psnr_adapt:.2f}"],
    "SSIM":           [f"{ssim_base:.4f}", f"{ssim_adapt:.4f}"],
    "Bitrate savings":["—", saving_str],
})
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

# Download buttons & Video Playback
if path_base and path_adapt:
    st.write("### Play / Download compressed video (audio removed)")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.video(path_base) # Bổ sung trình phát video Baseline
        with open(path_base, "rb") as f:
            st.download_button(
                label=f"⬇Baseline video (QP={int(base_qp)})",
                data=f, file_name=f"baseline_qp{int(base_qp)}.mp4",
                mime="video/mp4", use_container_width=True
            )
    with col_dl2:
        st.video(path_adapt) # Bổ sung trình phát video Đề xuất
        with open(path_adapt, "rb") as f:
            st.download_button(
                label=f"⬇Adaptive video (QP≈{int(math.ceil(avg_qp_adaptive))})",
                data=f, file_name=f"adaptive_qp{int(math.ceil(avg_qp_adaptive))}.mp4",
                mime="video/mp4", use_container_width=True
            )
# ── Part 4: RD-Curve từ baseline_metrics.csv ─────────────────────────────────
st.write("---")
st.header("Part 4 — Rate-Distortion Curve (Measured Baseline)")

df_baseline = load_baseline_csv("results/baseline_metrics.csv")

if df_baseline is not None:
    st.success(f"Loaded {len(df_baseline)} rows from results/baseline_metrics.csv")
    col_rd1, col_rd2 = st.columns([2, 1])
    with col_rd1:
        fig_rd = plot_rd_curve(df_baseline)
        st.pyplot(fig_rd);  plt.close(fig_rd)
    with col_rd2:
        st.subheader("Raw data")
        st.dataframe(df_baseline, use_container_width=True, hide_index=True)
    st.markdown(
        "**Reading the chart:** Lower-left = better (same quality, less bitrate). "
        "The Proposed method targets the same PSNR as Baseline at a lower bitrate."
    )
else:
    st.warning("Warning: `results/baseline_metrics.csv` not found. Run `python run_baseline.py` to generate it.")
