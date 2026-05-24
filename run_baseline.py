import os
import csv
import subprocess
import re
import cv2

# --- CÁC HÀM CÔNG CỤ XỬ LÝ THẬT ---

def get_video_duration(video_path):
    """Dùng OpenCV để lấy thời lượng video (giây) phục vụ tính Bitrate"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps > 0 else 1.0

def calculate_psnr(original_video, compressed_video):
    """Gọi FFmpeg để đo chỉ số chất lượng PSNR giữa 2 video"""
    # Lệnh FFmpeg so sánh 2 video và xuất ra PSNR
    cmd = [
        'ffmpeg', 
        '-i', compressed_video, 
        '-i', original_video, 
        '-lavfi', 'psnr', 
        '-f', 'null', '-'
    ]
    
    # Chạy lệnh và bắt kết quả text xuất ra Terminal
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    
    # Dùng biểu thức chính quy (Regex) để tìm con số trung bình (average:XX.XX)
    match = re.search(r'average:([0-9.]+)', result.stderr)
    if match:
        return float(match.group(1))
    return 0.0

# --- LUỒNG CHẠY CHÍNH ---

if __name__ == "__main__":
    os.makedirs("results/baseline_videos", exist_ok=True)
    
    videos = ["data/ducks_take_off.mp4", "data/old_town_cross.mp4", "data/vidyo1.mp4"]
    qps = [22, 27, 32, 37]
    csv_path = "results/baseline_metrics.csv"

    print("=== BẮT ĐẦU CHẠY BASELINE ===")
    print("Lưu ý: Quá trình này có thể mất từ 10 - 20 phút tùy cấu hình máy...\n")

    # Kiểm tra file tồn tại
    missing = [v for v in videos if not os.path.exists(v)]
    if missing:
        print("LỖI: Thiếu file video gốc trong thư mục data/:", missing)
        exit()

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "QP", "Bitrate (kbps)", "PSNR (dB)"])
        
        for video in videos:
            video_name = os.path.basename(video)
            duration = get_video_duration(video)
            print(f"\n[+] Đang xử lý video: {video_name} (Thời lượng: {duration:.2f}s)")
            
            for qp in qps:
                print(f"   -> Đang nén với QP = {qp}...", end="", flush=True)
                
                # Tạo tên file đầu ra
                output_video = f"results/baseline_videos/{video_name.replace('.mp4', '')}_qp{qp}.mp4"
                
                # 1. GỌI FFMPEG ĐỂ NÉN VIDEO THẬT
                compress_cmd = [
                    'ffmpeg', '-y', 
                    '-i', video, 
                    '-c:v', 'libx264', 
                    '-qp', str(qp), 
                    output_video
                ]
                subprocess.run(compress_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 2. TÍNH BITRATE THẬT (Dung lượng chia thời gian)
                file_size_bytes = os.path.getsize(output_video)
                bitrate_kbps = (file_size_bytes * 8) / duration / 1000
                
                # 3. ĐO PSNR THẬT BẰNG FFMPEG
                psnr_val = calculate_psnr(video, output_video)
                
                # Ghi số liệu thật vào Excel
                writer.writerow([video_name, qp, round(bitrate_kbps, 2), round(psnr_val, 2)])
                print(f" Xong! (Bitrate: {bitrate_kbps:.0f} kbps | PSNR: {psnr_val:.2f} dB)")
                
    print(f"\nHOÀN THÀNH! Số liệu đã được lưu tại: {csv_path}")