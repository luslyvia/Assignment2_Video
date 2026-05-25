# HƯỚNG DẪN CHẠY AS2
## 1. Cài đặt môi trường
Mở Terminal/Command Prompt tại thư mục dự án và chạy lệnh:
pip install opencv-python numpy streamlit matplotlib scikit-image

## 2. Khởi chạy hệ thống Live Demo
Gõ lệnh sau để mở giao diện Web tương tác trên trình duyệt:
streamlit run main.py

## 3. Cách tái lập kết quả thực nghiệm
- Bước 1: Truy cập vào giao diện Localhost hiển thị trên màn hình terminal.
- Bước 2: Tại thanh Sidebar bên trái, nhấn "Browse files" để tải lên ảnh trong thư mục dữ liệu mẫu.
- Bước 3: Điều chỉnh thanh trượt 'Base QP' và 'Perceptual Sensitivity' để quan sát sự thay đổi trực quan của bản đồ nhiệt QP Map và bảng so sánh số liệu PSNR/SSIM ở phía dưới.
