import cv2
import numpy as np

def analyze_frame_adaptive(frame, threshold=100):
    # Lay chieu cao va chieu rong cua khung hinh
    height, width, _ = frame.shape
    
    # Kich thuoc khoi theo tieu chuan H.264
    block_size = 16
    
    # Tinh so luong khoi theo hang va cot (Ket qua phai la 45 hang, 80 cot)
    rows = height // block_size
    cols = width // block_size
    
    # Khoi tao ma tran ket qua ban dau toan so 0
    delta_qp_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Chuyen anh sang anh xam (Gray) de tinh toan do sang cho nhe va chinh xac
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    print(f"Variance trung binh cua anh: {np.var(gray_frame)}")
    
    # Duyet qua tung khoi tren luoi 45x80 bang 2 vong lap for
    for r in range(rows):
        for c in range(cols):
            # Tinh toan toa do pixel bat dau va ket thuc cua khoi 16x16 hien tai
            y_start = r * block_size
            y_end = y_start + block_size
            x_start = c * block_size
            x_end = x_start + block_size
            
            # Cat ma tran con 16x16 ra tu anh xam
            block = gray_frame[y_start:y_end, x_start:x_end]
            
            # Tinh phuong sai (Variance) cua 256 pixel trong khoi nay
            variance = np.var(block)
            
            # Thuat toan ra quyet dinh dua tren nguong threshold (T)
            if variance < threshold:
                # Phuong sai nho hon nguong -> Vung phang (Low Detail)
                # Tang QP len +3 de nen manh tay, tiet kiem dung luong
                delta_qp_matrix[r][c] = 3
            else:
                # Phuong sai lon hon hoac bang nguong -> Vung chi tiet (High Texture)
                # Giam QP xuong -3 de giu lai do net cho nhan vat/chi tiet
                delta_qp_matrix[r][c] = -3
                
    return delta_qp_matrix

def debug_visualize_blocks(frame, delta_qp_matrix):
    # Ham nay dung de ve truc quan hoa len anh giup ban kiem tra bang mat thuong
    visual_image = frame.copy()
    block_size = 16
    rows = len(delta_qp_matrix)
    cols = len(delta_qp_matrix[0])

    for r in range(rows):
        for c in range(cols):
            y_start = r * block_size
            y_end = y_start + block_size
            x_start = c * block_size
            x_end = x_start + block_size
            
            # Neu matrix la 3 (vung phang), ve khung mau xanh la cay
            if delta_qp_matrix[r][c] == +3:
                cv2.rectangle(visual_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
            # Neu matrix la -3 (vung chi tiet), ve khung mau do
            elif delta_qp_matrix[r][c] == -3:
                cv2.rectangle(visual_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)
                
    return visual_image

if __name__ == "__main__":
    # LAY MAU 1 KHUNG HINH DE THU NGHIEM
    # Do Bạn A chua lam xong bo doc video, ban tu viet doan nay de lay anh mau test truoc
    video_path = "data/vidyo1.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, test_frame = cap.read()
    cap.release()
    
    if not ret:
        print("Loi: Khong tim thay file video tai duong dan data/vidyo1.mp4")
    else:
        # Chay thu thuat toan voi nguong tu chon T = 100
        chosen_threshold = 100
        matrix = analyze_frame_adaptive(test_frame, threshold=chosen_threshold)
        
        # Kiem tra kich thuoc ma tran dau ra de dam bao khong lam loi code Ban A
        print(f"Kich thuoc ma tran dau ra: {len(matrix)} hang x {len(matrix[0])} cot")
        
        # Xuat anh kiem tra truc quan de kiem tra xem thuat toan chay dung chua
        result_img = debug_visualize_blocks(test_frame, matrix)
        cv2.imwrite("results/debug_classification.png", result_img)
        print("Da xuat anh kiem tra tai: results/debug_classification.png")