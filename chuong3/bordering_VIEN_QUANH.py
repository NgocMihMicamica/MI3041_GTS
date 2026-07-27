import numpy as np

def read_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("Vui lòng nhập một số nguyên hợp lệ!")

def read_matrix(n, m, name):
    print(f"--- Nhập ma trận {name} ({n}x{m}) ---")
    matrix = []
    for i in range(n):
        while True:
            try:
                # Lọc dấu trừ PDF chống lỗi copy
                raw_input = input(f"Nhập hàng {i+1} (các số cách nhau khoảng trắng): ")
                clean_input = raw_input.replace('−', '-').replace('–', '-')
                row = list(map(float, clean_input.split()))
                
                if len(row) != m:
                    print(f"Hàng phải có đúng {m} phần tử. Vui lòng nhập lại!")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Dữ liệu nhập vào không hợp lệ. Vui lòng nhập lại!")
    return np.array(matrix, dtype=float)

def read_vector(n, name):
    print(f"--- Nhập vectơ cột {name} (kích thước {n}) ---")
    print("(Bạn có thể copy dán toàn bộ các số trên 1 dòng, HOẶC copy cột dọc đều được nhé!)")
    vector = []
    while len(vector) < n:
        try:
            raw_input = input()
            clean_input = raw_input.replace('−', '-').replace('–', '-')
            
            if not clean_input.strip():
                continue
                
            parts = list(map(float, clean_input.split()))
            vector.extend(parts)
            
        except ValueError:
            print("Dữ liệu có chứa ký tự lạ. Vui lòng nhập lại các số còn thiếu!")
            
    if len(vector) > n:
        print(f"(Lưu ý: Bạn nhập thừa dữ liệu. Chương trình sẽ tự động chỉ lấy {n} phần tử đầu tiên.)")
        vector = vector[:n]
        
    return np.array(vector, dtype=float).reshape(n, 1)

def bieu_dien_ma_tran(matrix, ten=""):
    """Hàm in ma trận trực quan để chép vào bài thi"""
    if ten:
        print(f"{ten} = ")
    for row in matrix:
        print("  [ " + " ".join([f"{x:9.4f}" for x in row]) + " ]")
    print()

def bordering_inverse_verbose(A_input):
    n = A_input.shape[0]
    A = A_input.copy() # Copy để không làm thay đổi ma trận gốc
    swaps = [] # Danh sách lưu lịch sử hoán vị hàng
    
    print("\n================ BẮT ĐẦU QUÁ TRÌNH VIỀN QUANH ================")
    
    # BƯỚC 1: XỬ LÝ CẤP 1X1
    print("--- Bước 1: Xét ma trận con cấp 1x1 ---")
    if abs(A[0, 0]) < 1e-9:
        print("CẢNH BÁO: Phần tử A[0,0] = 0. Tiến hành tìm hàng để hoán vị...")
        swap_ok = False
        for j in range(1, n):
            if abs(A[j, 0]) > 1e-9:
                A[[0, j]] = A[[j, 0]]
                swaps.append((0, j))
                print(f"-> Đã hoán vị HÀNG 1 và HÀNG {j+1} của ma trận gốc.")
                swap_ok = True
                break
        if not swap_ok:
            print("LỖI: Toàn bộ cột 1 bằng 0. Ma trận suy biến, không thể đảo!")
            return None
            
    A_k_inv = np.array([[1.0 / A[0, 0]]])
    bieu_dien_ma_tran(np.array([[A[0, 0]]]), "A_1")
    bieu_dien_ma_tran(A_k_inv, "A_1^-1")
    
    # BƯỚC 2: TĂNG DẦN CẤP ĐỘ
    for k in range(1, n):
        print(f"--- Bước {k+1}: Viền quanh từ cấp {k} lên cấp {k+1} ---")
        
        u_k = A[0:k, k:k+1]
        v_k = A[k:k+1, 0:k]
        alpha_k = A[k, k]
        
        # Tính thử beta_k xem có bị bằng 0 không
        A_inv_u = np.dot(A_k_inv, u_k)
        beta_k = alpha_k - np.dot(v_k, A_inv_u)[0, 0]
        
        # HOÁN VỊ NẾU BETA_K = 0
        if abs(beta_k) < 1e-9:
            print(f"CẢNH BÁO: beta_{k} = 0 (Định thức con bằng 0). Cần hoán vị hàng!")
            swap_ok = False
            for j in range(k+1, n):
                v_k_thu = A[j:j+1, 0:k]
                alpha_k_thu = A[j, k]
                beta_k_thu = alpha_k_thu - np.dot(v_k_thu, A_inv_u)[0, 0]
                
                if abs(beta_k_thu) > 1e-9:
                    # Đổi chỗ trong ma trận A
                    A[[k, j]] = A[[j, k]]
                    swaps.append((k, j))
                    print(f"-> Đã hoán vị HÀNG {k+1} và HÀNG {j+1}. Tính lại các đại lượng...")
                    # Cập nhật lại
                    v_k = A[k:k+1, 0:k]
                    alpha_k = A[k, k]
                    beta_k = beta_k_thu
                    swap_ok = True
                    break
            
            if not swap_ok:
                print("LỖI: Không tìm được hàng phù hợp để hoán vị. Ma trận suy biến!")
                return None
                
        bieu_dien_ma_tran(u_k, f"Vectơ cột u_{k}")
        bieu_dien_ma_tran(v_k, f"Vectơ hàng v_{k}")
        print(f"Phần tử góc alpha_{k} = {alpha_k:.4f}")
        
        v_A_inv = np.dot(v_k, A_k_inv)
        
        print(f"Tính beta_{k} = alpha_{k} - v_{k} * A_{k}^-1 * u_{k}")
        print(f"=> beta_{k} = {alpha_k:.4f} - {np.dot(v_k, A_inv_u)[0, 0]:.4f} = {beta_k:.4f}\n")
        
        # Ráp ma trận cấp k+1
        top_left = A_k_inv + (1.0 / beta_k) * np.dot(A_inv_u, v_A_inv)
        top_right = - (1.0 / beta_k) * A_inv_u
        bottom_left = - (1.0 / beta_k) * v_A_inv
        bottom_right = np.array([[1.0 / beta_k]])
        
        A_k_inv = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])
        
        bieu_dien_ma_tran(A_k_inv, f"Ma trận nghịch đảo cấp {k+1} (A_{k+1}^-1)")
        
    # BƯỚC 3: PHỤC HỒI HOÁN VỊ (ĐỔI CỘT)
    if swaps:
        print("--- Bước Cuối: Phục hồi hoán vị ---")
        print("Vì lúc đầu ta đã hoán vị hàng của A, giờ phải hoán vị CỘT của ma trận kết quả để ra A^-1 chuẩn.")
        for i, j in reversed(swaps):
            print(f"-> Hoán vị CỘT {i+1} và CỘT {j+1}")
            A_k_inv[:, [i, j]] = A_k_inv[:, [j, i]]
        bieu_dien_ma_tran(A_k_inv, "Ma trận nghịch đảo CUỐI CÙNG A^-1")
        
    print("================ KẾT THÚC QUÁ TRÌNH VIỀN QUANH ================\n")
    return A_k_inv

def solve_bordering(A, b):
    A_inv = bordering_inverse_verbose(A)
    if A_inv is not None:
        print("--- Giải hệ phương trình Ax = b ---")
        print("Công thức: x = A^-1 * b")
        x = np.dot(A_inv, b)
        bieu_dien_ma_tran(x, "Nghiệm của hệ phương trình x")
        return x
    else:
        print("Không thể giải hệ phương trình do không tìm được ma trận nghịch đảo.")
        return None

if __name__ == "__main__":
    print("=== PHUONG PHAP VIEN QUANH (BẢN FINAL VÔ ĐỊCH) ===")
    print("1. Chi tim ma tran nghich dao A^-1")
    print("2. Dung vien quanh de giai he Ax = b")
    print("Lưu ý: Chương trình bao trọn mọi Test Case khó nhất!")
    
    mode = read_int("Chon che do (1 hoặc 2): ")
    n = read_int("Nhap cap ma tran vuong n: ")
    A = read_matrix(n, n, "A")
    
    if mode == 1:
        bordering_inverse_verbose(A)
    elif mode == 2:
        b = read_vector(n, "b")
        solve_bordering(A, b)
    else:
        print("Lua chon khong hop le.")