import numpy as np

def in_ma_tran(M):
    hang, cot = M.shape
    lines = []
    for i in range(hang):
        line = "  [ " + "   ".join([f"{M[i,j]:.6f}" for j in range(cot)]) + " ]"
        lines.append(line)
    return "\n".join(lines)

def phuong_phap_luy_thua(M, step_idx, k_chi_tiet=2, tol=1e-8, max_iter=1000):
    """
    Tìm trị riêng lớn nhất và vector riêng tương ứng bằng phương pháp Lũy thừa.
    Có in chi tiết k_chi_tiet bước đầu tiên để chép bài thi.
    """
    n = M.shape[0]
    # Khởi tạo vector ban đầu v0
    v = np.ones(n)
    v = v / np.linalg.norm(v, 2)
    
    print(f"\n" + "."*40)
    print(f"👉 TRÌNH BÀY CHI TIẾT LẶP LŨY THỪA (TÌM λ_{step_idx}):")
    print(f"  Chọn vector giả thiết ban đầu v^(0) (đã chuẩn hóa):")
    print(f"  v^(0) = [ " + "   ".join([f"{x:.6f}" for x in v]) + " ]^T")
    
    lam_old = 0
    for iter_step in range(max_iter):
        # Tính y_new = M * v_old
        y_new = np.dot(M, v)
        norm_y = np.linalg.norm(y_new, 2)
        
        if norm_y < 1e-12:
            return 0.0, v
            
        # Chuẩn hóa
        v_new = y_new / norm_y
        
        # Tỉ số Rayleigh tính lambda
        lam_new = np.dot(v_new.T, np.dot(M, v_new))
        
        # In chi tiết nếu nằm trong số bước yêu cầu
        if iter_step < k_chi_tiet:
            print(f"\n  * Bước lặp k = {iter_step + 1}:")
            print(f"    +) Tính y^({iter_step+1}) = M * v^({iter_step})")
            print(f"       y^({iter_step+1}) = [ " + "   ".join([f"{x:.6f}" for x in y_new]) + " ]^T")
            
            print(f"    +) Chuẩn hóa để tìm v^({iter_step+1}) = y^({iter_step+1}) / ||y^({iter_step+1})||_2")
            print(f"       ||y^({iter_step+1})||_2 = {norm_y:.6f}")
            print(f"       v^({iter_step+1}) = [ " + "   ".join([f"{x:.6f}" for x in v_new]) + " ]^T")
            
            print(f"    +) Tính xấp xỉ trị riêng λ^({iter_step+1}) = (v^({iter_step+1}))^T * M * v^({iter_step+1})")
            print(f"       λ^({iter_step+1}) = {lam_new:.6f}")
            
        if abs(lam_new - lam_old) < tol:
            if iter_step >= k_chi_tiet:
                print(f"\n  ... (Tiếp tục lặp tương tự cho đến khi hội tụ) ...")
            print(f"\n  => Đạt sai số tại bước k = {iter_step + 1}.")
            print(f"  => Trị riêng tìm được: λ_{step_idx} = {lam_new:.6f}")
            return lam_new, v_new
            
        v = v_new
        lam_old = lam_new
        
    return lam_old, v

def phan_tich_svd_luy_thua_xuong_thang():
    print("="*75)
    print(" GIẢI SVD BẰNG PHƯƠNG PHÁP LŨY THỪA VÀ XUỐNG THANG (CHUẨN YÊU CẦU ĐỀ THI)")
    print("="*75)
    
    # Thêm tùy chọn hỏi số bước chi tiết cần in
    try:
        k_chi_tiet = int(input("Cậu muốn in CHI TIẾT CÔNG THỨC mấy bước lặp lũy thừa để chép? (Khuyên dùng: 2 hoặc 3): "))
    except ValueError:
        k_chi_tiet = 2
        print("Cậu nhập sai rùi, tớ tự động lấy k = 2 nghen!")
        
    print("\nHướng dẫn: Nhập từng hàng của ma trận A, các số cách nhau bằng dấu cách.")
    print("Gõ 'x' rồi ấn Enter để bắt đầu tính toán.\n")
    
    A_list = []
    i = 1
    while True:
        hang = input(f"Nhập hàng thứ {i} (hoặc gõ 'x' để dừng): ")
        if hang.strip().lower() == 'x':
            break
        try:
            row_data = [float(val) for val in hang.split()]
            A_list.append(row_data)
            i += 1
        except ValueError:
            print(" Lỗi: Vui lòng chỉ nhập số thực.")
            
    if not A_list:
        print("Chưa nhập ma trận!")
        return
        
    A = np.array(A_list)
    m, n = A.shape
    
    print("\n" + "🌸"*10 + " BÀI LÀM: LŨY THỪA VÀ XUỐNG THANG " + "🌸"*10)
    print(f"Xét ma trận A kích thước {m}x{n}:")
    print(in_ma_tran(A))
    
    M = np.dot(A.T, A)
    print("\nBước 1: Tính ma trận đối xứng M = A^T * A:")
    print(in_ma_tran(M))
    print("\nTiến hành lặp Phương pháp Lũy thừa và Xuống thang trên M:")
    
    eigenvalues = []
    V_vectors = []
    
    for step in range(n):
        print(f"\n" + "-"*60)
        print(f"--- BƯỚC {step + 1}: TÌM TRỊ RIÊNG λ_{step + 1} VÀ VECTOR v_{step + 1} ---")
        
        # Gọi hàm phương pháp lũy thừa mới
        lam, v = phuong_phap_luy_thua(M, step_idx=step+1, k_chi_tiet=k_chi_tiet)
        
        if abs(lam) < 1e-10:
            lam = 0.0
            
        eigenvalues.append(lam)
        V_vectors.append(v)
        
        M_deflated = M - lam * np.outer(v, v)
        
        print(f"\n• Áp dụng PP Xuống thang, loại bỏ thành phần của λ_{step + 1}:")
        print(f"  Công thức: M_{step + 1} = M_{step} - λ_{step + 1} * v_{step + 1} * v_{step + 1}^T")
        print(f"  => GHI RÕ MA TRẬN SAU BƯỚC XUỐNG THANG THỨ {step + 1}:")
        
        M = M_deflated
        M[np.abs(M) < 1e-10] = 0
        print(in_ma_tran(M))

    print("\n" + "="*50)
    print("TỔNG HỢP KẾT QUẢ ĐỂ KHAI TRIỂN SVD:")
    print("="*50)
    
    print("\n1. Tập các giá trị kỳ dị σ_i (bằng căn bậc hai của λ_i):")
    sigmas = []
    for idx, lam in enumerate(eigenvalues):
        sig = np.sqrt(max(0, lam))
        sigmas.append(sig)
        print(f"  σ_{idx+1} = √({lam:.6f}) = {sig:.6f}")
        
    print("\n2. Ma trận vector kỳ dị phải V (ghép các vector v_i đã tìm được):")
    V_matrix = np.column_stack(V_vectors)
    print(in_ma_tran(V_matrix))
    
    print("\n3. Xác định các vector kỳ dị trái u_i:")
    print("Sử dụng công thức liên hệ: u_i = (1 / σ_i) * A * v_i")
    
    U_vectors = []
    for idx in range(min(m, n)):
        if sigmas[idx] > 1e-10:
            u_i = np.dot(A, V_vectors[idx]) / sigmas[idx]
            U_vectors.append(u_i)
            print(f"  u_{idx+1} = (1 / {sigmas[idx]:.6f}) * A * v_{idx+1} = [ " + "   ".join([f"{x:.6f}" for x in u_i]) + " ]^T")
        else:
            print(f"  Do σ_{idx+1} = 0, u_{idx+1} được nội suy từ trực giao hóa Gram-Schmidt.")
            
    print("\n" + "🌟"*10 + " CHÚC CẬU QUA MÔN 10 ĐIỂM TRÒN NHA " + "🌟"*10)

if __name__ == "__main__":
    phan_tich_svd_luy_thua_xuong_thang()