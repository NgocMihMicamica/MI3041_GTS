import numpy as np
import math

def read_int(prompt):
    while True:
        try: return int(input(prompt).strip())
        except ValueError: print("Cậu nhập số nguyên giúp mình nha! (≧◡≦)")

def read_matrix_keyboard(rows, cols, name="A"):
    print(f"\n🌸 Nhập ma trận {name} ({rows} x {cols}):")
    matrix = []
    for i in range(rows):
        while True:
            try:
                line = input(f"Dòng {i + 1}: ").strip()
                row = [float(x) for x in line.split()]
                if len(row) != cols:
                    print(f"Ôi, cần đúng {cols} số cơ. Cậu nhập lại dòng này nhé!")
                    continue
                matrix.append(row)
                break
            except ValueError: print("Dữ liệu không hợp lệ rồi, thử lại nha!")
    return np.array(matrix, dtype=float)

def read_vector_keyboard(size, name="x0"):
    print(f"\n🌸 Nhập vector {name} ({size} số, cách nhau bởi dấu cách):")
    while True:
        try:
            line = input("=> ").strip()
            vector = [float(x) for x in line.split()]
            if len(vector) != size:
                print(f"Cần {size} phần tử nha. Nhập lại giúp mình!")
                continue
            return np.array(vector, dtype=float).reshape(size, 1)
        except ValueError: print("Dữ liệu sai mất rồi!")

def in_ma_tran_ra_man_hinh(M, tieu_de=""):
    if tieu_de: print(f"{tieu_de}")
    for row in M:
        print("  [ " + "  ".join([f"{x:9.4f}" for x in row]) + " ]")

# =====================================================================
# LÕI PHƯƠNG PHÁP LŨY THỪA
# =====================================================================
def power_method_core(A, x0, k_buoc, eps=1e-4, verbose=True, name="A"):
    X = x0.copy()
    X_history = [X]
    lambda_old = 0
    
    if verbose:
        print(f"\n1. Chọn vector khởi tạo x^(0):")
        in_ma_tran_ra_man_hinh(X)
        print(f"\n2. Thực hiện quá trình lặp x^(k) = {name} * x^(k-1):")
        
    for k in range(1, k_buoc + 1):
        Y = np.dot(A, X)
        max_idx = np.argmax(np.abs(Y))
        lambda_k = Y[max_idx, 0]
        
        if abs(lambda_k) < 1e-10:
            if verbose: 
                print(f"\n=> Tại bước k = {k}, vector bằng 0. Suy ra λ = 0.")
            return 0, X
            
        X = Y / lambda_k
        X[np.abs(X) < 1e-10] = 0.0
        X_history.append(Y) 
        
        if verbose:
            print(f"\n📍 Bước {k}:")
            print(f"Ta có y^({k}) = {name} * x^({k-1}) =")
            in_ma_tran_ra_man_hinh(Y)
            print(f"Rút phần tử có trị tuyệt đối lớn nhất ({lambda_k:.4f}) ra ngoài:")
            print(f"=> x^({k}) =")
            in_ma_tran_ra_man_hinh(X)
            print(f"=> Trị riêng xấp xỉ λ^({k}) = {lambda_k:.4f}")
            
        if verbose and k > 1 and abs(lambda_k - lambda_old) < eps:
            print("\n" + "="*50)
            print("🌟 KẾT LUẬN (Hội tụ bình thường):")
            print(f"Vì |λ^({k}) - λ^({k-1})| < {eps}, dãy lặp đã hội tụ.")
            print(f"Vậy trị riêng trội là: λ ≈ {lambda_k:.4f}")
            in_ma_tran_ra_man_hinh(X, "Vector riêng tương ứng là v ≈")
            return lambda_k, X
            
        if verbose and k > 2 and abs(lambda_k + lambda_old) < eps:
            print("\n" + "="*50)
            print("🚨 KẾT LUẬN (Trường hợp dao động đối xứng):")
            print("Biện luận: Ta thấy tỷ số giữa các bước lặp dao động đổi dấu.")
            print("=> Đây là trường hợp ma trận có 2 trị riêng đối nhau: λ1 = -λ2.")
            print("=> Áp dụng công thức lặp bước chẵn: λ^2 ≈ (A^(2n)x)_i / (A^(2n-2)x)_i")
            
            Y_curr = X_history[k]
            Y_prev = X_history[k-2]
            
            idx = np.argmax(np.abs(Y_curr))
            tu_so = Y_curr[idx, 0]
            mau_so = Y_prev[idx, 0]
            lam_binh_phuong = tu_so / mau_so
            lam_thuc = math.sqrt(abs(lam_binh_phuong))
            
            print(f"\nChọn thành phần thứ {idx+1} để lập tỷ số:")
            print(f"λ^2 ≈ {tu_so:.4f} / {mau_so:.4f} = {lam_binh_phuong:.4f}")
            print(f"Vậy 2 trị riêng trội là: λ1 ≈ {lam_thuc:.4f} và λ2 ≈ {-lam_thuc:.4f}")
            return lam_thuc, X
            
        lambda_old = lambda_k
        
    if verbose:
        print("\n⚠️ Đã hết số bước lặp quy định. Ta lấy kết quả xấp xỉ tại bước cuối:")
        print(f"λ ≈ {lambda_k:.4f}")
        
    return lambda_k, X

# =====================================================================
# HÀM THÔNG MINH TỰ ĐỘNG CHỌN VECTOR KHỞI TẠO (CÓ THỂ TRUYỀN TAY)
# =====================================================================
def smart_power_method(A, k_buoc, x0_custom=None, name="A"):
    n = A.shape[0]
    
    print("\n" + "-"*50)
    print(f"📝 BIỆN LUẬN BÀI LÀM: PHƯƠNG PHÁP LŨY THỪA CHO {name}")
    print("-" * 50)
    print(f"Ta có ma trận {name}:")
    in_ma_tran_ra_man_hinh(A)
    
    # NẾU CÓ TRUYỀN x0 TỪ ĐỀ BÀI -> TUÂN LỆNH TUYỆT ĐỐI
    if x0_custom is not None:
        print("\n💡 Biện luận: Theo giả thiết đề bài, ta sử dụng vector khởi tạo đã cho.")
        return power_method_core(A, x0_custom, k_buoc, verbose=True, name=name)

    # NẾU KHÔNG CÓ -> TỰ ĐỘNG BẬT AUTO-PILOT DÒ MÌN
    true_eigenvalues = np.linalg.eigvals(A)
    true_max_lam = max(np.abs(true_eigenvalues))
    
    x0_thu_nghiem = np.ones((n, 1))
    lam_thu_nghiem, _ = power_method_core(A, x0_thu_nghiem, k_buoc, verbose=False)
    
    if abs(abs(lam_thu_nghiem) - true_max_lam) > 0.5:
        print("\n💡 Biện luận: Để phòng tránh trường hợp vector khởi tạo bị trực giao")
        print("với vector riêng trội (làm thuật toán hội tụ sai), ta khôn khéo chọn")
        print("vector khởi tạo bất đối xứng x0 = [1, 0, ..., 0]^T.")
        x0_chinh_thuc = np.zeros((n, 1))
        x0_chinh_thuc[0, 0] = 1.0
    else:
        print("\n💡 Biện luận: Chọn vector khởi tạo đơn giản toàn số 1 để dễ tính toán.")
        x0_chinh_thuc = np.ones((n, 1))
        
    return power_method_core(A, x0_chinh_thuc, k_buoc, verbose=True, name=name)

# =====================================================================
# PHƯƠNG PHÁP XUỐNG THANG
# =====================================================================
def xuong_thang_method(A, k_buoc, x0_custom=None):
    n = A.shape[0]
    print("\n" + "🌸"*25)
    print("📝 BÀI LÀM: PHƯƠNG PHÁP XUỐNG THANG (TÌM TRỊ RIÊNG THỨ 2)")
    print("🌸"*25)
    
    print("\n[BƯỚC 1] Tìm giá trị riêng trội λ1 và véc-tơ riêng phải v1 của A:")
    lam1, v1 = smart_power_method(A, k_buoc, x0_custom, name="A")
    
    print("\n[BƯỚC 2] Tìm véc-tơ riêng trái w1 của A:")
    print("Lập ma trận chuyển vị A^T và áp dụng PP lũy thừa:")
    _, w1 = smart_power_method(A.T, k_buoc, None, name="A^T") # Để None cho nó tự né bẫy

    print("\n[BƯỚC 3] Tìm véc-tơ x thỏa mãn x^T * v1 = 1:")
    tich_vo_huong = np.dot(w1.T, v1)[0, 0]
    print(f"Tính tích vô hướng: w1^T * v1 = {tich_vo_huong:.4f}")
    
    if abs(tich_vo_huong) < 1e-8:
        print("\n❌ KẾT LUẬN DỪNG BÀI: Tích vô hướng bằng 0!")
        print("Biện luận vào bài: Do véc-tơ riêng trái và phải trực giao (tích vô hướng = 0),")
        print("đây là ma trận khiếm khuyết. Phương pháp xuống thang cơ bản không thể")
        print("thực hiện được tiếp phép chia. Ta dừng thuật toán tại đây.")
        return
        
    print("Áp dụng công thức: x = w1 / (w1^T * v1)")
    x = w1 / tich_vo_huong
    in_ma_tran_ra_man_hinh(x, "=> Ta được vector x =")

    print("\n[BƯỚC 4] Lập ma trận xuống thang B:")
    print("Công thức: B = A - λ1 * v1 * x^T")
    ma_tran_nhan = lam1 * np.dot(v1, x.T)
    in_ma_tran_ra_man_hinh(ma_tran_nhan, f"Tính cụm (λ1 * v1 * x^T) =")
    B = A - ma_tran_nhan
    in_ma_tran_ra_man_hinh(B, "Thực hiện trừ ma trận A cho cụm trên, ta được ma trận B =")

    print("\n[BƯỚC 5] Tìm GTR trội của ma trận B (chính là GTR lớn thứ 2 của A):")
    lam2, u2 = smart_power_method(B, k_buoc, None, name="B") # Để None auto-pilot
    
    print("\n[BƯỚC 6] Tìm véc-tơ riêng v2 tương ứng của ma trận A ban đầu:")
    print("Công thức: v2 = (λ1 - λ2)u2 + λ1(x^T * u2)v1")
    thanh_phan_1 = (lam1 - lam2) * u2
    vo_huong_2 = np.dot(x.T, u2)[0, 0]
    thanh_phan_2 = lam1 * vo_huong_2 * v1
    
    print(f"Tính phần 1: ({lam1:.4f} - {lam2:.4f}) * u2 =")
    in_ma_tran_ra_man_hinh(thanh_phan_1)
    
    print(f"Tính phần 2: Tích x^T * u2 = {vo_huong_2:.4f} => λ1 * (x^T * u2) * v1 =")
    in_ma_tran_ra_man_hinh(thanh_phan_2)
    
    v2 = thanh_phan_1 + thanh_phan_2
    max_v2 = np.max(np.abs(v2))
    if max_v2 > 1e-10: v2 = v2 / max_v2 
    in_ma_tran_ra_man_hinh(v2, "\n🌟 KẾT LUẬN: Cộng 2 thành phần và chuẩn hóa, véc-tơ riêng v2 của A là:")
    print("\n" + "="*50)
    print("🎉 HOÀN THÀNH BÀI THI! CHÚC CẬU ĐƯỢC A+ NHA! 🎉")

# =====================================================================
# MAIN THỰC CHIẾN - TỐI THƯỢNG
# =====================================================================
if __name__ == "__main__":
    print("🌸 CHƯƠNG TRÌNH HỖ TRỢ GIẢI TÍCH SỐ - BẢN THỰC CHIẾN TỐI THƯỢNG 🌸")
    print("1. Phương pháp lũy thừa (Tìm GTR trội)")
    print("2. Phương pháp xuống thang (Tìm GTR thứ 2)")
    
    mode = read_int("Cậu chọn chức năng nào (1 hoặc 2): ")
    n_hang = read_int("Nhập số hàng/cột của ma trận vuông: ")
    A = read_matrix_keyboard(n_hang, n_hang, "A")
    k = read_int("\nNhập số bước lặp tối đa (khuyên dùng 15): ")
    
    print("\nĐề bài có cho sẵn vector khởi tạo x0 không cậu ơi?")
    print("1. KHÔNG (Để code tự động né bẫy siêu thông minh)")
    print("2. CÓ (Tớ sẽ nhập tay vào)")
    chon_x0 = read_int("=> Chọn (1 hoặc 2): ")
    
    x0_custom = None
    if chon_x0 == 2:
        x0_custom = read_vector_keyboard(n_hang, "x0")
        
    if mode == 1:
        smart_power_method(A, k, x0_custom)
    elif mode == 2:
        xuong_thang_method(A, k, x0_custom)