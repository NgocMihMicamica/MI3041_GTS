import numpy as np

def read_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("Vui lòng nhập số nguyên hợp lệ.")

def read_matrix_interactive(rows, cols, name="A"):
    print(f"--- Nhập ma trận {name} ({rows}x{cols}) ---")
    matrix = []
    for i in range(rows):
        while True:
            try:
                line = input(f"Nhập hàng {i+1} (cách nhau khoảng trắng): ").replace('−', '-').replace('–', '-').split()
                if len(line) != cols:
                    print(f"Vui lòng nhập đúng {cols} phần tử!")
                    continue
                matrix.append([float(x) for x in line])
                break
            except ValueError:
                print("Dữ liệu không hợp lệ!")
    return np.array(matrix, dtype=float)

def read_vector_interactive(size, name="b"):
    print(f"--- Nhập vectơ {name} ({size} phần tử) ---")
    vector = []
    while len(vector) < size:
        try:
            raw_input = input()
            clean_input = raw_input.replace('−', '-').replace('–', '-')
            if not clean_input.strip(): continue
            parts = list(map(float, clean_input.split()))
            vector.extend(parts)
        except ValueError:
            print("Dữ liệu chứa ký tự lạ. Nhập lại!")
    return np.array(vector[:size], dtype=float)

def kiem_tra_cheo_troi_hang(A):
    n = len(A)
    for i in range(n):
        tong_hang = sum(abs(A[i, j]) for j in range(n) if i != j)
        if abs(A[i, i]) <= tong_hang:
            return False
    return True

def bieu_dien_ma_tran(matrix, ten=""):
    if ten: print(f"{ten} = ")
    for row in matrix:
        print("  [ " + " ".join([f"{x:9.4f}" for x in row]) + " ]")
    print()

def bieu_dien_vector(vector, ten=""):
    if ten: print(f"{ten} = ")
    print("  [ " + " ".join([f"{x:9.4f}" for x in vector]) + " ]^T\n")

def jacobi_verbose_fixed(A, b, k_buoc):
    n = len(b)
    
    # Khởi tạo X^(0)
    x_old = np.zeros(n)
    x_new = np.zeros(n)
    
    print("\n================ BẮT ĐẦU QUÁ TRÌNH LẶP JACOBI ================")
    print(f"Giả thiết ban đầu: X^(0) = {list(np.round(x_old, 4))}")
    
    for buoc in range(1, k_buoc + 1):
        print(f"\n--- Bước lặp k = {buoc} ---")
        for i in range(n):
            if A[i, i] == 0:
                print(f"🚨 LỖI: A[{i+1},{i+1}] = 0, không thể chia cho 0.")
                return
            
            # Khởi tạo chuỗi trình bày
            tong_giatri = b[i]
            chuoi_tru = []
            
            for j in range(n):
                if i != j:
                    tong_giatri -= A[i, j] * x_old[j]
                    dau = "+" if -A[i, j] > 0 else "-"
                    gia_tri_A = abs(A[i, j])
                    chuoi_tru.append(f"{dau} {gia_tri_A:.4f}*x_{j+1}^({buoc-1})")
            
            x_new[i] = tong_giatri / A[i, i]
            
            # Trình bày siêu sạch ra màn hình
            chuoi_bieu_thuc = f"{b[i]:.4f} " + " ".join(chuoi_tru)
            print(f"  x_{i+1}^({buoc}) = (1/{A[i,i]:.4f}) * [ {chuoi_bieu_thuc} ] = {x_new[i]:.4f}")
            
        # Kết thúc 1 bước lặp, cập nhật lại x_old bằng x_new
        # ĐÂY LÀ SỰ KHÁC BIỆT VỚI SEIDEL: Cập nhật ĐỒNG THỜI sau khi tính xong cả cụm
        x_old = x_new.copy()
        
        arr_str = ", ".join([f"{val:.4f}" for val in x_new])
        print(f"==> Kết luận k={buoc}: X^({buoc}) = [{arr_str}]^T")

    print("\n================ KẾT THÚC ================\n")

if __name__ == "__main__":
    print("=== PHƯƠNG PHÁP LẶP JACOBI (BẢN FINAL CHỐNG NERF) ===")
    n = read_int("Nhập hàng/cột n của hệ phương trình: ")
    k_buoc = read_int("Nhập số lần lặp k yêu cầu: ")
    
    A = read_matrix_interactive(n, n, "A")
    b = read_vector_interactive(n, "b")
    
    print("\n=== LẬP LUẬN ĐI THI KHÂU ĐẦU VÀO ===")
    if kiem_tra_cheo_troi_hang(A):
        print("Nhận xét: Ma trận A CHÉO TRỘI HÀNG. Chắc chắn hội tụ!")
        jacobi_verbose_fixed(A, b, k_buoc)
    else:
        print("Nhận xét: Ma trận A CHƯA CHÍNH TẮC (chưa chéo trội). Có nguy cơ phân kỳ!")
        print("=> XỬ LÝ: Nhân cả 2 vế với ma trận chuyển vị A^T để đảm bảo hội tụ an toàn.\n")
        
        A_new = np.dot(A.T, A)
        b_new = np.dot(A.T, b)
        
        print("--- HỆ PHƯƠNG TRÌNH MỚI (A^T * A)x = A^T * b ---")
        bieu_dien_ma_tran(A_new, "A_new")
        bieu_dien_vector(b_new, "b_new")
        
        jacobi_verbose_fixed(A_new, b_new, k_buoc)