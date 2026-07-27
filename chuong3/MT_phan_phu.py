import numpy as np

def read_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Vui lòng nhập số nguyên!")

def read_matrix_interactive(n, name="A"):
    print(f"--- Nhập ma trận {name} ({n}x{n}) ---")
    matrix = []
    for i in range(n):
        while True:
            try:
                line = input(f"Nhập hàng {i+1} (cách nhau khoảng trắng): ").replace('−', '-').replace('–', '-').split()
                if len(line) != n:
                    print(f"Vui lòng nhập đúng {n} phần tử!")
                    continue
                matrix.append([float(x) for x in line])
                break
            except ValueError:
                print("Dữ liệu phải là số!")
    return np.array(matrix, dtype=float)

def bieu_dien_ma_tran(matrix, ten=""):
    """In ma trận siêu đẹp để chép bài"""
    if ten:
        print(f"{ten} = ")
    for row in matrix:
        print("  [ " + " ".join([f"{x:8.4f}" for x in row]) + " ]")
    print()

def inverse_adjoint_verbose_fixed(A):
    n = A.shape[0]
    print("\n================ BẮT ĐẦU TÌM MA TRẬN NGHỊCH ĐẢO ================")
    
    # BƯỚC 1: Tính định thức (Giải thích siêu chi tiết)
    print("Bước 1: Tính định thức của ma trận A (Khai triển Laplace theo Hàng 1)")
    det_A = 0
    if n == 1:
        det_A = A[0,0]
        print(f"Ma trận 1x1, det(A) = {det_A:.4f}\n")
    else:
        cong_thuc_str = []
        thay_so_str = []
        for j in range(n):
            # Lấy ma trận con bỏ hàng 1 (index 0) và cột j+1 (index j)
            M_0j = np.delete(np.delete(A, 0, axis=0), j, axis=1)
            det_M_0j = np.linalg.det(M_0j)
            
            # Khử sai số -0.0000
            if abs(det_M_0j) < 1e-9: det_M_0j = 0.0
            
            # Tính dấu (-1)^(1 + j+1)
            dau = 1 if j % 2 == 0 else -1
            dau_str = "+" if dau == 1 else "-"
            if j == 0: dau_str = "" if dau == 1 else "-" # Số đầu tiên không cần dấu +
            
            phan_tu = A[0, j]
            gia_tri_thanh_phan = dau * phan_tu * det_M_0j
            det_A += gia_tri_thanh_phan
            
            # In chi tiết từng bước cho phần tử này
            print(f"  * Xét phần tử a_1{j+1} = {phan_tu:.4f}:")
            bieu_dien_ma_tran(M_0j, f"    Ma trận con M_1{j+1}")
            print(f"    det(M_1{j+1}) = {det_M_0j:.4f}\n")
            
            # Gom chuỗi công thức để in ra ở cuối
            cong_thuc_str.append(f"{dau_str} a_1{j+1}*det(M_1{j+1})")
            thay_so_str.append(f"{dau_str} ({phan_tu:.4f})*({det_M_0j:.4f})")
        
        print(f"=> det(A) = {' '.join(cong_thuc_str).strip()}")
        print(f"          = {' '.join(thay_so_str).strip()}")
        print(f"          = {det_A:.4f}\n")
    
    if abs(det_A) < 1e-9:
        print("🚨 NHẬN XÉT: det(A) = 0. Ma trận suy biến, KHÔNG CÓ ma trận nghịch đảo!")
        print("================ KẾT THÚC ================\n")
        return None
        
    # BƯỚC 2: Tìm ma trận phần bù đại số C
    print("Bước 2: Tìm Ma trận phần bù đại số C")
    print("Công thức: C_ij = (-1)^(i+j) * det(M_ij)")
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M_ij = np.delete(np.delete(A, i, axis=0), j, axis=1)
            det_M_ij = np.linalg.det(M_ij)
            C_ij = ((-1) ** (i + j)) * det_M_ij
            C[i, j] = C_ij
            if abs(C_ij) < 1e-9: C_ij = 0.0 
            print(f"  C_{i+1}{j+1} = (-1)^{i+1}+{j+1} * det(M_{i+1}{j+1}) = {C_ij:.4f}")
    
    print()
    bieu_dien_ma_tran(C, "Ma trận phần bù đại số C")

    # BƯỚC 3: Tìm ma trận phụ hợp adj(A)
    print("Bước 3: Tìm Ma trận phụ hợp adj(A) bằng cách chuyển vị C")
    adj_A = C.T
    bieu_dien_ma_tran(adj_A, "adj(A) = C^T")

    # BƯỚC 4: Tính ma trận nghịch đảo
    print("Bước 4: Tính ma trận nghịch đảo")
    print("Công thức: A^-1 = (1 / det(A)) * adj(A)")
    A_inv = (1.0 / det_A) * adj_A
    bieu_dien_ma_tran(A_inv, "Ma trận nghịch đảo CUỐI CÙNG A^-1")
    
    print("================ KẾT THÚC ================\n")
    return A_inv

if __name__ == "__main__":
    print("=== PHUONG PHAP MA TRAN PHAN PHU (BẢN CHUẨN CÓ TÍNH DET) ===")
    n = read_int("Nhap cap ma tran vuong n: ")
    A = read_matrix_interactive(n, "A")
    inverse_adjoint_verbose_fixed(A)