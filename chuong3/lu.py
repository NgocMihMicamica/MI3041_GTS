import numpy as np

def read_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("Vui long nhap so nguyen hop le.")

def read_matrix_interactive(rows, cols, name="A"):
    print(f"--- Nhập ma trận {name} ({rows}x{cols}) ---")
    matrix = []
    for i in range(rows):
        while True:
            try:
                line = input(f"Nhập hàng {i+1}: ").replace('−', '-').replace('–', '-')
                row = [float(x) for x in line.split()]
                if len(row) != cols:
                    print(f"Vui lòng nhập đúng {cols} phần tử!")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Dữ liệu không hợp lệ!")
    return np.array(matrix, dtype=float)

def read_vector_interactive(size, name="b"):
    print(f"--- Nhập vectơ cột {name} ({size} phần tử) ---")
    vector = []
    while len(vector) < size:
        try:
            line = input().replace('−', '-').replace('–', '-')
            if not line.strip(): continue
            vector.extend([float(x) for x in line.split()])
        except ValueError:
            print("Dữ liệu không hợp lệ!")
    return np.array(vector[:size], dtype=float).reshape(size, 1)

def bieu_dien_ma_tran(matrix, ten=""):
    if ten: print(f"{ten} = ")
    for row in matrix:
        print("  [ " + " ".join([f"{x:9.4f}" for x in row]) + " ]")
    print()

def solve_lu_pivoting_verbose_super_detailed(A, b):
    n = A.shape[0]
    
    print("\n================ BẮT ĐẦU PHÂN TÁCH LU (CÓ HOÁN VỊ PA=LU) ================")
    
    U = A.copy()
    L = np.zeros((n, n))
    P = np.eye(n)
    
    print(">>> BƯỚC 1: TÌM MA TRẬN L, U VÀ MA TRẬN HOÁN VỊ P <<<")
    print("Mục tiêu: Dùng phép khử Gauss để biến A thành ma trận tam giác trên U.")
    print("Đồng thời, các hệ số khử sẽ được cất trực tiếp vào ma trận tam giác dưới L.\n")
    
    bieu_dien_ma_tran(U, "Ma trận ban đầu U^(0) = A")

    for k in range(n - 1): # Chỉ cần khử n-1 cột
        print(f"--- KHỬ GAUSS CỘT {k+1} ---")
        
        # Tìm phần tử lớn nhất trong cột k để làm Pivot
        pivot_row = np.argmax(np.abs(U[k:n, k])) + k
        
        if abs(U[pivot_row, k]) < 1e-10:
            print(f"🚨 CẢNH BÁO: Cột {k+1} toàn số 0 từ hàng {k+1} trở xuống. Ma trận suy biến!")
            print("=> Không thể giải hệ phương trình duy nhất.")
            return None
            
        # Kiểm tra và Hoán vị
        if pivot_row != k:
            print(f"-> Phần tử trụ lớn nhất nằm ở HÀNG {pivot_row+1} (giá trị = {U[pivot_row, k]:.4f}).")
            print(f"-> HOÁN VỊ HÀNG {k+1} và HÀNG {pivot_row+1} của U, P (và L nếu có).")
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            if k > 0: # Chỉ hoán vị phần L đã tính ở các cột trước đó
                L[[k, pivot_row], 0:k] = L[[pivot_row, k], 0:k]
            bieu_dien_ma_tran(U, f"Ma trận U sau khi hoán vị")
        else:
            print(f"-> Phần tử trụ lớn nhất đã nằm sẵn ở HÀNG {k+1} (giá trị = {U[k, k]:.4f}). Không cần hoán vị.")

        # Khử Gauss
        print("Tiến hành khử các phần tử dưới đường chéo:")
        for i in range(k+1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor # Nhét hệ số vào L
            
            print(f"  + Xét hàng {i+1}:")
            print(f"    Hệ số khử l_{i+1},{k+1} = U_{i+1},{k+1} / U_{k+1},{k+1} = {U[i, k]:.4f} / {U[k, k]:.4f} = {factor:.4f}")
            print(f"    Thực hiện phép biến đổi: h_{i+1} = h_{i+1} - ({factor:.4f}) * h_{k+1}")
            
            U[i, k:] -= factor * U[k, k:]
            U[i, k] = 0.0 # Ép chính xác về 0 để tránh hiển thị số dị kiểu -0.0000
            
        bieu_dien_ma_tran(U, f"Ma trận U sau khi khử xong cột {k+1}")

    # Điền các số 1 lên đường chéo của L
    np.fill_diagonal(L, 1.0)
    
    # Ép sai số nhỏ về 0 cho lung linh
    L[np.abs(L) < 1e-10] = 0.0
    U[np.abs(U) < 1e-10] = 0.0
    
    print("\n--- TỔNG KẾT BƯỚC 1: KẾT QUẢ PHÂN TÁCH PA = LU ---")
    bieu_dien_ma_tran(P, "Ma trận hoán vị P")
    bieu_dien_ma_tran(L, "Ma trận tam giác dưới L (Chứa hệ số khử + Đường chéo = 1)")
    bieu_dien_ma_tran(U, "Ma trận tam giác trên U (Kết quả sau khi khử Gauss)")
    
    print("\n>>> BƯỚC 2: NHÂN VECTƠ B VỚI MA TRẬN HOÁN VỊ P <<<")
    print("Mục đích: Đổi chỗ các phần tử của vectơ b cho khớp với các bước hoán vị hàng ở trên.")
    Pb = np.dot(P, b)
    bieu_dien_ma_tran(Pb, "Vectơ Pb")
    
    print("\n>>> BƯỚC 3: GIẢI HỆ TRUNG GIAN L * y = Pb (Giải từ trên xuống dưới) <<<")
    y = np.zeros((n, 1))
    for i in range(n):
        tong = sum(L[i, j] * y[j, 0] for j in range(i))
        y[i, 0] = Pb[i, 0] - tong
        print(f"  y_{i+1} = Pb_{i+1} - sum(L*y) = {y[i, 0]:.4f}")
    print()
    bieu_dien_ma_tran(y, "Vectơ trung gian y")
    
    print("\n>>> BƯỚC 4: GIẢI HỆ CUỐI CÙNG U * x = y (Giải ngược từ dưới lên trên) <<<")
    x = np.zeros((n, 1))
    for i in range(n-1, -1, -1):
        tong = sum(U[i, j] * x[j, 0] for j in range(i+1, n))
        x[i, 0] = (y[i, 0] - tong) / U[i, i]
        print(f"  x_{i+1} = (y_{i+1} - sum(U*x)) / U_{i+1},{i+1} = {x[i, 0]:.4f}")
    print()
    bieu_dien_ma_tran(x, "Nghiệm của hệ phương trình X")
    
    print("================ KẾT THÚC ================\n")

if __name__ == "__main__":
    print("=== PHÂN TÁCH LU (BẢN FINAL CÓ HOÁN VỊ + GIẢI THÍCH CHI TIẾT) ===")
    n = read_int("Nhập cấp ma trận vuông n: ")
    A = read_matrix_interactive(n, n, "A")
    b = read_vector_interactive(n, "b")
    
    solve_lu_pivoting_verbose_super_detailed(A, b)