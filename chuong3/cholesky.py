import numpy as np
import math

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
                raw_input = input(f"Nhập hàng {i+1} (các số cách nhau khoảng trắng): ")
                clean_input = raw_input.replace('−', '-').replace('–', '-')
                row = list(map(float, clean_input.split()))
                if len(row) != m:
                    print(f"Hàng phải có đúng {m} phần tử. Vui lòng nhập lại!")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Dữ liệu nhập vào không hợp lệ!")
    return np.array(matrix, dtype=float)

def read_vector(n, name):
    print(f"--- Nhập vectơ cột {name} (kích thước {n}) ---")
    vector = []
    while len(vector) < n:
        try:
            raw_input = input()
            clean_input = raw_input.replace('−', '-').replace('–', '-')
            if not clean_input.strip(): continue
            parts = list(map(float, clean_input.split()))
            vector.extend(parts)
        except ValueError:
            print("Dữ liệu không hợp lệ!")
    return np.array(vector[:n], dtype=float).reshape(n, 1)

def bieu_dien_ma_tran(matrix, ten=""):
    if ten: print(f"{ten} = ")
    for row in matrix:
        print("  [ " + " ".join([f"{x:9.4f}" for x in row]) + " ]")
    print()

def thuc_hien_cholesky(A, b):
    """Hàm lõi thực hiện các bước Cholesky chuẩn"""
    n = A.shape[0]
    print("--- Bước 1: Phân tích A = L * L^T ---")
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            tong_k = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                gia_tri_trong_can = A[i][i] - tong_k
                if gia_tri_trong_can <= 1e-9:
                    print(f"🚨 LỖI SÂU: Dù đã biến đổi, biểu thức căn L_{i+1}{i+1} vẫn <= 0.")
                    return None
                L[i][i] = math.sqrt(gia_tri_trong_can)
                print(f"  L_{i+1}{i+1} = sqrt(A_{i+1}{i+1} - sum) = {L[i][i]:.4f}")
            else:
                L[i][j] = (A[i][j] - tong_k) / L[j][j]
                print(f"  L_{i+1}{j+1} = (A_{i+1}{j+1} - sum) / L_{j+1}{j+1} = {L[i][j]:.4f}")

    print("\n=> Ta thu được ma trận tam giác dưới L và ma trận chuyển vị L^T:")
    bieu_dien_ma_tran(L, "L")
    LT = L.T
    bieu_dien_ma_tran(LT, "L^T")

    print("--- Bước 2: Giải hệ phương trình trung gian L * y = b ---")
    y = np.zeros((n, 1))
    for i in range(n):
        tong_Ly = sum(L[i][j] * y[j][0] for j in range(i))
        y[i][0] = (b[i][0] - tong_Ly) / L[i][i]
        print(f"  y_{i+1} = (b_{i+1} - sum) / L_{i+1}{i+1} = {y[i][0]:.4f}")
    print()
    bieu_dien_ma_tran(y, "Vectơ trung gian y")

    print("--- Bước 3: Giải hệ phương trình L^T * x = y ---")
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        tong_LTx = sum(LT[i][j] * x[j][0] for j in range(i + 1, n))
        x[i][0] = (y[i][0] - tong_LTx) / LT[i][i]
        print(f"  x_{i+1} = (y_{i+1} - sum) / L^T_{i+1}{i+1} = {x[i][0]:.4f}")
    print()
    bieu_dien_ma_tran(x, "Nghiệm của hệ phương trình X")
    return x

def kiem_tra_xac_dinh_duong(A):
    """Dùng thử np.linalg.cholesky để check nhanh xem ma trận có xác định dương không"""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def solve_cholesky_smart(A, b):
    print("\n================ BẮT ĐẦU PHƯƠNG PHÁP CHOLESKY ================")
    
    # Kiểm tra xem A có thỏa mãn SPD (Đối xứng & Xác định dương) không
    is_symmetric = np.allclose(A, A.T, atol=1e-9)
    is_pd = kiem_tra_xac_dinh_duong(A) if is_symmetric else False
    
    if is_symmetric and is_pd:
        print("Nhận xét: Ma trận A đối xứng và xác định dương. Tiến hành Cholesky trực tiếp.")
        thuc_hien_cholesky(A, b)
    else:
        ly_do = "không đối xứng" if not is_symmetric else "không xác định dương"
        print(f"🚨 Nhận xét: Ma trận A {ly_do}!")
        print("=> XỬ LÝ: Nhân cả 2 vế với ma trận chuyển vị A^T để tạo hệ (A^T * A)x = A^T * b\n")
        
        A_new = np.dot(A.T, A)
        b_new = np.dot(A.T, b)
        
        print("--- THÔNG TIN HỆ PHƯƠNG TRÌNH MỚI ---")
        bieu_dien_ma_tran(A_new, "A_new = A^T * A")
        bieu_dien_ma_tran(b_new, "b_new = A^T * b")
        
        print("Tiến hành phân tích Cholesky trên hệ mới:")
        thuc_hien_cholesky(A_new, b_new)
        
    print("================ KẾT THÚC ================\n")

if __name__ == "__main__":
    print("=== PHUONG PHAP CHOLESKY (AUTO BIẾN ĐỔI A^T * A) ===")
    n = read_int("Nhap cap ma tran vuong n: ")
    A = read_matrix(n, n, "A")
    b = read_vector(n, "b")
    
    solve_cholesky_smart(A, b)