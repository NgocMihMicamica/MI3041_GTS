import numpy as np

def read_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("Vui lòng nhập số nguyên hợp lệ!")

def read_matrix(n, m, name):
    print(f"--- Nhập ma trận {name} ({n}x{m}) ---")
    matrix = []
    for i in range(n):
        while True:
            try:
                raw = input(f"Nhập hàng {i+1}: ").replace('−', '-').replace('–', '-')
                row = list(map(float, raw.split()))
                if len(row) != m:
                    print(f"Hàng phải có đúng {m} phần tử!")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Dữ liệu không hợp lệ!")
    return np.array(matrix, dtype=float)

def bieu_dien_ma_tran(matrix, ten=""):
    if ten: print(f"{ten} = ")
    for row in matrix:
        print("  [ " + " ".join([f"{x:10.6f}" for x in row]) + " ]")
    print()

def pseudo_inverse_svd_verbose(A):
    m, n = A.shape
    r = np.linalg.matrix_rank(A)

    print("\n================ TÍNH PSEUDO-INVERSE A⁺ BẰNG SVD ================")
    print(f"Ma trận A kích thước {m}x{n}")
    print(f"Hạng của A: rank(A) = {r}")
    bieu_dien_ma_tran(A, "A")

    # Bước 1: Tính SVD
    print(">>> BƯỚC 1: PHÂN TÍCH SVD A = U · Σ · Vᵀ <<<")
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T

    print(f"Số giá trị kỳ dị khác 0: {len([si for si in s if si > 1e-10])}")
    print("Các giá trị kỳ dị σ_i:")
    for i, si in enumerate(s):
        print(f"  σ_{i+1} = {si:.6f}")

    bieu_dien_ma_tran(U, "U (trái)")

    # Tạo ma trận Σ đầy đủ
    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        if i < len(s):
            Sigma[i, i] = s[i]
    bieu_dien_ma_tran(Sigma, "Σ (ma trận giá trị kỳ dị)")
    bieu_dien_ma_tran(V, "V (phải)")

    # Bước 2: Tính Σ⁺
    print(">>> BƯỚC 2: TÍNH Σ⁺ (pseudo-inverse của Σ) <<<")
    Sigma_plus = np.zeros((n, m))
    for i in range(min(m, n)):
        if i < len(s) and s[i] > 1e-10:
            Sigma_plus[i, i] = 1.0 / s[i]
            print(f"  Σ⁺[{i+1},{i+1}] = 1/σ_{i+1} = 1/{s[i]:.6f} = {1.0/s[i]:.6f}")
        else:
            print(f"  Σ⁺[{i+1},{i+1}] = 0 (vì σ_{i+1} ≈ 0)")

    bieu_dien_ma_tran(Sigma_plus, "Σ⁺")

    # Bước 3: Tính A⁺ = V · Σ⁺ · Uᵀ
    print(">>> BƯỚC 3: TÍNH A⁺ = V · Σ⁺ · Uᵀ <<<")
    print("Công thức: A⁺ = V · Σ⁺ · Uᵀ")

    A_plus = np.dot(V, np.dot(Sigma_plus, U.T))

    # Khử sai số nhỏ
    A_plus[np.abs(A_plus) < 1e-10] = 0.0

    bieu_dien_ma_tran(A_plus, "A⁺ (Pseudo-inverse)")

    # Bước 4: Kiểm tra
    print(">>> BƯỚC 4: KIỂM TRA TÍNH CHẤT CỦA A⁺ <<<")

    AAplusA = np.dot(A, np.dot(A_plus, A))
    AplusAAplus = np.dot(A_plus, np.dot(A, A_plus))

    err1 = np.linalg.norm(AAplusA - A, 'fro')
    err2 = np.linalg.norm(AplusAAplus - A_plus, 'fro')

    print(f"  ‖A·A⁺·A - A‖_F = {err1:.6e}  (≈ 0 là đúng)")
    print(f"  ‖A⁺·A·A⁺ - A⁺‖_F = {err2:.6e}  (≈ 0 là đúng)")

    if err1 < 1e-8 and err2 < 1e-8:
        print("  ✅ A⁺ thỏa mãn các tính chất pseudo-inverse!")
    else:
        print("  ⚠️ Có sai số nhỏ do làm tròn số.")

    # Bước 5: Tính số điều kiện
    if r == min(m, n) and len(s) > 0 and s[-1] > 1e-10:
        cond = s[0] / s[-1]
        print(f"\nSố điều kiện: cond(A) = σ_max/σ_min = {s[0]:.6f}/{s[-1]:.6f} = {cond:.4f}")
        if cond > 1000:
            print("  ⚠️ Ma trận ill-conditioned (số điều kiện lớn)")
        else:
            print("  ✅ Ma trận well-conditioned")
    else:
        print("\nSố điều kiện: VÔ HẠN (ma trận không full rank)")

    print("================ KẾT THÚC ================\n")
    return A_plus

def solve_pseudo_inverse(A, b):
    """Giải hệ Ax = b bằng pseudo-inverse (nghiệm bình phương tối tiểu)"""
    print("\n================ GIẢI HỆ Ax = b BẰNG PSEUDO-INVERSE ================")
    A_plus = pseudo_inverse_svd_verbose(A)

    print(">>> BƯỚC 5: TÍNH NGHIỆM x = A⁺ · b <<<")
    x = np.dot(A_plus, b)
    bieu_dien_ma_tran(x, "Nghiệm bình phương tối tiểu x = A⁺·b")

    # Kiểm tra
    residual = np.linalg.norm(np.dot(A, x) - b)
    print(f"  ‖Ax - b‖₂ = {residual:.6e}")

    if residual < 1e-8:
        print("  ✅ Nghiệm chính xác (hệ có nghiệm đúng)")
    else:
        print("  ℹ️ Nghiệm xấp xỉ tốt nhất theo nghĩa bình phương tối tiểu")

    return x

if __name__ == "__main__":
    print("=== PSEUDO-INVERSE A⁺ BẰNG SVD ===")
    print("1. Chỉ tính A⁺")
    print("2. Giải hệ Ax = b bằng A⁺")

    mode = read_int("Chọn (1 hoặc 2): ")
    m = read_int("Nhập số hàng m: ")
    n = read_int("Nhập số cột n: ")
    A = read_matrix(m, n, "A")

    if mode == 1:
        pseudo_inverse_svd_verbose(A)
    elif mode == 2:
        b = read_matrix(m, 1, "b")
        solve_pseudo_inverse(A, b)
