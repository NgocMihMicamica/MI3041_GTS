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
                line = input(f"Nhập hàng {i+1}: ").replace('−', '-').replace('–', '-').split()
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
            line = input().replace('−', '-').replace('–', '-')
            if not line.strip(): continue
            vector.extend([float(x) for x in line.split()])
        except ValueError:
            print("Dữ liệu không hợp lệ!")
    return np.array(vector[:size], dtype=float)

def bieu_dien_ma_tran(matrix, ten=""):
    if ten: print(f"{ten} = ")
    for row in matrix:
        print("  [ " + " ".join([f"{x:9.4f}" for x in row]) + " ]")
    print()

def bieu_dien_vector(vector, ten=""):
    if ten: print(f"{ten} = ")
    print("  [ " + " ".join([f"{x:9.4f}" for x in vector]) + " ]^T")
    print()

def kiem_tra_cheo_troi_hang(A):
    n = len(A)
    for i in range(n):
        tong_hang = sum(abs(A[i, j]) for j in range(n) if i != j)
        if abs(A[i, i]) <= tong_hang:
            return False
    return True

def gauss_seidel_sor_verbose(A, b, x0, k_buoc, omega=1.0):
    """
    omega = 1.0: Gauss-Seidel thường
    omega > 1.0: SOR (over-relaxation)
    omega < 1.0: under-relaxation
    """
    n = len(b)
    x = np.array(x0, dtype=float)

    print(f"\n================ GAUSS-SEIDEL SOR (ω = {omega}) ================")

    if omega == 1.0:
        print("Chế độ: Gauss-Seidel THƯỜNG (ω = 1)")
    else:
        print(f"Chế độ: SOR với hệ số thư giãn ω = {omega}")
        print("Công thức SOR: x_i^(k) = (1-ω)·x_i^(k-1) + (ω/a_ii)·[b_i - Σ(a_ij·x_j)]")

    print(f"Giả thiết ban đầu: X^(0) = {list(np.round(x, 4))}")

    for buoc in range(1, k_buoc + 1):
        print(f"\n--- Bước lặp k = {buoc} ---")
        x_old = x.copy()

        for i in range(n):
            if A[i, i] == 0:
                print(f"🚨 LỖI: A[{i+1},{i+1}] = 0, không thể chia!")
                return

            # Tính tổng các phần tử đã cập nhật (trái)
            tong_trai = sum(A[i, j] * x[j] for j in range(i))
            # Tính tổng các phần tử chưa cập nhật (phải)
            tong_phai = sum(A[i, j] * x_old[j] for j in range(i + 1, n))

            # Gauss-Seidel thường
            x_gs = (b[i] - tong_trai - tong_phai) / A[i, i]

            # SOR
            x[i] = (1 - omega) * x_old[i] + omega * x_gs

            if omega == 1.0:
                print(f"  x_{i+1}^({buoc}) = (1/{A[i,i]:.4f}) * [{b[i]:.4f} - {tong_trai:.4f} - {tong_phai:.4f}] = {x[i]:.4f}")
            else:
                print(f"  x_{i+1}^({buoc}) = (1-{omega:.2f})·{x_old[i]:.4f} + {omega:.2f}·({x_gs:.4f}) = {x[i]:.4f}")

        arr_str = ", ".join([f"{val:.4f}" for val in x])
        print(f"==> Kết luận k={buoc}: X^({buoc}) = [{arr_str}]^T")

        # Sai số
        if buoc > 1:
            err = max(abs(x[i] - x_old[i]) for i in range(n))
            print(f"  Sai số bước: max|x^({buoc}) - x^({buoc-1})| = {err:.6e}")

    print("\n================ KẾT THÚC ================\n")

def tim_omega_toi_uu(A, b, x0, max_iter=50, tol=1e-6):
    """Tìm omega tối ưu bằng cách thử nhiều giá trị"""
    print("\n>>> TÌM ω TỐI ƯU BẰNG THỬ NGHIỆM <<<")
    print("Thử các giá trị ω từ 0.1 đến 1.9 (bước 0.1)...")

    best_omega = 1.0
    best_iter = max_iter

    print(f"{'ω':>6} | {'Số bước hội tụ':>15} | {'Sai số cuối':>15}")
    print("-" * 45)

    for omega in [round(0.1 * i, 1) for i in range(1, 20)]:
        x = np.array(x0, dtype=float)
        n = len(b)

        for k in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                tong_trai = sum(A[i, j] * x[j] for j in range(i))
                tong_phai = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
                x_gs = (b[i] - tong_trai - tong_phai) / A[i, i]
                x[i] = (1 - omega) * x_old[i] + omega * x_gs

            err = max(abs(x[i] - x_old[i]) for i in range(n))
            if err < tol:
                print(f"{omega:>6.1f} | {k+1:>15} | {err:>15.6e}")
                if k + 1 < best_iter:
                    best_iter = k + 1
                    best_omega = omega
                break
        else:
            print(f"{omega:>6.1f} | {'KHÔNG hội tụ':>15} | {'-':>15}")

    print(f"\n✅ ω tối ưu ≈ {best_omega} (hội tụ sau {best_iter} bước)")
    return best_omega

if __name__ == "__main__":
    print("=== GAUSS-SEIDEL SOR (SUCCESSIVE OVER-RELAXATION) ===")
    print("1. Gauss-Seidel thường (ω = 1)")
    print("2. SOR với ω tự nhập")
    print("3. Tự động tìm ω tối ưu")

    choice = read_int("Chọn (1, 2 hoặc 3): ")

    n = read_int("Nhập số biến n: ")
    k_buoc = read_int("Nhập số lần lặp k: ")

    A = read_matrix_interactive(n, n, "A")
    b = read_vector_interactive(n, "b")
    x0 = np.zeros(n)

    print("\n=== LẬP LUẬN ĐẦU VÀO ===")
    if kiem_tra_cheo_troi_hang(A):
        print("Nhận xét: A CHÉO TRỘI HÀNG → Hội tụ chắc chắn với mọi ω ∈ (0, 2)")
    else:
        print("Nhận xét: A chưa chéo trội → Cần chọn ω cẩn thận (thường ω < 1)")

    if choice == 1:
        gauss_seidel_sor_verbose(A, b, x0, k_buoc, omega=1.0)
    elif choice == 2:
        omega = float(input("Nhập ω (khuyên 1.0-1.5 cho chéo trội): ").strip())
        gauss_seidel_sor_verbose(A, b, x0, k_buoc, omega)
    elif choice == 3:
        best_omega = tim_omega_toi_uu(A, b, x0)
        print(f"\nChạy SOR với ω = {best_omega}:")
        gauss_seidel_sor_verbose(A, b, x0, k_buoc, best_omega)
