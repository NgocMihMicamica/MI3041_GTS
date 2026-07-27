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

def solve_lu_crout_verbose(A, b):
    n = A.shape[0]

    print("\n================ BẮT ĐẦU PHÂN TÁCH LU (DẠNG CROUT) ================")
    print(">>> BƯỚC 1: PHÂN TÁCH A = L * U <<<")
    print("Đặc trưng Crout: L là ma trận tam giác dưới (có đường chéo),")
    print("                  U là ma trận tam giác trên (đường chéo = 1)")
    print()

    L = np.zeros((n, n))
    U = np.eye(n)  # U có đường chéo = 1

    print("Công thức tính:")
    print("  • L[i][j] = A[i][j] - Σ(L[i][k]*U[k][j]) với k=0..j-1  (i ≥ j)")
    print("  • U[i][j] = (A[i][j] - Σ(L[i][k]*U[k][j])) / L[i][i]   (i < j)")
    print()

    for i in range(n):
        for j in range(i + 1):
            # Tính L[i][j]
            tong = sum(L[i][k] * U[k][j] for k in range(j))
            L[i][j] = A[i][j] - tong
            print(f"  L[{i+1}][{j+1}] = A[{i+1}][{j+1}] - Σ = {A[i][j]:.4f} - ({tong:.4f}) = {L[i][j]:.4f}")

        # Kiểm tra L[i][i] = 0
        if abs(L[i][i]) < 1e-10:
            print(f"\n🚨 CẢNH BÁO: L[{i+1}][{i+1}] = 0. Không thể phân tách Crout!")
            print("=> Ma trận không khả nghịch theo Crout hoặc cần hoán vị hàng.")
            return None

        for j in range(i + 1, n):
            tong = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = (A[i][j] - tong) / L[i][i]
            print(f"  U[{i+1}][{j+1}] = (A[{i+1}][{j+1}] - Σ) / L[{i+1}][{i+1}] = ({A[i][j]:.4f} - {tong:.4f}) / {L[i][i]:.4f} = {U[i][j]:.4f}")
        print()

    # Khử sai số nhỏ
    L[np.abs(L) < 1e-10] = 0.0
    U[np.abs(U) < 1e-10] = 0.0

    print("--- KẾT QUẢ PHÂN TÁCH A = L * U (DẠNG CROUT) ---")
    bieu_dien_ma_tran(L, "Ma trận tam giác dưới L (đường chéo tự do)")
    bieu_dien_ma_tran(U, "Ma trận tam giác trên U (đường chéo = 1)")

    # Kiểm tra
    print(">>> KIỂM TRA: L * U =")
    check = np.dot(L, U)
    bieu_dien_ma_tran(check, "L * U")

    print(">>> BƯỚC 2: GIẢI HỆ L * y = b (Thế xuôi) <<<")
    y = np.zeros((n, 1))
    for i in range(n):
        tong = sum(L[i][j] * y[j, 0] for j in range(i))
        y[i, 0] = (b[i, 0] - tong) / L[i][i]
        print(f"  y_{i+1} = (b_{i+1} - Σ) / L_{i+1}{i+1} = ({b[i,0]:.4f} - {tong:.4f}) / {L[i,i]:.4f} = {y[i,0]:.4f}")
    print()
    bieu_dien_ma_tran(y, "Vectơ trung gian y")

    print(">>> BƯỚC 3: GIẢI HỆ U * x = y (Thế ngược) <<<")
    x = np.zeros((n, 1))
    for i in range(n-1, -1, -1):
        tong = sum(U[i][j] * x[j, 0] for j in range(i+1, n))
        x[i, 0] = y[i, 0] - tong  # U[i][i] = 1
        print(f"  x_{i+1} = y_{i+1} - Σ = {y[i,0]:.4f} - {tong:.4f} = {x[i,0]:.4f}")
    print()
    bieu_dien_ma_tran(x, "Nghiệm của hệ phương trình X")

    print("================ KẾT THÚC ================\n")
    return x

if __name__ == "__main__":
    print("=== PHÂN TÁCH LU - DẠNG CROUT (L có đường chéo, U có đường chéo = 1) ===")
    n = read_int("Nhập cấp ma trận vuông n: ")
    A = read_matrix_interactive(n, n, "A")
    b = read_vector_interactive(n, "b")
    solve_lu_crout_verbose(A, b)
