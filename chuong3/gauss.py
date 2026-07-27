import math

def read_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("Vui long nhap so nguyen hop le.")

def read_matrix_keyboard(rows, cols, name):
    print(f"Nhap ma tran {name} ({rows} x {cols}). Moi dong cach nhau boi dau cach.")
    matrix = []
    for i in range(rows):
        while True:
            try:
                line = input(f"{name}[{i + 1}]: ").strip()
                row = [float(x) for x in line.split()]
                if len(row) != cols:
                    print(f"Can dung {cols} so tren moi dong. Hay nhap lai.")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Du lieu khong hop le. Hay nhap lai.")
    return matrix

def print_matrix(C, m, n, p, title):
    print(f"\n--- {title} ---")
    for i in range(m):
        row_str = ""
        for j in range(n + p):
            if j == n:
                row_str += " | "
            row_str += f"{C[i][j]:8.4f} "
        print(row_str)
    print("-" * 40)

def solve_gauss_general():
    print("=== GAUSS CHUNG: NHAP MA TRAN A, B VOI KICH THUOC BAT KY ===")
    m = read_int("Nhap so hang m cua A: ")
    n = read_int("Nhap so cot n cua A: ")
    p = read_int("Nhap so cot p cua B (neu he 1 ve phai thi nhap 1): ")

    if m <= 0 or n <= 0 or p <= 0:
        print("Loi: m, n, p phai la so duong.")
        return

    A = read_matrix_keyboard(m, n, "A")
    B = read_matrix_keyboard(m, p, "B")

    C = [A[i] + B[i] for i in range(m)]
    ind = [-1] * m

    print("\n=== THUAT TOAN GAUSS (NHAP TREN BAN PHIM) ===")
    print(f"Kich thuoc A: {m}x{n}, B: {m}x{p}")
    print("\n>>> NOI DUNG TRINH BAY: BIEN LUAN HE PHUONG TRINH TUYEN TINH <<<")
    print_matrix(C, m, n, p, "MA TRAN BAN DAU C = [A|B]")

    print("\n>>> BAT DAU QUY TRINH THUAN <<<")

    i = 0
    j = 0
    while i < m and j < n:
        max_val = 0.0
        pivot_row = i
        for k in range(i, m):
            if abs(C[k][j]) > max_val:
                max_val = abs(C[k][j])
                pivot_row = k

        if max_val < 1e-10:
            print(f"Cot {j + 1} toan so 0 tu hang {i + 1}, bo qua.")
            j += 1
            continue

        if pivot_row != i:
            C[i], C[pivot_row] = C[pivot_row], C[i]
            print(f"Doi cho hang {i + 1} va hang {pivot_row + 1}")

        ind[i] = j

        for k in range(i + 1, m):
            if abs(C[k][j]) > 1e-10:
                factor = C[k][j] / C[i][j]
                for h in range(j, n + p):
                    C[k][h] -= factor * C[i][h]

        print_matrix(C, m, n, p, f"Sau khi khu cot {j + 1} (chot tai hang {i + 1})")
        i += 1
        j += 1

    print(f"Mang ind (Chi so cot chot): {[x + 1 if x != -1 else 0 for x in ind]}")
    print("\n>>> BAT DAU QUY TRINH NGUOC <<<")

    vo_nghiem = False
    for r in range(m):
        if ind[r] == -1:
            for c in range(n, n + p):
                if abs(C[r][c]) > 1e-10:
                    vo_nghiem = True
                    break
        if vo_nghiem:
            break

    if vo_nghiem:
        print("KET LUAN: He phuong trinh VO NGHIEM.")
        print("LY DO: Ton tai it nhat mot dong co he so A bang 0 nhung ve phai khac 0.")
        print("Da giai xong! He vo nghiem.")
        return

    so_hang_khac_0 = sum(1 for x in ind if x != -1)
    free_vars = [col for col in range(n) if col not in ind]

    if so_hang_khac_0 < n:
        print(f"He co VO SO NGHIEM. Cac bien tu do la: {', '.join([f'x{j + 1}' for j in free_vars])}\n")
        print("Cach trinh bay: bieu dien cac bien phu thuoc theo cac bien tu do.\n")
    else:
        print("He co NGHIEM DUY NHAT.\n")
        print("Cach trinh bay: dien nguoc tu duoi len tren de tim tung bien.\n")

    for c_b in range(p):
        print(f"--- Giai he cho cot B thu {c_b + 1} ---")
        X_expr = [[0.0] * (n + 1) for _ in range(n)]

        for j in free_vars:
            X_expr[j][j] = 1.0

        for r_idx in range(so_hang_khac_0 - 1, -1, -1):
            col_idx = ind[r_idx]
            pivot_val = C[r_idx][col_idx]
            X_expr[col_idx][n] = C[r_idx][n + c_b] / pivot_val

            for k in range(col_idx + 1, n):
                coeff = C[r_idx][k] / pivot_val
                for v in range(n + 1):
                    X_expr[col_idx][v] -= coeff * X_expr[k][v]

        for i_var in range(n):
            if i_var in free_vars:
                print(f"  x{i_var + 1} la bien tu do (thuoc R)")
            else:
                terms = []
                constant = X_expr[i_var][n]
                if abs(constant) > 1e-10 or len(free_vars) == 0:
                    terms.append(f"{constant:.4f}")

                for j in free_vars:
                    c = X_expr[i_var][j]
                    if abs(c) > 1e-10:
                        sign = " + " if c > 0 else " - "
                        if len(terms) == 0 and c < 0:
                            sign = "-"
                        if len(terms) == 0 and c > 0:
                            sign = ""

                        val = abs(c)
                        val_str = f"{val:.4f}*" if abs(val - 1.0) > 1e-10 else ""
                        terms.append(f"{sign}{val_str}x{j + 1}")

                expr_str = "".join(terms) if terms else "0.0000"
                print(f"  x{i_var + 1} = {expr_str}")
        print()

    print(">>> KET LUAN CUOI CUNG <<<")
    if vo_nghiem:
        print("He phuong trinh khong co nghiem.")
    elif so_hang_khac_0 < n:
        print("He phuong trinh co vo so nghiem.")
    else:
        print("He phuong trinh co nghiem duy nhat.")

    print("Da giai xong!")

if __name__ == "__main__":
    solve_gauss_general()