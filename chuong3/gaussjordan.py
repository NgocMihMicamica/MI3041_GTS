import numpy as np


def nhap_so_nguyen(thong_bao):
    while True:
        try:
            return int(input(thong_bao).strip())
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


def in_ma_tran_ra_man_hinh(M, tieu_de, n):
    print(f"\n--- {tieu_de} ---")
    m, cols = M.shape
    for i in range(m):
        row_str = ""
        for j in range(cols):
            if j == n:
                row_str += " | "
            val = 0.0 if abs(M[i, j]) < 1e-10 else M[i, j]
            row_str += f"{val:8.4f} "
        print(row_str)
    print("-" * 40)


def giai_gauss_jordan_uu_tien_toan_cuc():
    try:
        print("=== GAUSS-JORDAN CHUNG: NHAP MA TRAN A, B VOI KICH THUOC BAT KY ===")
        m = nhap_so_nguyen("Nhap so hang m cua A: ")
        n = nhap_so_nguyen("Nhap so cot n cua A: ")
        p = nhap_so_nguyen("Nhap so cot p cua B (neu he 1 ve phai thi nhap 1): ")

        if m <= 0 or n <= 0 or p <= 0:
            print("Loi: m, n, p phai la so duong.")
            return

        A = read_matrix_keyboard(m, n, "A")
        B = read_matrix_keyboard(m, p, "B")

        M = np.hstack((A, B)).astype(float)
        hang_da_dung = set()
        cot_da_dung = set()
        pivot_dict = {}

        print("\n>>> GIAI DOAN 1: TIM CHOT TOAN CUC VA KHU <<<")

        for step in range(min(m, n)):
            r_pivot, c_pivot = -1, -1

            for j in range(n):
                if j in cot_da_dung:
                    continue
                for i in range(m):
                    if i in hang_da_dung:
                        continue
                    if np.isclose(abs(M[i, j]), 1.0):
                        r_pivot, c_pivot = i, j
                        break
                if r_pivot != -1:
                    break

            if r_pivot == -1:
                max_val = -1
                for j in range(n):
                    if j in cot_da_dung:
                        continue
                    for i in range(m):
                        if i in hang_da_dung:
                            continue
                        if abs(M[i, j]) > max_val:
                            max_val = abs(M[i, j])
                            r_pivot, c_pivot = i, j

            if r_pivot == -1 or abs(M[r_pivot, c_pivot]) < 1e-10:
                print("Khong tim them duoc phan tu chot hop le. Dung qua trinh khu.")
                break

            hang_da_dung.add(r_pivot)
            cot_da_dung.add(c_pivot)
            pivot_dict[c_pivot] = r_pivot

            pivot_val = M[r_pivot, c_pivot]
            print(f"\n-> Buoc {step + 1}: Chon Pivot TOAN CUC tai M[{r_pivot + 1}, {c_pivot + 1}] = {pivot_val:.4f}")

            for i in range(m):
                if i != r_pivot and abs(M[i, c_pivot]) > 1e-10:
                    he_so = M[i, c_pivot] / pivot_val
                    print(f"  Khu hang {i + 1}: h{i + 1} = h{i + 1} - ({he_so:.4f}) * h{r_pivot + 1}")
                    M[i] = M[i] - he_so * M[r_pivot]

            in_ma_tran_ra_man_hinh(M, f"Trang thai sau khi khu xong cot {c_pivot + 1}", n)

        print("\n>>> GIAI DOAN 2: CHUAN HOA CAC CHOT VE 1 <<<")
        for c_piv, r_piv in pivot_dict.items():
            divisor = M[r_piv, c_piv]
            if abs(divisor - 1.0) > 1e-10:
                print(f"  Chuan hoa hang {r_piv + 1}: Chia cho {divisor:.4f}")
                M[r_piv] = M[r_piv] / divisor

        M = np.where(np.abs(M) < 1e-10, 0.0, M)

        print("\n>>> KET LUAN NGHIEM <<<")
        vo_nghiem = False
        for i in range(m):
            if np.all(M[i, :n] == 0) and np.any(M[i, n:] != 0):
                vo_nghiem = True
                break

        if vo_nghiem:
            print("He phuong trinh VO NGHIEM.")
            return

        bien_tu_do = [j for j in range(n) if j not in pivot_dict]

        if len(bien_tu_do) > 0:
            print(f"He co VO SO NGHIEM. Cac bien tu do: {', '.join([f'x{j + 1}' for j in bien_tu_do])}")
        else:
            print("He co NGHIEM DUY NHAT.")

        for c_b in range(p):
            print(f"--- Ket qua cho ve phai thu {c_b + 1} ---")
            for var_idx in range(n):
                if var_idx in bien_tu_do:
                    print(f"  x{var_idx + 1} la bien tu do (thuoc R)")
                else:
                    row = pivot_dict[var_idx]
                    constant = M[row, n + c_b]
                    terms = []
                    if abs(constant) > 0 or len(bien_tu_do) == 0:
                        terms.append(f"{constant:.4f}")

                    for f_var in bien_tu_do:
                        he_so = -M[row, f_var]
                        if abs(he_so) > 0:
                            sign = " + " if he_so > 0 else " - "
                            if not terms and he_so < 0:
                                sign = "-"
                            if not terms and he_so > 0:
                                sign = ""

                            val = abs(he_so)
                            val_str = f"{val:.4f}*" if abs(val - 1.0) > 1e-10 else ""
                            terms.append(f"{sign}{val_str}x{f_var + 1}")

                    expr = "".join(terms) if terms else "0.0000"
                    print(f"  x{var_idx + 1} = {expr}")

    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    giai_gauss_jordan_uu_tien_toan_cuc()