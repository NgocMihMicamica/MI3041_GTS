import math
import numpy as np
from itertools import permutations

EPS = 1e-10


def normalize_number_text(text):
    return text.replace("−", "-").replace("–", "-").replace("—", "-")


def read_int(prompt):
    while True:
        try:
            return int(normalize_number_text(input(prompt).strip()))
        except ValueError:
            print("Vui long nhap so nguyen hop le.")


def read_float(prompt):
    while True:
        try:
            return float(normalize_number_text(input(prompt).strip()))
        except ValueError:
            print("Vui long nhap so thuc hop le.")


def read_matrix(rows, cols, name):
    print(f"Nhap ma tran {name} ({rows} x {cols}). Moi dong cach nhau boi dau cach.")
    matrix = []
    for i in range(rows):
        while True:
            try:
                line = normalize_number_text(input(f"{name}[{i + 1}]: ").strip())
                row = [float(x) for x in line.split()]
                if len(row) != cols:
                    print(f"Can dung {cols} so tren moi dong. Hay nhap lai.")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Du lieu khong hop le. Hay nhap lai.")
    return np.array(matrix, dtype=float)


def read_vector(rows, name):
    print(f"Nhap vector {name} ({rows} phan tu). Moi dong 1 so.")
    data = []
    for i in range(rows):
        data.append(read_float(f"{name}[{i + 1}]: "))
    return np.array(data, dtype=float)


def fmt(v):
    if abs(v) < EPS:
        v = 0.0
    return f"{v:10.6f}"


def print_matrix(A, title="Ma tran"):
    print(f"\n--- {title} ---")
    for row in A:
        print(" ".join(fmt(x) for x in row))
    print("-" * 60)


def print_augmented(A, b, title="Ma tran bo sung [A|b]"):
    print(f"\n--- {title} ---")
    for i in range(A.shape[0]):
        print(" ".join(fmt(x) for x in A[i]) + " | " + fmt(b[i]))
    print("-" * 60)


def is_square(A):
    return A.shape[0] == A.shape[1]


def is_symmetric(A):
    return is_square(A) and np.allclose(A, A.T, atol=1e-8)


def is_diagonally_dominant(A):
    if not is_square(A):
        return False
    for i in range(A.shape[0]):
        diag = abs(A[i, i])
        others = np.sum(np.abs(A[i])) - diag
        if diag < others - 1e-12:
            return False
    return True


def is_strictly_diagonally_dominant(A):
    if not is_square(A):
        return False
    for i in range(A.shape[0]):
        diag = abs(A[i, i])
        others = np.sum(np.abs(A[i])) - diag
        if diag <= others + 1e-12:
            return False
    return True


def is_spd(A):
    if not is_symmetric(A):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
    
def leading_principal_minors_nonzero(A):
    if not is_square(A):
        return False
    n = A.shape[0]
    for k in range(1, n + 1):
        detk = float(np.linalg.det(A[:k, :k]))
        if abs(detk) < EPS:
            return False
    return True


def find_row_permutation_nonzero_diagonal(A, max_search_size=8):
    if not is_square(A):
        return None
    n = A.shape[0]
    if n > max_search_size:
        return None
    for perm in permutations(range(n)):
        Ap = A[list(perm), :]
        if np.all(np.abs(np.diag(Ap)) > EPS):
            return list(perm)
    return None


def apply_row_permutation(A, b, perm):
    perm = list(perm)
    return A[perm, :], b[perm]


def print_row_permutation_explanation(perm):
    print("Giai thich hoan vi dong:")
    for new_pos, old_pos in enumerate(perm, start=1):
        print(f"  Dong moi {new_pos} lay tu dong cu {old_pos + 1}")
    print("  Ta chi doi vi tri cac phuong trinh, khong doi bien so an.")


def permuted_matrix(A, perm):
    perm = list(perm)
    return A[np.ix_(perm, perm)]


def find_bordering_permutation(A, max_search_size=8):
    if not is_square(A):
        return None
    n = A.shape[0]
    if n > max_search_size:
        return None
    for perm in permutations(range(n)):
        Ap = permuted_matrix(A, perm)
        if leading_principal_minors_nonzero(Ap):
            return list(perm)
    return None


def print_permutation_explanation(perm):
    print("Giai thich hoan vi:")
    for new_pos, old_pos in enumerate(perm, start=1):
        print(f"  Dong/cot moi {new_pos} lay tu dong/cot cu {old_pos + 1}")
    print("  Nghia la ta doi cung luc dong va cot theo cung mot thu tu moi de giu nguyen tinh chat cua ma tran.")


def inverse_adjoint_verbose(A):
    if not is_square(A):
        print("Ma tran khong vuong nen khong the dung ma tran phan phu.")
        return None
    n = A.shape[0]
    detA = float(np.linalg.det(A))
    print_matrix(A, "Ma tran A")
    print(f"det(A) = {detA:.10f}")
    if abs(detA) < EPS:
        print("A suy bien nen khong co ma tran nghich dao bang phan phu.")
        return None
    cofactor = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            cofactor[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    adj = cofactor.T
    print_matrix(cofactor, "Ma tran phu dai so (cofactor)")
    print_matrix(adj, "Ma tran phan phu (adjoint)")
    invA = adj / detA
    print_matrix(invA, "Ma tran nghich dao A^-1 = adj(A)/det(A)")
    return invA


def solve_gauss_verbose(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    m, n = A.shape
    print_augmented(A, b, "Ban dau [A|b]")
    pivot_cols = []
    row = 0
    for col in range(n):
        pivot = None
        best = EPS
        for r in range(row, m):
            if abs(A[r, col]) > best:
                best = abs(A[r, col])
                pivot = r
        if pivot is None:
            print(f"Cot {col + 1} toan so 0, bo qua.")
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            b[[row, pivot]] = b[[pivot, row]]
            print(f"Doi cho hang {row + 1} va hang {pivot + 1}")
            print_augmented(A, b, f"Sau khi doi cho hang {row + 1} va hang {pivot + 1}")
        pivot_val = A[row, col]
        pivot_cols.append(col)
        for r in range(row + 1, m):
            if abs(A[r, col]) > EPS:
                factor = A[r, col] / pivot_val
                print(f"Khu hang {r + 1}: h{r + 1} = h{r + 1} - ({factor:.6f})*h{row + 1}")
                A[r, col:] -= factor * A[row, col:]
                b[r] -= factor * b[row]
        print_augmented(A, b, f"Sau khi khu cot {col + 1}")
        row += 1
        if row == m:
            break

    print("Mang cot chot:", [c + 1 for c in pivot_cols])
    rankA = 0
    inconsistent = False
    for i in range(m):
        if np.all(np.abs(A[i]) < EPS):
            if abs(b[i]) > EPS:
                inconsistent = True
                break
        else:
            rankA += 1
    if inconsistent:
        print("KET LUAN: He vo nghiem.")
        return None
    rankAb = rankA
    if rankA < n:
        free_vars = [j for j in range(n) if j not in pivot_cols]
        print("KET LUAN: He co vo so nghiem.")
        print("Bien tu do:", ", ".join(f"x{j + 1}" for j in free_vars))
    else:
        print("KET LUAN: He co nghiem duy nhat.")
    x = np.zeros(n)
    for i in range(len(pivot_cols) - 1, -1, -1):
        r = i
        c = pivot_cols[i]
        rhs = b[r] - np.dot(A[r, c + 1:], x[c + 1:])
        x[c] = rhs / A[r, c]
    print("Nghiem gan dung / nghiem duy nhat:")
    for i in range(n):
        print(f"x{i + 1} = {x[i]:.10f}")
    return x


def solve_gauss_jordan_verbose(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    m, n = A.shape
    print_augmented(A, b, "Ban dau [A|b]")
    row = 0
    pivot_cols = []
    for col in range(n):
        pivot = None
        best = EPS
        for r in range(row, m):
            if abs(A[r, col]) > best:
                best = abs(A[r, col])
                pivot = r
        if pivot is None:
            print(f"Cot {col + 1} toan so 0, bo qua.")
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            b[[row, pivot]] = b[[pivot, row]]
            print(f"Doi cho hang {row + 1} va hang {pivot + 1}")
        pivot_val = A[row, col]
        print(f"Chon pivot tai hang {row + 1}, cot {col + 1} = {pivot_val:.6f}")
        A[row, col:] /= pivot_val
        b[row] /= pivot_val
        print_augmented(A, b, f"Sau khi chuan hoa hang {row + 1}")
        for r in range(m):
            if r != row and abs(A[r, col]) > EPS:
                factor = A[r, col]
                print(f"Khu hang {r + 1}: h{r + 1} = h{r + 1} - ({factor:.6f})*h{row + 1}")
                A[r, col:] -= factor * A[row, col:]
                b[r] -= factor * b[row]
        print_augmented(A, b, f"Sau khi khu xong cot {col + 1}")
        pivot_cols.append(col)
        row += 1
        if row == m:
            break

    inconsistent = False
    for i in range(m):
        if np.all(np.abs(A[i]) < EPS) and abs(b[i]) > EPS:
            inconsistent = True
            break
    if inconsistent:
        print("KET LUAN: He vo nghiem.")
        return None
    if len(pivot_cols) < n:
        free_vars = [j for j in range(n) if j not in pivot_cols]
        print("KET LUAN: He co vo so nghiem.")
        print("Bien tu do:", ", ".join(f"x{j + 1}" for j in free_vars))
    else:
        print("KET LUAN: He co nghiem duy nhat.")
    x = np.zeros(n)
    for i in range(len(pivot_cols) - 1, -1, -1):
        c = pivot_cols[i]
        x[c] = b[i] - np.dot(A[i, c + 1:], x[c + 1:])
    print("Nghiem:")
    for i in range(n):
        print(f"x{i + 1} = {x[i]:.10f}")
    return x


def lu_doolittle_verbose(A):
    if not is_square(A):
        print("LU can ma tran vuong.")
        return None, None, None
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)
    print_matrix(A, "Ma tran A ban dau")
    for k in range(n - 1):
        pivot = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[pivot, k]) < EPS:
            print("Phat hien cot pivot = 0, LU khong kha thi tren ma tran nay.")
            print("Goi y: neu muon tiep tuc, hay chuyen sang Gauss/Gauss-Jordan de xet hang va nghiem.")
            print("Neu de bai co the doi du lieu, can kiem tra lai ma tran vao vi cot hien tai dang toan so 0 tu hang k tro di.")
            return None, None, None
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
            print(f"Doi cho hang {k + 1} va hang {pivot + 1} trong U va P")
            print_matrix(U, f"U sau khi doi cho buoc {k + 1}")
        for i in range(k + 1, n):
            if abs(U[k, k]) < EPS:
                print("Pivot bang 0, khong the tiep tuc LU.")
                print("Goi y: dung Gauss/Gauss-Jordan de xu ly truong hop nay, hoac sap xep lai ma tran neu de bai cho phep.")
                return None, None, None
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
            print(f"Khu U[{i + 1},{k + 1}] bang he so {factor:.6f}")
        print_matrix(L, f"L sau buoc {k + 1}")
        print_matrix(U, f"U sau buoc {k + 1}")
    return P, L, U


def solve_lu(A, b):
    P, L, U = lu_doolittle_verbose(A)
    if P is None:
        return None
    pb = P @ b
    print("Vector Pb:")
    print(pb)
    y = np.zeros_like(b, dtype=float)
    for i in range(len(b)):
        y[i] = pb[i] - np.dot(L[i, :i], y[:i])
    x = np.zeros_like(b, dtype=float)
    for i in range(len(b) - 1, -1, -1):
        if abs(U[i, i]) < EPS:
            print("U co pivot 0, khong the giai tiep bang LU.")
            print("Goi y: day la dau hieu ma tran co the suy bien hoac can chon phuong phap khac nhu Gauss/Gauss-Jordan.")
            return None
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    print("Nghiem tu LU:")
    for i in range(len(x)):
        print(f"x{i + 1} = {x[i]:.10f}")
    return x


def cholesky_verbose(A):
    if not is_spd(A):
        if not is_symmetric(A):
            print("Ma tran khong doi xung nen khong dung duoc Cholesky.")
            print("Goi y: kiem tra lai du lieu nhap; neu de bai khong doi xung thi hay dung LU hoac Gauss.")
        else:
            print("Ma tran doi xung nhung khong xac dinh duong nen khong dung duoc Cholesky.")
            print("Goi y: Cholesky chi dung khi ma tran doi xung xac dinh duong; hay chuyen sang LU hoac Gauss-Gauss-Jordan.")
        return None
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    print_matrix(A, "Ma tran A")
    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                val = A[i, i] - s
                if val <= EPS:
                    print("Khong the phan tach Cholesky.")
                    return None
                L[i, j] = math.sqrt(val)
                print(f"L[{i + 1},{j + 1}] = sqrt({A[i,i]:.6f} - {s:.6f}) = {L[i, j]:.6f}")
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
                print(f"L[{i + 1},{j + 1}] = ({A[i,j]:.6f} - {s:.6f}) / {L[j,j]:.6f} = {L[i, j]:.6f}")
        print_matrix(L, f"L sau khi tinh hang {i + 1}")
    return L
    
def bordering_inverse_verbose(A):
    if not is_square(A):
        print("Phuong phap vien quanh can ma tran vuong.")
        return None
    n = A.shape[0]
    perm = None
    if not leading_principal_minors_nonzero(A):
        perm = find_bordering_permutation(A)
        if perm is None:
            print("Khong tim duoc hoan vi hang/cot de cac dinh thuc con chinh cap dan deu khac 0.")
            print("Phuong phap vien quanh khong ap dung duoc voi ma tran nay theo dang hien tai.")
            return None
        print("Phat hien cac dinh thuc con chinh goc bi bang 0.")
        print("Tim thay hoan vi de dua ve dang co the dung vien quanh.")
        print("Thu tu moi:", [i + 1 for i in perm])
        print_permutation_explanation(perm)
        A = permuted_matrix(A, perm)
        print_matrix(A, "Ma tran A sau khi hoan vi dong/cot")
    print_matrix(A, "Ma tran A")
    if abs(A[0, 0]) < EPS:
        print("Phan tu dau tien bang 0, phuong phap vien quanh khong an toan tren ma tran nay.")
        return None
    inv = np.array([[1.0 / A[0, 0]]], dtype=float)
    print(f"Buoc 1: A1^-1 = [1/{A[0,0]:.6f}] = [{inv[0,0]:.6f}]")
    print_matrix(inv, "Nghich dao cap 1")
    for k in range(2, n + 1):
        Ak = A[:k, :k]
        B = inv
        u = Ak[:k - 1, k - 1].reshape(-1, 1)
        v = Ak[k - 1, :k - 1].reshape(1, -1)
        alpha = Ak[k - 1, k - 1]
        print(f"\nMoi rong len cap {k}:")
        print_matrix(Ak, f"A_{k}")
        print_matrix(u, "Vector cot u")
        print_matrix(v, "Vector hang v")
        print(f"alpha = {alpha:.6f}")
        r = B @ u
        s = v @ B
        schur = float(alpha - (v @ r)[0, 0])
        print_matrix(r, "r = B*u")
        print_matrix(s, "s = v*B")
        print(f"Schur = alpha - v*B*u = {schur:.10f}")
        if abs(schur) < EPS:
            print("Schur bang 0, khong the tiep tuc phuong phap vien quanh.")
            return None
        top_left = B + (r @ s) / schur
        top_right = -r / schur
        bottom_left = -s / schur
        bottom_right = np.array([[1.0 / schur]], dtype=float)
        inv = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])
        print_matrix(inv, f"Nghich dao cap {k}")
    if perm is not None:
        P = np.eye(n)[perm]
        inv = P.T @ inv @ P
        print_matrix(inv, "Ma tran nghich dao sau khi hoan vi ve thu tu goc")
    return inv


def solve_cholesky(A, b):
    L = cholesky_verbose(A)
    if L is None:
        return None
    y = np.zeros_like(b, dtype=float)
    for i in range(len(b)):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    x = np.zeros_like(b, dtype=float)
    LT = L.T
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i + 1:], x[i + 1:])) / LT[i, i]
    print("Nghiem tu Cholesky:")
    for i in range(len(x)):
        print(f"x{i + 1} = {x[i]:.10f}")
    return x
    
def solve_bordering(A, b):
    invA = bordering_inverse_verbose(A)
    if invA is None:
        return None
    x = invA @ b
    print("Nghiem tu phuong phap vien quanh:")
    for i in range(len(x)):
        print(f"x{i + 1} = {x[i]:.10f}")
    return x


def jacobi_verbose(A, b, x0=None, max_iter=50, tol=1e-8):
    if not is_square(A):
        print("Jacobi can ma tran vuong.")
        return None
    if np.any(np.abs(np.diag(A)) < EPS):
        perm = find_row_permutation_nonzero_diagonal(A)
        if perm is None:
            print("Jacobi khong dung duoc vi co phan tu duong cheo bang 0 va khong tim duoc cach doi dong de sua.")
            print("Goi y: thu doi thu tu cac phuong trinh neu de bai cho phep, neu khong hay chon Gauss/Gauss-Jordan.")
            return None
        print("Jacobi bi chan vi duong cheo co so 0, nhung tim thay cach doi dong de xu ly tiep.")
        print("Thu tu dong moi:", [i + 1 for i in perm])
        print_row_permutation_explanation(perm)
        A, b = apply_row_permutation(A, b, perm)
    if x0 is None:
        x = np.zeros(len(b), dtype=float)
    else:
        x = np.array(x0, dtype=float)
    print_matrix(A, "Ma tran A")
    print("Cot dong lap Jacobi:")
    print(f"{'k':>3} | {'x':<40} | {'sai so'}")
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(max_iter):
        x_new = (b - R @ x) / D
        err = np.linalg.norm(x_new - x, ord=np.inf)
        print(f"{k:>3} | {np.array2string(x_new, precision=8, suppress_small=True):<40} | {err:.3e}")
        if err < tol:
            print("Jacobi hoi tu.")
            return x_new
        x = x_new
    print("Jacobi dung do dat gioi han lap.")
    return x


def gauss_seidel_verbose(A, b, x0=None, max_iter=50, tol=1e-8):
    if not is_square(A):
        print("Gauss-Seidel can ma tran vuong.")
        return None
    if np.any(np.abs(np.diag(A)) < EPS):
        perm = find_row_permutation_nonzero_diagonal(A)
        if perm is None:
            print("Gauss-Seidel khong dung duoc vi co phan tu duong cheo bang 0 va khong tim duoc cach doi dong de sua.")
            print("Goi y: doi thu tu phuong trinh neu co the, hoac chuyen sang Gauss/Gauss-Jordan de bien luan he.")
            return None
        print("Gauss-Seidel bi chan vi duong cheo co so 0, nhung tim thay cach doi dong de xu ly tiep.")
        print("Thu tu dong moi:", [i + 1 for i in perm])
        print_row_permutation_explanation(perm)
        A, b = apply_row_permutation(A, b, perm)
    n = len(b)
    x = np.zeros(n, dtype=float) if x0 is None else np.array(x0, dtype=float)
    print_matrix(A, "Ma tran A")
    print("Cot dong lap Gauss-Seidel:")
    print(f"{'k':>3} | {'x':<40} | {'sai so'}")
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        err = np.linalg.norm(x - x_old, ord=np.inf)
        print(f"{k:>3} | {np.array2string(x, precision=8, suppress_small=True):<40} | {err:.3e}")
        if err < tol:
            print("Gauss-Seidel hoi tu.")
            return x
    print("Gauss-Seidel dung do dat gioi han lap.")
    return x


def power_method_verbose(A, x0=None, max_iter=100, tol=1e-10):
    if not is_square(A):
        print("PP luy thua can ma tran vuong.")
        return None, None
    n = A.shape[0]
    x = np.ones(n, dtype=float) if x0 is None else np.array(x0, dtype=float)
    x = x / np.linalg.norm(x)
    lam_old = None
    print_matrix(A, "Ma tran A")
    print(f"{'k':>3} | {'lambda':>15} | {'||x||':>10} | vector chuan hoa")
    for k in range(max_iter):
        y = A @ x
        normy = np.linalg.norm(y)
        if normy < EPS:
            print("Vector bi chuyen ve 0, khong the tiep tuc.")
            print("Goi y: doi vector khoi tao x0, kiem tra lai ma tran, hoac chon phuong phap khac neu hang cua A khong phu hop.")
            return None, None
        x_new = y / normy
        lam = float(x_new @ (A @ x_new))
        print(f"{k:>3} | {lam:15.8f} | {np.linalg.norm(x_new):10.6f} | {np.array2string(x_new, precision=8, suppress_small=True)}")
        if lam_old is not None and abs(lam - lam_old) < tol:
            print("PP luy thua hoi tu.")
            return lam, x_new
        lam_old = lam
        x = x_new
    print("PP luy thua dung do dat gioi han lap.")
    return lam_old, x


def next_eigen_by_deflation(A):
    if not is_symmetric(A):
        print("Phan giai tim gia tri rieng tiep theo chi an toan cho ma tran doi xung.")
        return None
    print("Tim gia tri rieng troi bang PP luy thua...")
    lam1, v1 = power_method_verbose(A)
    if lam1 is None:
        return None
    v1 = v1 / np.linalg.norm(v1)
    print(f"Gia tri rieng troi: lambda1 = {lam1:.10f}")
    print("Tao ma tran giam cap (deflation): B = A - lambda1 * v1 * v1^T")
    B = A - lam1 * np.outer(v1, v1)
    print_matrix(B, "Ma tran sau deflation")
    lam2, v2 = power_method_verbose(B)
    if lam2 is None:
        return None
    print(f"Gia tri rieng tiep theo xap xi: lambda2 = {lam2:.10f}")
    return lam2, v2


def faddeev_leverrier_verbose(A):
    if not is_square(A):
        print("Da thuc dac trung can ma tran vuong.")
        return None
    n = A.shape[0]
    B = np.eye(n)
    coeffs = []
    print_matrix(A, "Ma tran A")
    for k in range(1, n + 1):
        c = -np.trace(A @ B) / k
        coeffs.append(c)
        print(f"Buoc {k}: c{k} = {-np.trace(A @ B):.8f} / {k} = {c:.8f}")
        B = A @ B + c * np.eye(n)
        print_matrix(B, f"B{k}")
    poly = [1.0] + coeffs
    print("Da thuc dac trung:")
    terms = []
    deg = n
    for i, c in enumerate(poly):
        power = deg - i
        if abs(c) < EPS:
            continue
        sign = "+" if c >= 0 else "-"
        val = abs(c)
        if power == 0:
            term = f"{val:.8f}"
        elif power == 1:
            term = f"{val:.8f}*λ"
        else:
            term = f"{val:.8f}*λ^{power}"
        terms.append((sign, term))
    expr = ""
    for idx, (sign, term) in enumerate(terms):
        if idx == 0 and sign == "+":
            expr += term
        else:
            expr += f" {sign} {term}"
    print("p(λ) =", expr)
    return poly


def choose_linear_solver(A, b):
    print("Kiem tra dieu kien de tu dong chon phuong phap...")
    if not is_square(A):
        print("A khong vuong -> chon Gauss/Gauss-Jordan.")
        return "gauss"
    detA = float(np.linalg.det(A))
    print(f"det(A) = {detA:.10f}")
    if is_spd(A):
        print("A doi xung xac dinh duong -> Cholesky.")
        return "cholesky"
    if leading_principal_minors_nonzero(A):
        print("Cac dinh thuc con chinh cap dan deu khac 0 -> Phuong phap vien quanh.")
        return "bordering"
    if is_strictly_diagonally_dominant(A):
        print("A cheo troi nghiem ngat -> uu tien Gauss-Seidel, Jacobi canh bao.")
        return "seidel"
    if abs(detA) < EPS:
        print("A suy bien -> Gauss/Gauss-Jordan de bien luan he.")
        return "gauss"
    print("A vuong, khong suy bien -> LU (co pivot) la lua chon thong dung.")
    return "lu"


def solve_linear_system_exam():
    print("=== GIAI HE Ax=b (BAN DI THI) ===")
    m = read_int("Nhap so hang m cua A: ")
    n = read_int("Nhap so cot n cua A: ")
    p = read_int("Nhap so cot p cua B (neu he 1 ve phai thi nhap 1): ")
    if p <= 0 or m <= 0 or n <= 0:
        print("m, n, p phai la so duong.")
        return
    A = read_matrix(m, n, "A")
    B = read_matrix(m, p, "B")
    if B.shape[0] != m:
        print("So hang A va B khong khop.")
        return
    if p != 1:
        print("Luu y: ban dang nhap nhieu ve phai. Chuong trinh se xu ly tung cot cua B.")
    method = choose_linear_solver(A, B[:, 0]) if p == 1 else choose_linear_solver(A, B[:, 0])
    if method == "cholesky":
        if p != 1:
            print("Cholesky trong ban thi nay dang ap dung cho ve phai 1 cot de de trinh bay.")
        solve_cholesky(A, B[:, 0])
    elif method == "bordering":
        if p != 1:
            print("Phuong phap vien quanh trong ban thi nay dang ap dung cho ve phai 1 cot de de trinh bay.")
        solve_bordering(A, B[:, 0])
    elif method == "lu":
        if p != 1:
            print("LU trong ban thi nay dang ap dung cho ve phai 1 cot de de trinh bay.")
        solve_lu(A, B[:, 0])
    elif method == "seidel":
        x0 = np.zeros(A.shape[1], dtype=float)
        gauss_seidel_verbose(A, B[:, 0], x0=x0)
    elif method == "gauss":
        solve_gauss_verbose(A, B[:, 0])
    else:
        solve_gauss_jordan_verbose(A, B[:, 0])



def eigen_exam():
    print("=== GIA TRI RIENG VA DA THUC DAC TRUNG ===")
    n = read_int("Nhap cap ma tran vuong n: ")
    A = read_matrix(n, n, "A")
    print("Chon che do:")
    print("1. PP luy thua tim gia tri rieng troi")
    print("2. PP down-thang / deflation tim gia tri rieng tiep theo")
    print("3. Da thuc dac trung (Faddeev-LeVerrier)")
    print("4. Chay tat ca (neu hop le)")
    ch = read_int("Lua chon: ")
    if ch == 1:
        power_method_verbose(A)
    elif ch == 2:
        next_eigen_by_deflation(A)
    elif ch == 3:
        faddeev_leverrier_verbose(A)
    else:
        lam1, v1 = power_method_verbose(A)
        if lam1 is not None and is_symmetric(A):
            next_eigen_by_deflation(A)
        faddeev_leverrier_verbose(A)



def menu():
    print("=== BAN DI THI GIẢI TÍCH SỐ - MA TRAN ===")
    print("1. Giai Ax=b tu dong (Gauss / LU / Cholesky / Jacobi / Seidel)")
    print("2. Gia tri rieng va da thuc dac trung")
    print("3. Ma tran phan phu nghich dao")
    choice = read_int("Chon muc: ")
    if choice == 1:
        solve_linear_system_exam()
    elif choice == 2:
        eigen_exam()
    elif choice == 3:
        n = read_int("Nhap cap ma tran vuong n: ")
        A = read_matrix(n, n, "A")
        inverse_adjoint_verbose(A)
    else:
        print("Lua chon khong hop le.")


if __name__ == "__main__":
    menu()
