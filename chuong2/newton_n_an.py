import numpy as np
import math
import sympy as sp
import re

EPS = 1e-15
COND_THRESHOLD = 1000

def trunc8(val):
    if math.isnan(val) or math.isinf(val):
        return str(val).ljust(12)
    if val == 0.0 or val == -0.0: return "0.00000000"
    s = f"{val:.15f}"
    return s[:s.index('.') + 9]

def nhap_so_thuc(thong_bao):
    while True:
        try:
            return float(input(thong_bao).strip())
        except ValueError:
            print("Vui lòng nhập số thực hợp lệ.")

def nhap_so_nguyen(thong_bao):
    while True:
        try:
            return int(input(thong_bao).strip())
        except ValueError:
            print("Vui lòng nhập số nguyên hợp lệ.")

def nhap_diem_khoi_tao(n, prompt="Nhập điểm khởi tạo X0"):
    raw = input(f"{prompt} (VD: 0, 1, 0.5 hoặc nhập từng số): ").strip()
    raw = raw.strip('()[]{}')
    parts = re.split(r'[,\s]+', raw)
    parts = [p for p in parts if p != '']

    if len(parts) == n:
        try:
            return [float(p) for p in parts]
        except ValueError:
            pass

    print(f"Không hiểu định dạng. Vui lòng nhập từng số riêng lẻ:")
    result = []
    for i in range(1, n + 1):
        result.append(nhap_so_thuc(f"  Nhập x{i} khởi tạo: "))
    return result

def newton_nan(f_strs, x0, n, mode="thuong", filename=None):
    """
    mode: "thuong" hoặc "cai_bien"
    """
    if filename is None:
        filename = f"kq_newton_{mode}_{n}an.txt"

    symbols = sp.symbols(' '.join([f'x{i+1}' for i in range(n)]))
    f_exprs = [sp.sympify(s) for s in f_strs]

    # Tính đạo hàm riêng
    jacobi = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(sp.diff(f_exprs[i], symbols[j]))
        jacobi.append(row)

    def evaluate_F(values):
        return np.array([float(f_exprs[i].subs({symbols[j]: values[j] for j in range(n)}).evalf(15)) for i in range(n)], dtype=float)

    def evaluate_J(values):
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                J[i,j] = float(jacobi[i][j].subs({symbols[k]: values[k] for k in range(n)}).evalf(15))
        return J

    x = np.array(x0, dtype=float)
    history = []

    with open(filename, "w", encoding="utf-8") as f:
        def log(message):
            print(message)
            f.write(message + "\n")

        title = "NEWTON CẢI BIẾN" if mode == "cai_bien" else "NEWTON THƯỜNG"
        log("="*110)
        log(f" PHƯƠNG PHÁP {title} {n} ẨN - TỰ ĐỘNG TÍNH ĐẠO HÀM")
        log("="*110)
        log("I. HỆ PHƯƠNG TRÌNH GỐC:")
        log("-"*110)
        for i in range(n):
            log(f"   f_{i+1} = {f_exprs[i]} = 0")
        log("")
        log("   Ma trận Jacobi:")
        for i in range(n):
            row_str = "   "
            for j in range(n):
                row_str += f"∂f_{i+1}/∂x_{j+1}={str(jacobi[i][j])}  "
            log(row_str)
        log("-"*110)

        if mode == "cai_bien":
            log("II. CÔNG THỨC NEWTON CẢI BIẾN:")
            log("   X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾ - J⁻¹(X⁽⁰⁾) · F(X⁽ᵏ⁾)")
            log("   → J(X⁰) và J⁻¹(X⁰) chỉ tính MỘT LẦN DUY NHẤT")
        else:
            log("II. CÔNG THỨC NEWTON THƯỜNG:")
            log("   X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾ - J⁻¹(X⁽ᵏ⁾) · F(X⁽ᵏ⁾)")
            log("   → J(X) được tính LẠI Ở MỖI BƯỚC")
        log("="*110)
        log(f"ĐIỂM KHỞI TẠO: X⁽⁰⁾ = {x0}")
        log(f"SỐ BƯỚC LẶP: n = {n}")
        log("")

        # Nếu cải biến: tính J0 một lần
        if mode == "cai_bien":
            log("="*110)
            log("■ TÍNH J(X⁽⁰⁾) VÀ J⁻¹(X⁽⁰⁾) - CHỈ MỘT LẦN")
            log("-"*110)
            J0 = evaluate_J(x0)
            det_J0 = np.linalg.det(J0)
            log(f"  J(X⁽⁰⁾) =")
            for row in J0:
                log("    [" + ", ".join([trunc8(v) for v in row]) + "]")
            log(f"  det(J⁰) = {det_J0:.6e}")

            if abs(det_J0) < EPS:
                log("\n❌ DỪNG: Jacobian suy biến tại X⁰!")
                return
            try:
                J0_inv = np.linalg.inv(J0)
            except:
                log("\n❌ LỖI: Không tính được nghịch đảo J⁰")
                return
            log("  J⁻¹(X⁰) đã tính xong. DÙNG LẠI cho mọi bước.")
            log("="*110)
            log("")

        dx_old = 0

        for i in range(n + 1):
            log("="*110)
            log(f"■ BƯỚC LẶP k = {i}:")
            log("-"*110)

            Fx = evaluate_F(x)
            norm_F = np.linalg.norm(Fx, ord=np.inf)
            log(f"  Bước 1: F(X⁽{i}⁾) = [{', '.join([trunc8(v) for v in Fx])}]ᵀ")
            log(f"    → ‖F‖∞ = {norm_F:.6e}")

            history.append((i, x.copy(), norm_F))

            if i == n:
                log(f"    → Đạt số bước lặp tối đa.")
                break

            if mode == "thuong":
                Jx = evaluate_J(x)
                det_J = np.linalg.det(Jx)
                log("")
                log(f"  Bước 2: J(X⁽{i}⁾) =")
                for row in Jx:
                    log("    [" + ", ".join([trunc8(v) for v in row]) + "]")
                log(f"    det(J) = {det_J:.6e}")

                if abs(det_J) < EPS:
                    log(f"    ❌ DỪNG: Jacobian suy biến!")
                    break
                try:
                    J_inv = np.linalg.inv(Jx)
                except:
                    log(f"    ❌ LỖI nghịch đảo!")
                    break
            else:
                J_inv = J0_inv
                log("")
                log(f"  Bước 2: DÙNG LẠI J⁻¹(X⁰) đã tính")

            delta_X = -np.dot(J_inv, Fx)
            x_new = x + delta_X
            norm_delta = np.linalg.norm(delta_X, ord=np.inf)

            log("")
            log(f"  Bước 3: dX⁽{i}⁾ = [{', '.join([trunc8(v) for v in delta_X])}]ᵀ")
            log(f"    ‖dX⁽{i}⁾‖∞ = {norm_delta:.6e}")

            # Hậu nghiệm
            log("")
            log("="*80)
            log(f"  📐 ĐÁNH GIÁ HẬU NGHIỆM TẠI BƯỚC k = {i}:")
            log("="*80)

            if mode == "cai_bien" and i > 0 and dx_old > 0:
                q_est = norm_delta / dx_old
                log(f"  q ≈ ‖dX⁽{i}⁾‖/‖dX⁽{i-1}⁾‖ = {q_est:.8f}")
                if q_est < 1:
                    hau = (q_est / (1 - q_est)) * norm_delta
                    log(f"  Δₖ = [q/(1-q)]·‖dX⁽{i}⁾‖ = {hau:.8e}")
                else:
                    log(f"  ‖X⁽{i}⁾ - X*‖∞ ≈ ‖dX⁽{i}⁾‖∞ = {norm_delta:.8e}")
            else:
                log(f"  ‖X⁽{i}⁾ - X*‖∞ ≈ ‖dX⁽{i}⁾‖∞ = {norm_delta:.8e}")
            log("="*80)

            log("")
            log(f"  Bước 4: X⁽{i+1}⁾ = [{', '.join([trunc8(v) for v in x_new])}]ᵀ")

            dx_old = norm_delta
            x = x_new

        # Bảng tổng hợp
        log("")
        log("="*110)
        log(f" BẢNG TỔNG HỢP TIẾN TRÌNH LẶP {title} {n} ẨN")
        log("="*110)
        header = f"{'k':<4} |"
        for j in range(n):
            header += f" {'x_{j+1}⁽ᵏ⁾':<14} |"
        header += f" {'‖F‖∞':<14} | {'Nhận xét':<20}"
        log(header)
        log("-"*110)
        for row in history:
            k_v, x_v, f_v = row
            if f_v < 1e-10:
                nx = "Đã hội tụ"
            elif f_v < 1e-6:
                nx = "Gần nghiệm"
            else:
                nx = "Cần lặp thêm"
            out = f"{k_v:<4} |"
            for j in range(n):
                out += f" {trunc8(x_v[j]):<14} |"
            out += f" {f_v:<14.6e} | {nx:<20}"
            log(out)
        log("="*110)
        log(f"✅ NGHIỆM CUỐI: X* = ({', '.join([trunc8(v) for v in x])})")
        log(f"📁 Đã lưu: {filename}")

if __name__ == "__main__":
    print(">>> NEWTON n ẨN - NHẬP PHƯƠNG TRÌNH TỪ BÀN PHÍM")
    print("Mẹo: dùng * nhân, ** mũ, x1, x2, x3,...")

    n = nhap_so_nguyen("Nhập số ẩn n: ")
    f_strs = []
    for i in range(n):
        eq = input(f"Nhập f_{i+1}(x1,...,x{n}) = 0: ").strip()
        f_strs.append(eq)

    x0 = nhap_diem_khoi_tao(n)

    print("\nChọn phương pháp:")
    print("1. Newton THƯỜNG (tính lại J mỗi bước)")
    print("2. Newton CẢI BIẾN (J cố định tại X⁰)")
    mode_choice = input("Chọn (1 hoặc 2): ").strip()
    mode = "cai_bien" if mode_choice == "2" else "thuong"

    newton_nan(f_strs, x0, n, mode)
