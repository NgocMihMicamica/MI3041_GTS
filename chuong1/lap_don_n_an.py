import numpy as np
import math
import sympy as sp
import re

MAX_ITER = 100

def trunc8(val):
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

def chuan_vo_cung(vector):
    return max(abs(x) for x in vector)

def lap_don_nan(n, phi_strs, x0, ep, q=None, filename="kq_lap_don_nan.txt"):
    """
    phi_strs: list of n strings, each is phi_i(x1, x2, ..., xn)
    """
    symbols = sp.symbols(' '.join([f'x{i+1}' for i in range(n)]))

    phi_exprs = [sp.sympify(s) for s in phi_strs]

    # Tự động tính đạo hàm riêng
    jacobi = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(sp.diff(phi_exprs[i], symbols[j]))
        jacobi.append(row)

    def eval_phi(values):
        return [float(phi_exprs[i].subs({symbols[j]: values[j] for j in range(n)}).evalf(15)) for i in range(n)]

    def tinh_jacobi_uoc_luong_q(x_val):
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                J[i,j] = float(jacobi[i][j].subs({symbols[k]: x_val[k] for k in range(n)}).evalf(15))
        tong_hang = [sum(abs(J[i,j]) for j in range(n)) for i in range(n)]
        return max(tong_hang)

    with open(filename, "w", encoding="utf-8") as f:
        def log(message):
            print(message)
            f.write(message + "\n")

        log("="*100)
        log(f" PHƯƠNG PHÁP LẶP ĐƠN {n} ẨN - TỰ ĐỘNG TÍNH ĐẠO HÀM")
        log("="*100)
        log("I. HỆ PHƯƠNG TRÌNH DẠNG LẶP:")
        log("-"*100)
        for i in range(n):
            log(f"   x_{i+1} = {phi_exprs[i]}")
        log("")
        log("   Ma trận Jacobi của hệ lặp:")
        for i in range(n):
            row_str = "   "
            for j in range(n):
                row_str += f"∂φ_{i+1}/∂x_{j+1} = {str(jacobi[i][j])}  "
            log(row_str)
        log("-"*100)

        # Tính q
        if q is None:
            q_calc = tinh_jacobi_uoc_luong_q(x0)
            log(f"\n[TỰ ĐỘNG TÍNH q] Tại X⁰: ‖Jφ(X⁰)‖∞ = {q_calc:.6f}")
            if q_calc < 1:
                q = q_calc
                log(f"   → Thỏa mãn ánh xạ co. Sử dụng q = {q:.6f}")
                log(f"   Biện luận: Vì q < 1, theo định lý ánh xạ co của Banach, dãy lặp sẽ hội tụ.")
            else:
                log(f"\n❌ NGHIÊM TÚC: q = {q_calc:.6f} ≥ 1")
                log("   HỆ KHÔNG PHẢI ÁNH XẠ CO - LẶP ĐƠN SẼ PHÂN KỲ!")
                log("   BẠN PHẢI BIẾN ĐỔI LẠI HỆ PHƯƠNG TRÌNH TRƯỚC KHI CHẠY.")
                return
        else:
            if not (0 < q < 1):
                log(f"\n❌ NGHIÊM TÚC: q = {q} không nằm trong (0,1)")
                log("   HỆ KHÔNG THỎA MÃN ĐIỀU KIỆN ÁNH XẠ CO.")
                return
            log(f"\n[NHẬP TAY] q = {q:.6f}")

        log("")
        log("II. CÔNG THỨC SAI SỐ HẬU NGHIỆM:")
        log(f"   Δₖ = [q/(1-q)] · ‖X⁽ᵏ⁾ - X⁽ᵏ⁻¹⁾‖∞")
        log(f"   Điều kiện dừng: Δₖ ≤ εpsilon = {ep}")
        log("="*100)
        log("")
        log(f"ĐIỂM KHỞI TẠO: X⁽⁰⁾ = {x0}")
        log("")

        # Header bảng
        header = f"{'k':<4} |"
        for i in range(n):
            header += f" {'x_{i+1}⁽ᵏ⁾':<14} |"
        header += f" {'‖dX‖∞':<14} | {'Δₖ':<14} | {'Nhận xét':<15}"

        log("-"*100)
        log(header)
        log("-"*100)

        x = list(x0)
        log(f"{'0':<4} |" + "".join([f" {trunc8(x[i]):<14} |" for i in range(n)]) + f" {'-':<14} | {'-':<14} | {'Khởi tạo':<15}")

        k = 1
        history = []

        while True:
            if k > MAX_ITER:
                log("-"*100)
                log(f"🛑 DỪNG: Đạt giới hạn {MAX_ITER} bước lặp.")
                break

            x_old = x.copy()
            x = eval_phi(x_old)

            ly_do = chuan_vo_cung([x[i] - x_old[i] for i in range(n)])
            saiso = (q / (1 - q)) * ly_do

            if saiso <= ep:
                nx = "✅ Đạt ε"
            elif saiso <= ep * 10:
                nx = "Gần đạt"
            else:
                nx = "Lặp tiếp"

            row = f"{k:<4} |"
            for i in range(n):
                row += f" {trunc8(x[i]):<14} |"
            row += f" {ly_do:<14.6e} | {saiso:<14.6e} | {nx:<15}"
            log(row)
            history.append((k, x.copy(), ly_do, saiso))

            if saiso <= ep:
                log("-"*100)
                log(f"✅ HỘI TỤ tại bước k={k}: Δₖ = {saiso:.6e} ≤ ε = {ep}")
                log(f"   NGHỊỆM: X* = ({', '.join([trunc8(xi) for xi in x])})")
                break

            k += 1

        # Bảng tổng hợp
        log("")
        log("="*100)
        log(f" BẢNG TỔNG HỢP TIẾN TRÌNH LẶP ĐƠN {n} ẨN")
        log("="*100)

        header2 = f"{'k':<4} |"
        for i in range(n):
            header2 += f" {'x_{i+1}⁽ᵏ⁾':<14} |"
        header2 += f" {'‖dX‖∞':<14} | {'Δₖ':<16} | {'Đánh giá':<20}"
        log(header2)
        log("-"*100)

        for row in history:
            k_v, x_v, d_v, s_v = row
            if s_v <= ep:
                dg = "✅ Đủ chính xác"
            elif s_v <= ep * 100:
                dg = "Chấp nhận được"
            else:
                dg = "Cần lặp thêm"
            out = f"{k_v:<4} |"
            for i in range(n):
                out += f" {trunc8(x_v[i]):<14} |"
            out += f" {d_v:<14.6e} | {s_v:<16.6e} | {dg:<20}"
            log(out)
        log("="*100)
        log(f"📁 Đã lưu kết quả vào: {filename}")

if __name__ == "__main__":
    print(">>> LẶP ĐƠN n ẨN - NHẬP PHƯƠNG TRÌNH DẠNG LẶP")
    print("Mẹo nhập: dùng * để nhân, ** để mũ, x1, x2, x3,...")

    n = nhap_so_nguyen("Nhập số ẩn n: ")
    phi_strs = []
    for i in range(n):
        eq = input(f"Nhập φ_{i+1}(x1,...,x{n}): ").strip()
        phi_strs.append(eq)

    x0 = nhap_diem_khoi_tao(n)
    ep = float(input("Nhập epsilon (VD: 1e-6): ").strip())

    q_input = input("Nhập q (để trống để tự động tính): ").strip()
    q = float(q_input) if q_input else None

    lap_don_nan(n, phi_strs, x0, ep, q)
