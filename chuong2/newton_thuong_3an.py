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

def nhap_diem_khoi_tao(dim, prompt="Nhập điểm khởi tạo X0"):
    raw = input(f"{prompt} (VD: 0, 1, 0.5 hoặc nhập từng số): ").strip()
    raw = raw.strip('()[]{}')
    parts = re.split(r'[,\s]+', raw)
    parts = [p for p in parts if p != '']

    if len(parts) == dim:
        try:
            return [float(p) for p in parts]
        except ValueError:
            pass

    print(f"Không hiểu định dạng. Vui lòng nhập từng số riêng lẻ:")
    result = []
    for i in range(1, dim + 1):
        result.append(nhap_so_thuc(f"  Nhập x{i} khởi tạo: "))
    return result

def to_scalar(val):
    """Chuyển numpy array/scalar về float thuần"""
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)

def newton_thuong_3an(f1_str, f2_str, f3_str, x0, n, filename="kq_newton_thuong_3an.txt"):
    x1, x2, x3 = sp.symbols('x1 x2 x3')

    f1_expr = sp.sympify(f1_str)
    f2_expr = sp.sympify(f2_str)
    f3_expr = sp.sympify(f3_str)

    # Tự động tính đạo hàm
    df1_dx1 = sp.diff(f1_expr, x1)
    df1_dx2 = sp.diff(f1_expr, x2)
    df1_dx3 = sp.diff(f1_expr, x3)
    df2_dx1 = sp.diff(f2_expr, x1)
    df2_dx2 = sp.diff(f2_expr, x2)
    df2_dx3 = sp.diff(f2_expr, x3)
    df3_dx1 = sp.diff(f3_expr, x1)
    df3_dx2 = sp.diff(f3_expr, x2)
    df3_dx3 = sp.diff(f3_expr, x3)

    # Lambdify để tính nhanh
    f1_func = sp.lambdify((x1, x2, x3), f1_expr, 'numpy')
    f2_func = sp.lambdify((x1, x2, x3), f2_expr, 'numpy')
    f3_func = sp.lambdify((x1, x2, x3), f3_expr, 'numpy')

    j11 = sp.lambdify((x1, x2, x3), df1_dx1, 'numpy')
    j12 = sp.lambdify((x1, x2, x3), df1_dx2, 'numpy')
    j13 = sp.lambdify((x1, x2, x3), df1_dx3, 'numpy')
    j21 = sp.lambdify((x1, x2, x3), df2_dx1, 'numpy')
    j22 = sp.lambdify((x1, x2, x3), df2_dx2, 'numpy')
    j23 = sp.lambdify((x1, x2, x3), df2_dx3, 'numpy')
    j31 = sp.lambdify((x1, x2, x3), df3_dx1, 'numpy')
    j32 = sp.lambdify((x1, x2, x3), df3_dx2, 'numpy')
    j33 = sp.lambdify((x1, x2, x3), df3_dx3, 'numpy')

    def evaluate_F(x_val, y_val, z_val):
        return np.array([
            to_scalar(f1_func(x_val, y_val, z_val)),
            to_scalar(f2_func(x_val, y_val, z_val)),
            to_scalar(f3_func(x_val, y_val, z_val))
        ], dtype=float)

    def evaluate_J(x_val, y_val, z_val):
        return np.array([
            [to_scalar(j11(x_val, y_val, z_val)), to_scalar(j12(x_val, y_val, z_val)), to_scalar(j13(x_val, y_val, z_val))],
            [to_scalar(j21(x_val, y_val, z_val)), to_scalar(j22(x_val, y_val, z_val)), to_scalar(j23(x_val, y_val, z_val))],
            [to_scalar(j31(x_val, y_val, z_val)), to_scalar(j32(x_val, y_val, z_val)), to_scalar(j33(x_val, y_val, z_val))]
        ], dtype=float)

    x = np.array(x0, dtype=float)
    history = []

    with open(filename, "w", encoding="utf-8") as f:
        def log(message):
            print(message)
            f.write(message + "\n")

        log("="*110)
        log(" PHƯƠNG PHÁP NEWTON-RAPHSON THƯỜNG 3 ẨN - TỰ ĐỘNG TÍNH ĐẠO HÀM")
        log("="*110)
        log("I. HỆ PHƯƠNG TRÌNH GỐC VÀ MA TRẬN JACOBI TỰ ĐỘNG")
        log("-"*110)
        log(f"   f₁(x₁,x₂,x₃) = {f1_expr} = 0")
        log(f"   f₂(x₁,x₂,x₃) = {f2_expr} = 0")
        log(f"   f₃(x₁,x₂,x₃) = {f3_expr} = 0")
        log("")
        log("   Các đạo hàm riêng tự động tính:")
        log(f"   ∂f₁/∂x₁ = {str(df1_dx1)}")
        log(f"   ∂f₁/∂x₂ = {str(df1_dx2)}")
        log(f"   ∂f₁/∂x₃ = {str(df1_dx3)}")
        log(f"   ∂f₂/∂x₁ = {str(df2_dx1)}")
        log(f"   ∂f₂/∂x₂ = {str(df2_dx2)}")
        log(f"   ∂f₂/∂x₃ = {str(df2_dx3)}")
        log(f"   ∂f₃/∂x₁ = {str(df3_dx1)}")
        log(f"   ∂f₃/∂x₂ = {str(df3_dx2)}")
        log(f"   ∂f₃/∂x₃ = {str(df3_dx3)}")
        log("-"*110)
        log("II. CÔNG THỨC LẶP NEWTON THƯỜNG:")
        log("   X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾ - J⁻¹(X⁽ᵏ⁾) · F(X⁽ᵏ⁾)")
        log("   → J(X) được tính LẠI Ở MỖI BƯỚC LẶP (đặc trưng Newton thường)")
        log("-"*110)
        log("III. ĐÁNH GIÁ HẬU NGHIỆM:")
        log("   Công thức: ‖X⁽ᵏ⁾ - X*‖∞ ≤ ‖dX⁽ᵏ⁾‖∞ = max(|dx₁|, |dx₂|, |dx₃|)")
        log("   Hoặc chặt chẽ: ‖X⁽ᵏ⁾ - X*‖∞ ≤ ‖J⁻¹(X⁽ᵏ⁾)‖∞ · ‖F(X⁽ᵏ⁾)‖∞")
        log("="*110)
        log("")
        log(f"ĐIỂM KHỞI TẠO: X⁽⁰⁾ = [{x0[0]}, {x0[1]}, {x0[2]}]ᵀ")
        log(f"SỐ BƯỚC LẶP: n = {n}")
        log("")

        for i in range(n):
            log("="*110)
            log(f"■ BƯỚC LẶP k = {i}:")
            log("-"*110)

            # Tính F(X⁽ᵏ⁾)
            Fx = evaluate_F(x[0], x[1], x[2])
            norm_F = np.linalg.norm(Fx, ord=np.inf)
            log(f"  Bước 1: Tính F(X⁽{i}⁾)")
            log(f"    X⁽{i}⁾ = [{trunc8(x[0])}, {trunc8(x[1])}, {trunc8(x[2])}]ᵀ")
            log(f"    F(X⁽{i}⁾) = [{trunc8(Fx[0])}, {trunc8(Fx[1])}, {trunc8(Fx[2])}]ᵀ")
            log(f"    → ‖F(X⁽{i}⁾)‖∞ = {norm_F:.6e}")

            # Tính J(X⁽ᵏ⁾)
            Jx = evaluate_J(x[0], x[1], x[2])
            det_J = np.linalg.det(Jx)
            log("")
            log(f"  Bước 2: Tính ma trận Jacobi J(X⁽{i}⁾)")
            log(f"    J(X⁽{i}⁾) =")
            for row in Jx:
                log(f"      [{trunc8(row[0])}, {trunc8(row[1])}, {trunc8(row[2])}]")
            log(f"    det(J) = {det_J:.6e}")

            try:
                cond_J = np.linalg.cond(Jx, p=np.inf)
                status = "ỔN ĐỊNH" if cond_J < COND_THRESHOLD else "CẢNH BÁO"
            except:
                cond_J = float('inf')
                status = "SUY BIẾN"

            log(f"    cond(J) = {cond_J:.2f} ({status})")
            if math.isinf(cond_J) or abs(det_J) < EPS:
                log(f"    ❌ DỪNG: Ma trận Jacobi suy biến tại bước k={i}.")
                break
            elif cond_J > COND_THRESHOLD:
                log(f"    ⚠️ CẢNH BÁO: Ma trận gần suy biến.")
            else:
                log(f"    ✅ Ma trận Jacobi khả nghịch.")

            # Tính J⁻¹, dX, cập nhật
            try:
                J_inv = np.linalg.inv(Jx)
                delta_X = -np.dot(J_inv, Fx)
                x_new = x + delta_X
                norm_delta = np.linalg.norm(delta_X, ord=np.inf)
                norm_J_inv = np.linalg.norm(J_inv, ord=np.inf)

                log("")
                log(f"  Bước 3: Tính J⁻¹(X⁽{i}⁾) và dX⁽{i}⁾")
                log(f"    J⁻¹(X⁽{i}⁾) =")
                for row in J_inv:
                    log(f"      [{trunc8(row[0])}, {trunc8(row[1])}, {trunc8(row[2])}]")
                log(f"    dX⁽{i}⁾ = [{trunc8(delta_X[0])}, {trunc8(delta_X[1])}, {trunc8(delta_X[2])}]ᵀ")
                log(f"    ‖dX⁽{i}⁾‖∞ = {norm_delta:.6e}")

                # ====== ĐÁNH GIÁ HẬU NGHIỆM - CÔNG THỨC + THAY SỐ ======
                log("")
                log("="*80)
                log(f"  📐 ĐÁNH GIÁ HẬU NGHIỆM TẠI BƯỚC k = {i}:")
                log("="*80)
                log("")
                log("  Cách 1: Dùng ‖dX⁽ᵏ⁾‖∞ (ước lượng đơn giản)")
                log("  ──────────────────────────────────────────────────────────────────")
                log("  Công thức:")
                log("        ‖X⁽ᵏ⁾ - X*‖∞ ≤ ‖dX⁽ᵏ⁾‖∞ = max(|dx₁⁽ᵏ⁾|, |dx₂⁽ᵏ⁾|, |dx₃⁽ᵏ⁾|)")
                log("")
                log("  Thay số:")
                log(f"        dX⁽{i}⁾ = [{trunc8(delta_X[0])}, {trunc8(delta_X[1])}, {trunc8(delta_X[2])}]ᵀ")
                log(f"        |dx₁| = |{trunc8(delta_X[0])}| = {abs(delta_X[0]):.8e}")
                log(f"        |dx₂| = |{trunc8(delta_X[1])}| = {abs(delta_X[1]):.8e}")
                log(f"        |dx₃| = |{trunc8(delta_X[2])}| = {abs(delta_X[2]):.8e}")
                log(f"        ‖dX⁽{i}⁾‖∞ = max({abs(delta_X[0]):.8e}, {abs(delta_X[1]):.8e}, {abs(delta_X[2]):.8e})")
                log(f"                   = {norm_delta:.8e}")
                log("")
                log(f"  ✅ KẾT LUẬN: ‖X⁽{i}⁾ - X*‖∞ ≤ {norm_delta:.8e}")
                log("")
                log("  Cách 2: Dùng ‖J⁻¹‖∞·‖F‖∞ (ước lượng chặt chẽ hơn)")
                log("  ──────────────────────────────────────────────────────────────────")
                log("  Công thức:")
                log("        ‖X⁽ᵏ⁾ - X*‖∞ ≤ ‖J⁻¹(X⁽ᵏ⁾)‖∞ · ‖F(X⁽ᵏ⁾)‖∞")
                log("")
                log("  Thay số:")
                log(f"        ‖J⁻¹(X⁽{i}⁾)‖∞ = {norm_J_inv:.8e}")
                log(f"        ‖F(X⁽{i}⁾)‖∞   = {norm_F:.8e}")
                log(f"        ‖J⁻¹‖∞ · ‖F‖∞ = {norm_J_inv:.8e} × {norm_F:.8e}")
                log(f"                        = {norm_J_inv * norm_F:.8e}")
                log("")
                log(f"  ✅ KẾT LUẬN: ‖X⁽{i}⁾ - X*‖∞ ≤ {norm_J_inv * norm_F:.8e}")
                log("")
                log("  Biện luận:")
                log("  • Cách 1 đơn giản, dễ tính, nhưng có thể đánh giá thiếu chặt.")
                log("  • Cách 2 chặt chẽ hơn nhưng cần tính thêm ‖J⁻¹‖∞.")
                log(f"  • Nếu sai số < ε thì X⁽{i}⁾ đã đủ chính xác, có thể dừng lặp.")
                log("="*80)

                log("")
                log(f"  Bước 4: Cập nhật nghiệm")
                log(f"    X⁽{i+1}⁾ = [{trunc8(x_new[0])}, {trunc8(x_new[1])}, {trunc8(x_new[2])}]ᵀ")

                history.append((i, x[0], x[1], x[2], norm_F, cond_J, norm_delta))
                x = x_new

            except np.linalg.LinAlgError:
                log(f"    ❌ LỖI: Không tính được nghịch đảo ma trận tại bước {i}.")
                break

        # Tính F cuối
        Fx_final = evaluate_F(x[0], x[1], x[2])
        history.append((n, x[0], x[1], x[2], np.linalg.norm(Fx_final, ord=np.inf), "-", "-"))

        # Bảng tổng hợp
        log("")
        log("="*110)
        log(" BẢNG TỔNG HỢP TIẾN TRÌNH LẶP NEWTON THƯỜNG 3 ẨN")
        log("="*110)
        log(f"{'k':<4} | {'x₁⁽ᵏ⁾':<14} | {'x₂⁽ᵏ⁾':<14} | {'x₃⁽ᵏ⁾':<14} | {'‖F‖∞':<12} | {'cond(J)':<10} | {'‖dX‖∞ (Hậu nghiệm)':<20}")
        log("-"*110)
        for row in history:
            k_v, x1_v, x2_v, x3_v, f_v, c_v, d_v = row
            f_s = f"{f_v:.4e}" if isinstance(f_v, float) else str(f_v)
            c_s = f"{c_v:.2f}" if isinstance(c_v, float) else str(c_v)
            d_s = f"{d_v:.4e}" if isinstance(d_v, float) else str(d_v)
            log(f"{k_v:<4} | {trunc8(x1_v):<14} | {trunc8(x2_v):<14} | {trunc8(x3_v):<14} | {f_s:<12} | {c_s:<10} | {d_s:<20}")
        log("="*110)

        log("")
        log(f"✅ NGHIỆM XẤP XỈ CUỐI CÙNG: X* = ({trunc8(x[0])}, {trunc8(x[1])}, {trunc8(x[2])})")
        log(f"📁 Đã lưu kết quả vào: {filename}")

if __name__ == "__main__":
    print(">>> PHƯƠNG PHÁP NEWTON THƯỜNG 3 ẨN - NHẬP PHƯƠNG TRÌNH TỪ BÀN PHÍM")
    print("Mẹo nhập: dùng * để nhân, ** để mũ (VD: x1**2 + x2 - 4)")
    print("Dùng x1, x2, x3 làm biến (không dùng x,y,z)")

    eq1 = input("Nhập f₁(x₁,x₂,x₃) = 0: ").strip()
    eq2 = input("Nhập f₂(x₁,x₂,x₃) = 0: ").strip()
    eq3 = input("Nhập f₃(x₁,x₂,x₃) = 0: ").strip()

    x0 = nhap_diem_khoi_tao(3)
    num_iter = nhap_so_nguyen("Nhập số lần lặp (n): ")

    newton_thuong_3an(eq1, eq2, eq3, x0, num_iter)
