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
    raw = input(f"{prompt} (VD: 0, 1 hoặc nhập từng số): ").strip()
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

def eval_expr(expr, symbols, values):
    """Đánh giá sympy expression an toàn nhất - subs + evalf(15) + float()"""
    subs_dict = dict(zip(symbols, values))
    try:
        result = expr.subs(subs_dict)
        if result.has(sp.Symbol):
            result = result.evalf(15)
        return float(result)
    except Exception as e:
        print(f"LỖI đánh giá biểu thức: {e}")
        print(f"Expression: {expr}")
        print(f"Values: {values}")
        print(f"Kết quả subs: {expr.subs(subs_dict)}")
        raise

def newton_thuong_2an(f1_str, f2_str, x0, n, filename="kq_newton_thuong_2an.txt"):
    x1, x2 = sp.symbols('x1 x2')
    syms = (x1, x2)

    f1_expr = sp.sympify(f1_str)
    f2_expr = sp.sympify(f2_str)

    # Tự động tính đạo hàm
    df1_dx1 = sp.diff(f1_expr, x1)
    df1_dx2 = sp.diff(f1_expr, x2)
    df2_dx1 = sp.diff(f2_expr, x1)
    df2_dx2 = sp.diff(f2_expr, x2)

    def evaluate_F(x_val, y_val):
        vals = (x_val, y_val)
        return np.array([
            eval_expr(f1_expr, syms, vals),
            eval_expr(f2_expr, syms, vals)
        ], dtype=float)

    def evaluate_J(x_val, y_val):
        vals = (x_val, y_val)
        return np.array([
            [eval_expr(df1_dx1, syms, vals), eval_expr(df1_dx2, syms, vals)],
            [eval_expr(df2_dx1, syms, vals), eval_expr(df2_dx2, syms, vals)]
        ], dtype=float)

    x = np.array(x0, dtype=float)
    history = []

    with open(filename, "w", encoding="utf-8") as f:
        def log(message):
            print(message)
            f.write(message + "\n")

        log("="*100)
        log(" PHƯƠNG PHÁP NEWTON-RAPHSON THƯỜNG 2 ẨN - TỰ ĐỘNG TÍNH ĐẠO HÀM")
        log("="*100)
        log("I. HỆ PHƯƠNG TRÌNH GỐC VÀ MA TRẬN JACOBI TỰ ĐỘNG")
        log("-"*100)
        log(f"   f₁(x₁,x₂) = {f1_expr} = 0")
        log(f"   f₂(x₁,x₂) = {f2_expr} = 0")
        log("")
        log("   Các đạo hàm riêng tự động tính:")
        log(f"   ∂f₁/∂x₁ = {str(df1_dx1)}")
        log(f"   ∂f₁/∂x₂ = {str(df1_dx2)}")
        log(f"   ∂f₂/∂x₁ = {str(df2_dx1)}")
        log(f"   ∂f₂/∂x₂ = {str(df2_dx2)}")
        log("-"*100)
        log("II. CÔNG THỨC LẶP NEWTON THƯỜNG:")
        log("   X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾ - J⁻¹(X⁽ᵏ⁾) · F(X⁽ᵏ⁾)")
        log("   → J(X) được tính LẠI Ở MỖI BƯỚC LẶP (đây là đặc trưng Newton thường)")
        log("-"*100)
        log("III. ĐÁNH GIÁ HẬU NGHIỆM:")
        log("   Công thức: ‖X⁽ᵏ⁾ - X*‖∞ ≤ ‖dX⁽ᵏ⁾‖∞ = max(|dx₁|, |dx₂|)")
        log("   Hoặc chặt chẽ hơn: ‖X⁽ᵏ⁾ - X*‖∞ ≤ ‖J⁻¹(X⁽ᵏ⁾)‖∞ · ‖F(X⁽ᵏ⁾)‖∞")
        log("="*100)
        log("")
        log(f"ĐIỂM KHỞI TẠO: X⁽⁰⁾ = [{x0[0]}, {x0[1]}]ᵀ")
        log(f"SỐ BƯỚC LẶP: n = {n}")
        log("")

        for i in range(n):
            log("="*100)
            log(f"■ BƯỚC LẶP k = {i}:")
            log("-"*100)

            # Tính F(X⁽ᵏ⁾)
            Fx = evaluate_F(x[0], x[1])
            log(f"  Bước 1: Tính F(X⁽{i}⁾)")
            log(f"    X⁽{i}⁾ = [{trunc8(x[0])}, {trunc8(x[1])}]ᵀ")
            log(f"    F(X⁽{i}⁾) = [{trunc8(Fx[0])}, {trunc8(Fx[1])}]ᵀ")
            log(f"    → ‖F(X⁽{i}⁾)‖₂ = {np.linalg.norm(Fx, ord=2):.6e}")

            # Tính J(X⁽ᵏ⁾)
            Jx = evaluate_J(x[0], x[1])
            det_J = np.linalg.det(Jx)
            log("")
            log(f"  Bước 2: Tính ma trận Jacobi J(X⁽{i}⁾)")
            log(f"    J(X⁽{i}⁾) = [[{trunc8(Jx[0,0])}, {trunc8(Jx[0,1])}]")
            log(f"              [{trunc8(Jx[1,0])}, {trunc8(Jx[1,1])}]]")
            log(f"    det(J) = {det_J:.6e}")

            # Kiểm tra điều kiện
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

            # Tính J⁻¹ và dX
            try:
                J_inv = np.linalg.inv(Jx)
                delta_X = -np.dot(J_inv, Fx)
                x_new = x + delta_X
                norm_delta = np.linalg.norm(delta_X, ord=np.inf)
                norm_F = np.linalg.norm(Fx, ord=np.inf)
                norm_J_inv = np.linalg.norm(J_inv, ord=np.inf)
                hau_nghiem_JF = norm_J_inv * norm_F

                log("")
                log(f"  Bước 3: Tính J⁻¹(X⁽{i}⁾) và dX⁽{i}⁾")
                log(f"    J⁻¹(X⁽{i}⁾) = [[{trunc8(J_inv[0,0])}, {trunc8(J_inv[0,1])}]")
                log(f"                 [{trunc8(J_inv[1,0])}, {trunc8(J_inv[1,1])}]]")
                log(f"    dX⁽{i}⁾ = -J⁻¹·F = [{trunc8(delta_X[0])}, {trunc8(delta_X[1])}]ᵀ")
                log(f"    ‖dX⁽{i}⁾‖∞ = {norm_delta:.6e}")

                # ====== ĐÁNH GIÁ HẬU NGHIỆM - CÔNG THỨC + THAY SỐ ======
                log("")
                log("="*70)
                log(f"  📐 ĐÁNH GIÁ HẬU NGHIỆM TẠI BƯỚC k = {i}:")
                log("="*70)
                log("")
                log("  Cách 1: Dùng ‖dX⁽ᵏ⁾‖∞ (ước lượng đơn giản)")
                log("  ─────────────────────────────────────────────────────────────")
                log("  Công thức:")
                log("        ‖X⁽ᵏ⁾ - X*‖∞ ≤ ‖dX⁽ᵏ⁾‖∞ = max(|dx₁⁽ᵏ⁾|, |dx₂⁽ᵏ⁾|)")
                log("")
                log("  Thay số:")
                log(f"        dX⁽{i}⁾ = [{trunc8(delta_X[0])}, {trunc8(delta_X[1])}]ᵀ")
                log(f"        |dx₁| = |{trunc8(delta_X[0])}| = {abs(delta_X[0]):.8e}")
                log(f"        |dx₂| = |{trunc8(delta_X[1])}| = {abs(delta_X[1]):.8e}")
                log(f"        ‖dX⁽{i}⁾‖∞ = max({abs(delta_X[0]):.8e}, {abs(delta_X[1]):.8e})")
                log(f"                   = {norm_delta:.8e}")
                log("")
                log(f"  ✅ KẾT LUẬN: ‖X⁽{i}⁾ - X*‖∞ ≤ {norm_delta:.8e}")
                log("")
                log("  Cách 2: Dùng ‖J⁻¹‖∞·‖F‖∞ (ước lượng chặt chẽ hơn)")
                log("  ─────────────────────────────────────────────────────────────")
                log("  Công thức:")
                log("        ‖X⁽ᵏ⁾ - X*‖∞ ≤ ‖J⁻¹(X⁽ᵏ⁾)‖∞ · ‖F(X⁽ᵏ⁾)‖∞")
                log("")
                log("  Thay số:")
                log(f"        ‖J⁻¹(X⁽{i}⁾)‖∞ = {norm_J_inv:.8e}")
                log(f"        ‖F(X⁽{i}⁾)‖∞   = {norm_F:.8e}")
                log(f"        ‖J⁻¹‖∞ · ‖F‖∞ = {norm_J_inv:.8e} × {norm_F:.8e}")
                log(f"                        = {hau_nghiem_JF:.8e}")
                log("")
                log(f"  ✅ KẾT LUẬN: ‖X⁽{i}⁾ - X*‖∞ ≤ {hau_nghiem_JF:.8e}")
                log("")
                log("  Biện luận:")
                log("  • Cách 1 đơn giản, dễ tính, nhưng có thể đánh giá thiếu chặt.")
                log("  • Cách 2 chặt chẽ hơn nhưng cần tính thêm ‖J⁻¹‖∞.")
                log(f"  • Nếu sai số < ε (ví dụ ε = 10⁻⁶) thì X⁽{i}⁾ đủ chính xác.")
                log("="*70)

                log("")
                log(f"  Bước 4: Cập nhật nghiệm")
                log(f"    X⁽{i+1}⁾ = X⁽{i}⁾ + dX⁽{i}⁾ = [{trunc8(x_new[0])}, {trunc8(x_new[1])}]ᵀ")

                history.append((i, x[0], x[1], norm_F, cond_J, norm_delta, norm_delta, hau_nghiem_JF))
                x = x_new

            except np.linalg.LinAlgError:
                log(f"    ❌ LỖI: Không tính được nghịch đảo ma trận tại bước {i}.")
                break

        # Bảng tổng hợp
        log("")
        log("="*100)
        log(" BẢNG TỔNG HỢP TIẾN TRÌNH LẶP NEWTON THƯỜNG 2 ẨN")
        log("="*100)
        log(f"{'k':<4} | {'x₁⁽ᵏ⁾':<16} | {'x₂⁽ᵏ⁾':<16} | {'‖F‖∞':<12} | {'cond(J)':<10} | {'‖dX‖∞ (Hậu nghiệm)':<20}")
        log("-"*100)
        for row in history:
            k_v, x1_v, x2_v, f_v, c_v, d_v, h1, h2 = row
            c_s = f"{c_v:.2f}" if isinstance(c_v, float) else str(c_v)
            log(f"{k_v:<4} | {trunc8(x1_v):<16} | {trunc8(x2_v):<16} | {f_v:<12.4e} | {c_s:<10} | {h1:<20.4e}")
        log("="*100)

        log("")
        log(f"✅ NGHIỆM XẤP XỈ CUỐI CÙNG: X* = ({trunc8(x[0])}, {trunc8(x[1])})")
        log(f"📁 Đã lưu kết quả vào: {filename}")

if __name__ == "__main__":
    print(">>> PHƯƠNG PHÁP NEWTON THƯỜNG 2 ẨN - NHẬP PHƯƠNG TRÌNH TỪ BÀN PHÍM")
    print("Mẹo nhập: dùng * để nhân, ** để mũ (VD: x1**2 + x2 - 4)")
    print("Nhập điểm khởi tạo có thể nhập: 0, 1 hoặc (0, 1)")

    eq1 = input("Nhập f₁(x₁,x₂) = 0: ").strip()
    eq2 = input("Nhập f₂(x₁,x₂) = 0: ").strip()

    x0 = nhap_diem_khoi_tao(2)
    num_iter = nhap_so_nguyen("Nhập số lần lặp (n): ")

    newton_thuong_2an(eq1, eq2, x0, num_iter)
