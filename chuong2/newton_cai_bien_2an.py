import numpy as np
import math
import sympy as sp
import re

EPS = 1e-15

def trunc8(val):
    if math.isnan(val) or math.isinf(val):
        return str(val).ljust(15)
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

def newton_cai_bien_2an(f1_str, f2_str, x0, n, filename="kq_newton_cai_bien_2an.txt"):
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
        log(" PHƯƠNG PHÁP NEWTON CẢI BIÊN 2 ẨN - TỰ ĐỘNG TÍNH ĐẠO HÀM")
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
        log("II. CÔNG THỨC NEWTON CẢI BIÊN:")
        log("   X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾ - J⁻¹(X⁽⁰⁾) · F(X⁽ᵏ⁾)")
        log("   → J(X⁰) và J⁻¹(X⁰) chỉ tính MỘT LẦN DUY NHẤT tại điểm khởi tạo")
        log("   → Đây là đặc trưng PHÂN BIỆT với Newton thường!")
        log("-"*100)
        log("III. ĐÁNH GIÁ HẬU NGHIỆM:")
        log("   Vì J cố định, dùng công thức tiên nghiệm-like:")
        log("   ‖X⁽ᵏ⁾ - X*‖∞ ≤ [q/(1-q)] · ‖X⁽ᵏ⁾ - X⁽ᵏ⁻¹⁾‖∞")
        log("   Trong đó q ước lượng từ tỷ số ‖dX⁽ᵏ⁾‖/‖dX⁽ᵏ⁻¹⁾‖")
        log("="*100)
        log("")
        log(f"ĐIỂM KHỞI TẠO: X⁽⁰⁾ = [{x0[0]}, {x0[1]}]ᵀ")
        log(f"SỐ BƯỚC LẶP: n = {n}")
        log("")

        # TÍNH J(X⁰) MỘT LẦN DUY NHẤT
        log("="*100)
        log("■ TÍNH J(X⁽⁰⁾) VÀ J⁻¹(X⁽⁰⁾) - CHỈ THỰC HIỆN MỘT LẦN")
        log("-"*100)
        J0 = evaluate_J(x0[0], x0[1])
        det_J0 = np.linalg.det(J0)

        log(f"  J(X⁽⁰⁾) = [[{trunc8(J0[0,0])}, {trunc8(J0[0,1])}]")
        log(f"           [{trunc8(J0[1,0])}, {trunc8(J0[1,1])}]]")
        log(f"  det(J(X⁰)) = {det_J0:.6e}")

        if abs(det_J0) < EPS:
            log("\n❌ DỪNG: Ma trận Jacobian tại X⁰ suy biến (det = 0).")
            log("   Giải pháp: Biến đổi lại phương trình hoặc đổi điểm khởi tạo theo đề bài.")
            log("   KHÔNG ĐƯỢC tự ý đổi điểm khởi tạo - sẽ bị trừ điểm!")
            return

        try:
            J0_inv = np.linalg.inv(J0)
        except np.linalg.LinAlgError:
            log("\n❌ LỖI: Không tính được nghịch đảo ma trận J(X⁰).")
            return

        log(f"  J⁻¹(X⁰) = [[{trunc8(J0_inv[0,0])}, {trunc8(J0_inv[0,1])}]")
        log(f"            [{trunc8(J0_inv[1,0])}, {trunc8(J0_inv[1,1])}]]")
        log(f"  ✅ Đã tính xong J⁻¹(X⁰). Từ bây giờ sẽ DÙNG LẠI ma trận này cho mọi bước lặp.")
        log("="*100)
        log("")

        x_old = x.copy()
        dx_old = 0

        for i in range(n + 1):
            log("="*100)
            log(f"■ BƯỚC LẶP k = {i}:")
            log("-"*100)

            # Tính F(X⁽ᵏ⁾)
            Fx = evaluate_F(x[0], x[1])
            norm_F = np.linalg.norm(Fx, ord=2)

            log(f"  Bước 1: Tính F(X⁽{i}⁾)")
            log(f"    X⁽{i}⁾ = [{trunc8(x[0])}, {trunc8(x[1])}]ᵀ")
            log(f"    F(X⁽{i}⁾) = [{trunc8(Fx[0])}, {trunc8(Fx[1])}]ᵀ")
            log(f"    → ‖F(X⁽{i}⁾)‖₂ = {norm_F:.6e}")

            history.append((i, x[0], x[1], norm_F))

            if i == n:
                log(f"    → Đã đạt số bước lặp tối đa n = {n}.")
                break

            # Dùng J⁻¹(X⁰) đã tính sẵn
            log("")
            log(f"  Bước 2: Tính dX⁽{i}⁾ = -J⁻¹(X⁰) · F(X⁽{i}⁾)")
            log(f"    → DÙNG LẠI J⁻¹(X⁰) đã tính ở trên (KHÔNG tính lại J!)")
            tich_phan = np.dot(J0_inv, Fx)
            x_new = x - tich_phan
            dx = np.linalg.norm(x_new - x, ord=np.inf)

            log(f"    J⁻¹(X⁰) · F(X⁽{i}⁾) = [{trunc8(tich_phan[0])}, {trunc8(tich_phan[1])}]ᵀ")
            log(f"    dX⁽{i}⁾ = [{trunc8(-tich_phan[0])}, {trunc8(-tich_phan[1])}]ᵀ")
            log(f"    ‖dX⁽{i}⁾‖∞ = {dx:.6e}")

            # ====== ĐÁNH GIÁ HẬU NGHIỆM - CÔNG THỨC + THAY SỐ ======
            log("")
            log("="*70)
            log(f"  📐 ĐÁNH GIÁ HẬU NGHIỆM TẠI BƯỚC k = {i}:")
            log("="*70)
            log("")
            log("  Công thức (vì J cố định, dùng tiên nghiệm-like):")
            log("  ─────────────────────────────────────────────────────────────")
            log("        ‖X⁽ᵏ⁾ - X*‖∞ ≤ [q/(1-q)] · ‖X⁽ᵏ⁾ - X⁽ᵏ⁻¹⁾‖∞")
            log("")
            log("  Trong đó q ước lượng từ tỷ số:")
            log("        q ≈ ‖dX⁽ᵏ⁾‖∞ / ‖dX⁽ᵏ⁻¹⁾‖∞")
            log("")

            if i > 0 and dx_old > 0:
                q_est = dx / dx_old
                log("  Thay số:")
                log(f"        ‖dX⁽{i}⁾‖∞   = {dx:.8e}")
                log(f"        ‖dX⁽{i-1}⁾‖∞ = {dx_old:.8e}")
                log(f"        q = {dx:.8e} / {dx_old:.8e} = {q_est:.8f}")
                log("")

                if q_est < 1:
                    hau_nghiem = (q_est / (1 - q_est)) * dx
                    log("  Tính sai số hậu nghiệm:")
                    log(f"        q/(1-q) = {q_est:.8f} / (1 - {q_est:.8f})")
                    log(f"                = {q_est:.8f} / {1 - q_est:.8f}")
                    log(f"                = {q_est/(1-q_est):.8f}")
                    log("")
                    log(f"        Δₖ = [q/(1-q)] · ‖dX⁽{i}⁾‖∞")
                    log(f"           = {q_est/(1-q_est):.8f} × {dx:.8e}")
                    log(f"           = {hau_nghiem:.8e}")
                    log("")
                    log(f"  ✅ KẾT LUẬN: ‖X⁽{i}⁾ - X*‖∞ ≤ {hau_nghiem:.8e}")
                    log("")
                    log("  Biện luận:")
                    if q_est < 0.5:
                        log(f"  • q = {q_est:.4f} < 0.5 → Hội tụ rất nhanh (ánh xạ co mạnh).")
                    elif q_est < 1:
                        log(f"  • q = {q_est:.4f} < 1 → Thỏa mãn ánh xạ co, dãy lặp hội tụ.")
                    log(f"  • Sai số giảm theo cấp số nhân với công bội q ≈ {q_est:.4f}.")
                    log(f"  • Nếu Δₖ < ε thì X⁽{i}⁾ đã đủ chính xác.")
                else:
                    log(f"  ⚠️ q = {q_est:.4f} ≥ 1 → Không thỏa mãn ánh xạ co chặt.")
                    log(f"  → Dùng ước lượng đơn giản: ‖X⁽{i}⁾ - X*‖∞ ≈ ‖dX⁽{i}⁾‖∞ = {dx:.8e}")
            else:
                log("  Bước đầu tiên (k=0), chưa có dX⁽ᵏ⁻¹⁾ để ước lượng q.")
                log(f"  → Dùng ước lượng thô: ‖X⁽{i}⁾ - X*‖∞ ≈ ‖dX⁽{i}⁾‖∞ = {dx:.8e}")
                log("")
                log("  Biện luận: Cần ít nhất 2 bước lặp để ước lượng q chính xác.")

            log("="*70)

            log("")
            log(f"  Bước 3: Cập nhật nghiệm")
            log(f"    X⁽{i+1}⁾ = X⁽{i}⁾ + dX⁽{i}⁾ = [{trunc8(x_new[0])}, {trunc8(x_new[1])}]ᵀ")

            dx_old = dx
            x_old = x.copy()
            x = x_new

        # Bảng tổng hợp
        log("")
        log("="*100)
        log(" BẢNG TỔNG HỢP TIẾN TRÌNH LẶP NEWTON CẢI BIÊN 2 ẨN")
        log("="*100)
        log(f"{'k':<4} | {'x₁⁽ᵏ⁾':<16} | {'x₂⁽ᵏ⁾':<16} | {'‖F‖₂':<14} | {'Nhận xét':<30}")
        log("-"*100)
        for row in history:
            k_v, x1_v, x2_v, f_v = row
            if f_v < 1e-10:
                nx = "Đã hội tụ rất tốt"
            elif f_v < 1e-6:
                nx = "Gần nghiệm chính xác"
            elif f_v < 1e-3:
                nx = "Chấp nhận được"
            else:
                nx = "Cần lặp thêm"
            log(f"{k_v:<4} | {trunc8(x1_v):<16} | {trunc8(x2_v):<16} | {f_v:<14.6e} | {nx:<30}")
        log("="*100)

        log("")
        log(f"✅ NGHIỆM XẤP XỈ CUỐI CÙNG: X* = ({trunc8(x[0])}, {trunc8(x[1])})")
        log(f"📁 Đã lưu kết quả vào: {filename}")

if __name__ == "__main__":
    print(">>> PHƯƠNG PHÁP NEWTON CẢI BIÊN 2 ẨN - NHẬP PHƯƠNG TRÌNH TỪ BÀN PHÍM")
    print("Mẹo nhập: dùng * để nhân, ** để mũ (VD: x1**2 + x2 - 4)")
    print("Nhập điểm khởi tạo có thể nhập: 0, 1 hoặc (0, 1)")

    eq1 = input("Nhập f₁(x₁,x₂) = 0: ").strip()
    eq2 = input("Nhập f₂(x₁,x₂) = 0: ").strip()

    x0 = nhap_diem_khoi_tao(2)
    num_iter = nhap_so_nguyen("Nhập số lần lặp (n): ")

    newton_cai_bien_2an(eq1, eq2, x0, num_iter)
