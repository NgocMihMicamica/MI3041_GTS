import math
import sympy as sp
import numpy as np
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

def chuan_vo_cung_3(x1, x2, x3):
    return max(abs(x1), abs(x2), abs(x3))

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

def lap_don_3an(phi1_str, phi2_str, phi3_str, x0, ep, q=None, filename="kq_lap_don_3an.txt"):
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    syms = (x1, x2, x3)

    phi1_expr = sp.sympify(phi1_str)
    phi2_expr = sp.sympify(phi2_str)
    phi3_expr = sp.sympify(phi3_str)

    # Tự động tính đạo hàm
    dphi1_dx1 = sp.diff(phi1_expr, x1)
    dphi1_dx2 = sp.diff(phi1_expr, x2)
    dphi1_dx3 = sp.diff(phi1_expr, x3)
    dphi2_dx1 = sp.diff(phi2_expr, x1)
    dphi2_dx2 = sp.diff(phi2_expr, x2)
    dphi2_dx3 = sp.diff(phi2_expr, x3)
    dphi3_dx1 = sp.diff(phi3_expr, x1)
    dphi3_dx2 = sp.diff(phi3_expr, x2)
    dphi3_dx3 = sp.diff(phi3_expr, x3)

    def tinh_jacobi_uoc_luong_q(x_val, y_val, z_val):
        vals = (x_val, y_val, z_val)
        Jx = np.array([
            [eval_expr(dphi1_dx1, syms, vals), eval_expr(dphi1_dx2, syms, vals), eval_expr(dphi1_dx3, syms, vals)],
            [eval_expr(dphi2_dx1, syms, vals), eval_expr(dphi2_dx2, syms, vals), eval_expr(dphi2_dx3, syms, vals)],
            [eval_expr(dphi3_dx1, syms, vals), eval_expr(dphi3_dx2, syms, vals), eval_expr(dphi3_dx3, syms, vals)]
        ], dtype=float)
        tong_hang = [sum(abs(Jx[i,j]) for j in range(3)) for i in range(3)]
        return max(tong_hang)

    with open(filename, "w", encoding="utf-8") as f:
        def log(message):
            print(message)
            f.write(message + "\n")

        log("="*100)
        log(" PHƯƠNG PHÁP LẶP ĐƠN 3 ẨN - TỰ ĐỘNG TÍNH ĐẠO HÀM")
        log("="*100)
        log("I. HỆ PHƯƠNG TRÌNH DẠNG LẶP:")
        log("-"*100)
        log(f"   x₁ = {phi1_expr}")
        log(f"   x₂ = {phi2_expr}")
        log(f"   x₃ = {phi3_expr}")
        log("")
        log("   Ma trận Jacobi của hệ lặp (ma trận đạo hàm của φ):")
        log(f"   ∂φ₁/∂x₁ = {str(dphi1_dx1)}")
        log(f"   ∂φ₁/∂x₂ = {str(dphi1_dx2)}")
        log(f"   ∂φ₁/∂x₃ = {str(dphi1_dx3)}")
        log(f"   ∂φ₂/∂x₁ = {str(dphi2_dx1)}")
        log(f"   ∂φ₂/∂x₂ = {str(dphi2_dx2)}")
        log(f"   ∂φ₂/∂x₃ = {str(dphi2_dx3)}")
        log(f"   ∂φ₃/∂x₁ = {str(dphi3_dx1)}")
        log(f"   ∂φ₃/∂x₂ = {str(dphi3_dx2)}")
        log(f"   ∂φ₃/∂x₃ = {str(dphi3_dx3)}")
        log("-"*100)

        # Tính q
        if q is None:
            q_calc = tinh_jacobi_uoc_luong_q(x0[0], x0[1], x0[2])
            log(f"\n[TỰ ĐỘNG TÍNH q] Tại X⁰: ‖Jφ(X⁰)‖∞ = {q_calc:.6f}")
            if q_calc < 1:
                q = q_calc
                log(f"   → Thỏa mãn ánh xạ co. Sử dụng q = {q:.6f}")
                log(f"   Biện luận: Vì q < 1, theo định lý ánh xạ co của Banach, dãy lặp sẽ hội tụ.")
            else:
                log(f"\n❌ NGHIÊM TÚC: q = {q_calc:.6f} ≥ 1")
                log("   HỆ KHÔNG PHẢI ÁNH XẠ CO - LẶP ĐƠN SẼ PHÂN KỲ!")
                log("   BẠN PHẢI BIẾN ĐỔI LẠI HỆ PHƯƠNG TRÌNH TRƯỚC KHI CHẠY.")
                log("   Biện luận: q ≥ 1 nghĩa là ‖φ(x) - φ(y)‖ ≥ ‖x - y‖, dãy lặp không co lại.")
                return
        else:
            if not (0 < q < 1):
                log(f"\n❌ NGHIÊM TÚC: q = {q} không nằm trong (0,1)")
                log("   HỆ KHÔNG THỎA MÃN ĐIỀU KIỆN ÁNH XẠ CO.")
                return
            log(f"\n[NHẬP TAY] q = {q:.6f}")
            log(f"   Biện luận: q do người dùng cung cấp, giả định đã kiểm chứng tính co.")

        log("")
        log("II. CÔNG THỨC SAI SỐ HẬU NGHIỆM:")
        log("   Δₖ = [q/(1-q)] · ‖X⁽ᵏ⁾ - X⁽ᵏ⁻¹⁾‖∞")
        log("   Điều kiện dừng: Δₖ ≤ εpsilon = {ep}")
        log("   Biện luận: Sai số hậu nghiệm cho biết sai số tối đa của nghiệm xấp xỉ so với nghiệm đúng.")
        log("             Nếu Δₖ ≤ ε thì X⁽ᵏ⁾ đủ chính xác, có thể dừng lặp.")
        log("="*100)
        log("")
        log(f"ĐIỂM KHỞI TẠO: X⁽⁰⁾ = [{x0[0]}, {x0[1]}, {x0[2]}]")
        log("")

        log("-"*100)
        log(f"{'k':<4} | {'x₁⁽ᵏ⁾':<16} | {'x₂⁽ᵏ⁾':<16} | {'x₃⁽ᵏ⁾':<16} | {'‖dX‖∞':<14} | {'Δₖ':<14} | {'Nhận xét':<15}")
        log("-"*100)

        xv, yv, zv = x0[0], x0[1], x0[2]
        log(f"{'0':<4} | {trunc8(xv):<16} | {trunc8(yv):<16} | {trunc8(zv):<16} | {'-':<14} | {'-':<14} | {'Khởi tạo':<15}")

        k = 1
        history = []

        while True:
            if k > MAX_ITER:
                log("-"*100)
                log(f"🛑 DỪNG: Đạt giới hạn {MAX_ITER} bước lặp.")
                log("   Biện luận: Số bước lặp vượt quá giới hạn, có thể hội tụ chậm hoặc phân kỳ.")
                break

            tmp1, tmp2, tmp3 = xv, yv, zv
            vals = (tmp1, tmp2, tmp3)
            xv = eval_expr(phi1_expr, syms, vals)
            yv = eval_expr(phi2_expr, syms, vals)
            zv = eval_expr(phi3_expr, syms, vals)

            ly_do = chuan_vo_cung_3(xv-tmp1, yv-tmp2, zv-tmp3)
            saiso = (q / (1 - q)) * ly_do

            if saiso <= ep:
                nx = "✅ Đạt ε"
            elif saiso <= ep * 10:
                nx = "Gần đạt"
            else:
                nx = "Lặp tiếp"

            log(f"{k:<4} | {trunc8(xv):<16} | {trunc8(yv):<16} | {trunc8(zv):<16} | {ly_do:<14.6e} | {saiso:<14.6e} | {nx:<15}")
            history.append((k, xv, yv, zv, ly_do, saiso))

            if saiso <= ep:
                log("-"*100)
                log(f"✅ HỘI TỤ tại bước k={k}: Δₖ = {saiso:.6e} ≤ εpsilon = {ep}")
                log(f"   Biện luận: Sai số hậu nghiệm đã nhỏ hơn ngưỡng ε.")
                log(f"   Theo định lý ánh xạ co: ‖X⁽ᵏ⁾ - X*‖∞ ≤ Δₖ = {saiso:.6e}")
                log(f"   NGHỊỆM: X* = ({trunc8(xv)}, {trunc8(yv)}, {trunc8(zv)})")
                log(f"   Độ chính xác: Các thành phần xấp xỉ đúng đến ít nhất {max(0, int(-math.log10(ep)))} chữ số thập phân.")
                break

            k += 1

        # Bảng tổng hợp
        log("")
        log("="*100)
        log(" BẢNG TỔNG HỢP TIẾN TRÌNH LẶP ĐƠN 3 ẨN")
        log("="*100)
        log(f"{'k':<4} | {'x₁⁽ᵏ⁾':<16} | {'x₂⁽ᵏ⁾':<16} | {'x₃⁽ᵏ⁾':<16} | {'‖dX‖∞':<14} | {'Δₖ (hậu nghiệm)':<18} | {'Đánh giá':<20}")
        log("-"*100)
        for row in history:
            k_v, x1_v, x2_v, x3_v, d_v, s_v = row
            if s_v <= ep:
                dg = "✅ Đủ chính xác"
            elif s_v <= ep * 100:
                dg = "Chấp nhận được"
            else:
                dg = "Cần lặp thêm"
            log(f"{k_v:<4} | {trunc8(x1_v):<16} | {trunc8(x2_v):<16} | {trunc8(x3_v):<16} | {d_v:<14.6e} | {s_v:<18.6e} | {dg:<20}")
        log("="*100)

        # ====== PHẦN ĐÁNH GIÁ HẬU NGHIỆM CHI TIẾT ======
        log("")
        log("="*100)
        log(" 📐 PHẦN ĐÁNH GIÁ HẬU NGHIỆM CHI TIẾT (CÔNG THỨC + THAY SỐ)")
        log("="*100)
        log("")
        log("  Công thức sai số hậu nghiệm của phương pháp lặp đơn:")
        log("  ────────────────────────────────────────────────────────────────────────")
        log("        Δₖ = [q / (1 - q)] · ‖X⁽ᵏ⁾ - X⁽ᵏ⁻¹⁾‖∞")
        log("")
        log("  Trong đó:")
        log("    • q = chỉ số co (đã tính ở trên)")
        log("    • ‖X⁽ᵏ⁾ - X⁽ᵏ⁻¹⁾‖∞ = max(|x₁⁽ᵏ⁾-x₁⁽ᵏ⁻¹⁾|, |x₂⁽ᵏ⁾-x₂⁽ᵏ⁻¹⁾|, |x₃⁽ᵏ⁾-x₃⁽ᵏ⁻¹⁾|)")
        log("")
        log("  Thay số từng bước:")
        log("")

        for idx, row in enumerate(history):
            k_v, x1_v, x2_v, x3_v, d_v, s_v = row
            log(f"  ► Bước k = {k_v}:")
            log(f"      q = {q:.6f}")
            log(f"      1 - q = {1 - q:.6f}")
            log(f"      q/(1-q) = {q:.6f} / {1 - q:.6f} = {q/(1-q):.6f}")
            log(f"      ‖X⁽{k_v}⁾ - X⁽{k_v-1}⁾‖∞ = {d_v:.8e}")
            log(f"      Δₖ = {q/(1-q):.6f} × {d_v:.8e} = {s_v:.8e}")
            if s_v <= ep:
                log(f"      ✅ Δₖ = {s_v:.8e} ≤ ε = {ep} → ĐỦ CHÍNH XÁC")
            else:
                log(f"      ⚠️ Δₖ = {s_v:.8e} > ε = {ep} → CẦN LẶP TIẾP")
            log("")

        log("  Biện luận tổng quát:")
        log(f"  • Với q = {q:.4f} < 1, hệ thỏa mãn điều kiện ánh xạ co.")
        log(f"  • Sai số giảm theo cấp số nhân với tốc độ q = {q:.4f}.")
        log(f"  • Khi k → ∞, Δₖ → 0, nghiệm xấp xỉ X⁽ᵏ⁾ → X* (nghiệm đúng).")
        log("="*100)

        log("")
        log(f"📁 Đã lưu kết quả vào: {filename}")

if __name__ == "__main__":
    print(">>> LẶP ĐƠN 3 ẨN - NHẬP PHƯƠNG TRÌNH DẠNG LẶP")
    print("Mẹo nhập: dùng * để nhân, ** để mũ")
    print("Nhập điểm khởi tạo có thể nhập: 0, 0, 0 hoặc (0, 0, 0)")

    eq1 = input("Nhập φ₁(x₁,x₂,x₃): ").strip()
    eq2 = input("Nhập φ₂(x₁,x₂,x₃): ").strip()
    eq3 = input("Nhập φ₃(x₁,x₂,x₃): ").strip()

    x0 = nhap_diem_khoi_tao(3)
    ep = float(input("Nhập epsilon (VD: 1e-6): ").strip())

    q_input = input("Nhập q (để trống để tự động tính): ").strip()
    q = float(q_input) if q_input else None

    lap_don_3an(eq1, eq2, eq3, x0, ep, q)
