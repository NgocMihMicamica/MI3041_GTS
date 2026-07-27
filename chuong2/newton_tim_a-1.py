import numpy as np


def read_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("Vui lòng nhập số nguyên hợp lệ!")


def read_matrix(n, name):
    print(f"--- Nhập ma trận {name} ({n}×{n}) ---")
    matrix = []
    for i in range(n):
        while True:
            try:
                raw = input(f"Nhập hàng {i+1} (các phần tử cách nhau bởi dấu cách): ")
                row = list(map(float, raw.replace('−', '-').replace('–', '-').split()))
                if len(row) != n:
                    print(f"Hàng phải có đúng {n} phần tử!")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Dữ liệu không hợp lệ, vui lòng nhập lại!")
    return np.array(matrix, dtype=float)


def bieu_dien_ma_tran(matrix, ten="", width=12):
    if ten:
        print(f"{ten} =")
    for row in matrix:
        print("  [" + ", ".join([f"{x:>{width}.8f}" for x in row]) + " ]")
    print()


def format_scientific(val):
    """Định dạng số khoa học đẹp"""
    if abs(val) < 1e-10:
        return "0.00000000"
    return f"{val:.8e}"


def newton_inverse_iterative(A, max_iter=100, tol=1e-12):
    """
    Phương pháp lặp Newton tìm ma trận nghịch đảo.
    Công thức: G_{k+1} = G_k · (2I - A · G_k)
    Điều kiện hội tụ: ρ(I - A·G_0) < 1  (spectral radius)
    """
    n = A.shape[0]
    I = np.eye(n)

    print("=" * 100)
    print(" PHƯƠNG PHÁP LẶP NEWTON TÌM MA TRẬN NGHỊCH ĐẢO")
    print("=" * 100)
    print("I. CƠ SỞ LÝ THUYẾT")
    print("-" * 100)
    print("   Cho ma trận A khả nghịch, xét phương trình ma trận:")
    print("        A · X = I")
    print("   Đặt F(X) = A·X - I = 0. Áp dụng phương pháp Newton cho ma trận:")
    print("        X_{k+1} = X_k - [F'(X_k)]^{-1} · F(X_k)")
    print("   Với F'(X) = A (đạo hàm Fréchet), ta có:")
    print("        X_{k+1} = X_k - A^{-1}·(A·X_k - I)")
    print("                = X_k - X_k + A^{-1}")
    print("   Điều này không khả thi vì cần A^{-1}. Thay vào đó, xét dạng lặp:")
    print("        G_{k+1} = G_k · (2I - A · G_k)")
    print("   Đặt R_k = I - A·G_k. Khi đó:")
    print("        R_{k+1} = I - A·G_{k+1}")
    print("                = I - A·G_k·(2I - A·G_k)")
    print("                = I - 2A·G_k + (A·G_k)^2")
    print("                = (I - A·G_k)^2 = R_k^2")
    print("   Suy ra: R_k = (R_0)^{2^k}. Nếu ρ(R_0) < 1 thì R_k → 0, do đó")
    print("        G_k → A^{-1}  (khi k → ∞)")
    print("-" * 100)
    print("II. CHỌN XẤP XỈ BAN ĐẦU G_0")
    print("-" * 100)
    print("   Theo Ben-Israel (1965), chọn:")
    print("        G_0 = α · A^T")
    print("   với α ∈ (0, 2/λ₁(A·A^T)), λ₁ là trị riêng lớn nhất của A·A^T.")
    print("   Thực tế, chọn α = 1/‖A‖_F² = 1/trace(A·A^T) là an toàn vì")
    print("   trace(A·A^T) = Σ λ_i ≥ λ₁, do đó α ≤ 1/λ₁ < 2/λ₁.")
    print("   Điều kiện đủ: ρ(I - α·A·A^T) < 1.")
    print("=" * 100)
    print()

    bieu_dien_ma_tran(A, "Ma trận A")

    # Bước 1: Chọn G_0
    print(">>> BƯỚC 1: CHỌN G_0 (XẤP XỈ BAN ĐẦU) <<<")
    print()
    print("   Tính A·A^T:")
    AAT = A @ A.T
    bieu_dien_ma_tran(AAT, "A·A^T")

    trace_AAT = np.trace(AAT)
    print(f"   trace(A·A^T) = {trace_AAT:.8f}")
    print()

    alpha = 1.0 / trace_AAT
    print(f"   Chọn α = 1 / trace(A·A^T) = 1 / {trace_AAT:.8f} = {alpha:.8f}")
    print()

    G = alpha * A.T
    print(f"   G_0 = α · A^T = {alpha:.8f} · A^T")
    bieu_dien_ma_tran(G, "G_0")

    # Bước 2: Kiểm tra điều kiện hội tụ
    print(">>> BƯỚC 2: KIỂM TRA ĐIỀU KIỆN HỘI TỤ <<<")
    print()
    print("   Điều kiện cần và đủ để hội tụ: ρ(I - A·G_0) < 1")
    print("   trong đó ρ(M) là bán kính phổ (spectral radius) của ma trận M,")
    print("   tức là trị riêng có modul lớn nhất: ρ(M) = max|λ_i(M)|.")
    print()

    R0 = I - A @ G
    print("   Tính R_0 = I - A·G_0:")
    bieu_dien_ma_tran(R0, "R_0 = I - A·G_0")

    eigenvalues = np.linalg.eigvals(R0)
    rho_R0 = max(abs(eigenvalues))
    print(f"   Các trị riêng của R_0: {[f'{ev.real:.8f}{ev.imag:+.8f}j' if abs(ev.imag) > 1e-10 else f'{ev.real:.8f}' for ev in eigenvalues]}")
    print(f"   ρ(R_0) = max|λ_i| = {rho_R0:.8f}")
    print()

    # Sửa: dùng np.isclose để xử lý sai số float
    if rho_R0 >= 1 - 1e-14 or np.isclose(rho_R0, 1.0, atol=1e-10):
        print(f"   ⚠️ CẢNH BÁO: ρ(R_0) = {rho_R0:.8f} ≥ 1 (hoặc rất gần 1).")
        print("   Điều kiện hội tụ chưa thỏa mãn. Tiến hành điều chỉnh α...")
        print()

        found = False
        for scale in [0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]:
            alpha_test = scale / trace_AAT
            G_test = alpha_test * A.T
            R_test = I - A @ G_test
            rho_test = max(abs(np.linalg.eigvals(R_test)))
            print(f"   Thử α = {scale}/trace = {alpha_test:.8f} → ρ(R_0) = {rho_test:.8f}", end="")
            if rho_test < 1 - 1e-10 and not np.isclose(rho_test, 1.0, atol=1e-10):
                alpha = alpha_test
                G = G_test
                R0 = R_test
                rho_R0 = rho_test
                print(" ✅")
                print()
                print(f"   ✅ Đã điều chỉnh: α = {alpha:.8f}, ρ(R_0) = {rho_R0:.8f} < 1")
                bieu_dien_ma_tran(G, "G_0 (đã điều chỉnh)")
                found = True
                break
            else:
                print(" ❌")

        if not found:
            print()
            print("   ❌ KHÔNG tìm được G_0 thỏa mãn ρ(R_0) < 1.")
            print("   Nguyên nhân: Ma trận A có thể suy biến (det = 0),")
            print("   hoặc điều kiện hội tụ quá khắt khe với ma trận này.")
            print("   Phương pháp lặp Newton không áp dụng được.")
            return None
    else:
        print(f"   ✅ Thỏa mãn điều kiện hội tụ: ρ(R_0) = {rho_R0:.8f} < 1")

    print()
    print("   Biện luận:")
    if rho_R0 < 0.5:
        print(f"   • ρ(R_0) = {rho_R0:.4f} < 0.5 → Hội tụ rất nhanh (bình phương mỗi bước).")
    elif rho_R0 < 0.9:
        print(f"   • ρ(R_0) = {rho_R0:.4f} < 0.9 → Hội tụ nhanh.")
    else:
        print(f"   • ρ(R_0) = {rho_R0:.4f} gần 1 → Hội tụ chậm, cần nhiều bước lặp hơn.")
    print(f"   • Sai số giảm theo cấp số nhân với công bội ≈ ρ(R_0)^{{2^k}}.")
    print()

    # Bước 3: Quá trình lặp
    print(">>> BƯỚC 3: QUÁ TRÌNH LẶP <<<")
    print()
    print(f"{'k':>3} | {'‖G_{k+1} - G_k‖_F':>18} | {'‖I - A·G_k‖_F':>18} | {'ρ(R_k)':>12} | {'Nhận xét'}")
    print("-" * 85)

    history = []
    for k in range(max_iter):
        AG = A @ G
        G_new = G @ (2 * I - AG)

        diff = np.linalg.norm(G_new - G, 'fro')
        err_F = np.linalg.norm(I - AG, 'fro')

        # Tính ρ(R_k)
        Rk = I - AG
        rho_Rk = max(abs(np.linalg.eigvals(Rk)))

        history.append((k, diff, err_F, rho_Rk))

        if diff < tol:
            print(f"{k:>3} | {diff:>18.6e} | {err_F:>18.6e} | {rho_Rk:>12.6e} | ✅ Hội tụ")
            break
        else:
            status = "Lặp tiếp"
            if diff < 1e-6:
                status = "Gần hội tụ"
            elif diff < 1e-3:
                status = "Hội tụ tốt"
            print(f"{k:>3} | {diff:>18.6e} | {err_F:>18.6e} | {rho_Rk:>12.6e} | {status}")

        G = G_new
    else:
        print(f"{max_iter:>3} | {diff:>18.6e} | {err_F:>18.6e} | {rho_Rk:>12.6e} | Đạt max iter")

    print()

    # Khử sai số nhỏ
    G[np.abs(G) < 1e-14] = 0.0

    # Kết quả
    print("=" * 100)
    print(" KẾT QUẢ")
    print("=" * 100)
    bieu_dien_ma_tran(G, f"G_{k+1} ≈ A⁻¹")

    # Kiểm tra
    print(">>> KIỂM TRA: A · G ≈ I <<<")
    check = A @ G
    check[np.abs(check) < 1e-12] = 0.0
    bieu_dien_ma_tran(check, "A · G")

    err_final = np.linalg.norm(A @ G - I, 'fro')
    print(f"‖A·G - I‖_F = {err_final:.6e}")
    print()

    if err_final < 1e-10:
        print("✅ Kết quả rất chính xác!")
    elif err_final < 1e-6:
        print("✅ Kết quả chính xác.")
    else:
        print("⚠️ Có sai số đáng kể, có thể cần thêm bước lặp.")

    # So sánh với nghịch đảo thực
    print()
    print(">>> SO SÁNH VỚI numpy.linalg.inv <<<")
    try:
        A_inv = np.linalg.inv(A)
        bieu_dien_ma_tran(A_inv, "A⁻¹ (numpy)")
        err_vs_inv = np.linalg.norm(G - A_inv, 'fro')
        print(f"‖G - A⁻¹‖_F = {err_vs_inv:.6e}")
        if err_vs_inv < 1e-10:
            print("✅ G trùng khớp hoàn toàn với A⁻¹.")
        else:
            print("⚠️ Có sai khác nhỏ do sai số làm tròn.")
    except np.linalg.LinAlgError:
        print("❌ numpy không tính được nghịch đảo (ma trận suy biến).")

    # Bảng tổng hợp
    print()
    print("=" * 100)
    print(" BẢNG TỔNG HỢP TIẾN TRÌNH LẶP")
    print("=" * 100)
    print(f"{'k':>3} | {'‖G_{k+1}-G_k‖_F':>16} | {'‖I-A·G_k‖_F':>16} | {'ρ(R_k)':>14} | {'Tốc độ hội tụ'}")
    print("-" * 75)
    for k, diff, err_F, rho in history:
        if k == 0:
            rate = "Khởi tạo"
        elif diff > 0 and k > 0:
            prev_diff = history[k-1][1]
            if prev_diff > 0:
                ratio = diff / prev_diff
                rate = f"Tỷ số: {ratio:.4f}"
            else:
                rate = "-"
        else:
            rate = "-"
        print(f"{k:>3} | {diff:>16.6e} | {err_F:>16.6e} | {rho:>14.6e} | {rate}")
    print("=" * 100)

    print()
    print("   Biện luận tổng quát:")
    print("   • Phương pháp lặp Newton có tốc độ hội tụ bình phương (quadratic)")
    print("     đối với sai số theo nghĩa: R_{k+1} = R_k², do đó số chữ số")
    print("     chính xác tăng gấp đôi mỗi bước lặp.")
    print("   • Tuy nhiên, nếu ρ(R_0) gần 1 thì cần nhiều bước lặp ban đầu")
    print("     để đạt được vùng hội tụ nhanh.")
    print("   • Ưu điểm: Không cần giải hệ phương trình tuyến tính ở mỗi bước,")
    print("     chỉ cần nhân ma trận, rất hiệu quả với ma trận lớn và thưa.")
    print("=" * 100)
    print()

    return G


if __name__ == "__main__":
    print("=== PHƯƠNG PHÁP LẶP NEWTON TÌM A⁻¹ ===")
    print("Công thức: G_{k+1} = G_k · (2I - A · G_k)")
    print("Điều kiện hội tụ: ρ(I - A·G_0) < 1  (bán kính phổ)")
    print()

    n = read_int("Nhập cấp ma trận vuông n: ")
    A = read_matrix(n, "A")

    newton_inverse_iterative(A)
