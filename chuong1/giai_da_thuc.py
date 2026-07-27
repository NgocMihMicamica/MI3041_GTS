import numpy as np
import math

def read_floats(prompt):
    while True:
        try:
            return [float(x) for x in input(prompt).strip().split()]
        except ValueError:
            print("Vui lòng nhập các số thực cách nhau bởi dấu cách!")

def horner_eval(coeffs, x):
    """Tính giá trị đa thức và đạo hàm bằng Horner"""
    n = len(coeffs)
    p = coeffs[0]
    dp = 0
    for i in range(1, n):
        dp = dp * x + p
        p = p * x + coeffs[i]
    return p, dp

def tach_nghiem(coeffs, a=-100, b=100, step=0.5):
    """Tách nghiệm bằng cách quét đổi dấu"""
    khoang_nghiem = []
    x = a
    p1, _ = horner_eval(coeffs, x)

    while x < b:
        x2 = x + step
        p2, _ = horner_eval(coeffs, x2)
        if p1 * p2 < 0:
            khoang_nghiem.append((x, x2))
        x = x2
        p1 = p2

    return khoang_nghiem

def newton_horner(coeffs, x0, ep=1e-6, max_iter=100):
    """Newton kết hợp Horner để giải đa thức"""
    print(f"\n>>> GIẢI BẰNG NEWTON-HORNER <<<")
    print(f"Điểm khởi tạo x0 = {x0:.4f}")

    x = x0
    for k in range(max_iter):
        p, dp = horner_eval(coeffs, x)

        if abs(dp) < 1e-12:
            print(f"❌ Đạo hàm ≈ 0 tại bước {k}. Dừng!")
            return None

        x_new = x - p / dp
        delta = abs(x_new - x)

        print(f"  Bước {k}: x = {x:.6f}, P(x) = {p:.6e}, P'(x) = {dp:.6f}, |Δx| = {delta:.6e}")

        if delta < ep:
            print(f"✅ Hội tụ: x ≈ {x_new:.8f}")
            return x_new

        x = x_new

    print(f"⚠️ Đạt max iter. x ≈ {x:.6f}")
    return x

def chia_da_thuc(coeffs, root):
    """Chia đa thức cho (x - root) bằng Horner"""
    n = len(coeffs)
    new_coeffs = [coeffs[0]]

    for i in range(1, n - 1):
        new_coeffs.append(new_coeffs[-1] * root + coeffs[i])

    # Kiểm tra số dư
    remainder = new_coeffs[-1] * root + coeffs[-1]
    print(f"  Số dư (phải ≈ 0): {remainder:.6e}")

    return new_coeffs

def giai_da_thuc_batn():
    print("="*80)
    print(" GIẢI ĐA THỨC BẬC n: TÁCH NGHIỆM + NEWTON-HORNER + CHIA ĐA THỨC")
    print("="*80)
    print("Nhập hệ số đa thức từ bậc cao đến bậc thấp:")
    print("VD: x³ - 3x² + 2x - 5 → nhập: 1 -3 2 -5")

    coeffs = read_floats("Nhập hệ số: ")
    n = len(coeffs) - 1

    print(f"\nĐa thức bậc {n}: P(x) = ", end="")
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        if abs(c) > 1e-10:
            if power == 0:
                terms.append(f"{c:.2f}")
            elif power == 1:
                terms.append(f"{c:.2f}x")
            else:
                terms.append(f"{c:.2f}x^{power}")
    print(" + ".join(terms))

    # Bước 1: Tách nghiệm
    print("\n>>> BƯỚC 1: TÁCH NGHIỆM <<<")
    print("Quét đổi dấu trong khoảng [-100, 100] với bước 0.5...")

    khoang_nghiem = tach_nghiem(coeffs)

    if not khoang_nghiem:
        print("❌ Không tìm thấy khoảng đổi dấu. Thử khoảng rộng hơn hoặc đa thức không có nghiệm thực.")
        return

    print(f"Tìm thấy {len(khoang_nghiem)} khoảng phân ly nghiệm:")
    for i, (a, b) in enumerate(khoang_nghiem):
        print(f"  Khoảng {i+1}: [{a:.1f}, {b:.1f}]")

    # Bước 2: Giải từng nghiệm
    print("\n>>> BƯỚC 2: GIẢI TỪNG NGHIỆM BẰNG NEWTON-HORNER <<<")

    all_roots = []
    current_coeffs = coeffs.copy()

    for idx, (a, b) in enumerate(khoang_nghiem):
        if len(current_coeffs) <= 1:
            break

        print(f"\n--- Tìm nghiệm thứ {idx+1} ---")
        x0 = (a + b) / 2

        root = newton_horner(current_coeffs, x0)

        if root is not None:
            all_roots.append(root)

            # Chia đa thức để giảm bậc
            if len(current_coeffs) > 2:
                print(f"\nChia đa thức cho (x - {root:.6f}):")
                current_coeffs = chia_da_thuc(current_coeffs, root)
                print(f"Đa thức mới bậc {len(current_coeffs)-1}: ", end="")
                terms2 = []
                for i, c in enumerate(current_coeffs):
                    power = len(current_coeffs) - 1 - i
                    if abs(c) > 1e-10:
                        if power == 0:
                            terms2.append(f"{c:.4f}")
                        elif power == 1:
                            terms2.append(f"{c:.4f}x")
                        else:
                            terms2.append(f"{c:.4f}x^{power}")
                print(" + ".join(terms2))

    # Bước 3: Tổng kết
    print("\n" + "="*80)
    print(" TỔNG KẾT CÁC NGHIỆM")
    print("="*80)

    if all_roots:
        print(f"Tìm được {len(all_roots)} nghiệm thực:")
        for i, r in enumerate(all_roots):
            # Kiểm tra lại
            p_check, _ = horner_eval(coeffs, r)
            print(f"  x_{i+1} = {r:.8f}  (P(x_{i+1}) = {p_check:.6e})")

        # Nếu còn thiếu nghiệm (bậc > số nghiệm tìm được)
        if len(all_roots) < n:
            print(f"\n⚠️ Chỉ tìm được {len(all_roots)}/{n} nghiệm thực.")
            print(f"   Còn {n - len(all_roots)} nghiệm có thể là nghiệm phức hoặc nằm ngoài khoảng quét.")
    else:
        print("Không tìm được nghiệm thực nào.")

    print("="*80)

if __name__ == "__main__":
    giai_da_thuc_batn()
