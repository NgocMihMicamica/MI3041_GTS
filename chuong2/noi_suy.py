import numpy as np

def read_points():
    """Nhập các điểm dữ liệu (x, y)"""
    print("--- Nhập các điểm dữ liệu ---")
    print("Nhập số điểm n, sau đó nhập từng cặp (x, y)")

    while True:
        try:
            n = int(input("Số điểm n = ").strip())
            if n < 2:
                print("Cần ít nhất 2 điểm!")
                continue
            break
        except ValueError:
            print("Vui lòng nhập số nguyên!")

    x = []
    y = []
    for i in range(n):
        while True:
            try:
                line = input(f"Điểm {i+1} (x y): ").strip().split()
                if len(line) != 2:
                    print("Nhập đúng 2 số cách nhau bởi dấu cách!")
                    continue
                x.append(float(line[0]))
                y.append(float(line[1]))
                break
            except ValueError:
                print("Dữ liệu không hợp lệ!")

    return np.array(x), np.array(y)

def noi_suy_lagrange(x, y):
    """Nội suy Lagrange - trình bày chi tiết"""
    n = len(x)

    print("\n================ NỘI SUY LAGRANGE ================")
    print(f"Số điểm: {n}")
    print("Các điểm dữ liệu:")
    for i in range(n):
        print(f"  (x_{i}, y_{i}) = ({x[i]:.4f}, {y[i]:.4f})")

    print("\nCông thức Lagrange:")
    print("  P(x) = Σ y_i · L_i(x)")
    print("  trong đó L_i(x) = Π (x - x_j)/(x_i - x_j) với j ≠ i")

    # Tính các hệ số đa thức Lagrange
    coeffs = np.zeros(n)

    for i in range(n):
        # Tính L_i(x)
        Li = np.array([1.0])  # Bắt đầu từ đa thức bậc 0
        mau = 1.0

        print(f"\n--- Tính L_{i}(x) ---")
        for j in range(n):
            if i != j:
                # Nhân với (x - x_j)
                Li = np.convolve(Li, [1, -x[j]])
                mau *= (x[i] - x[j])
                print(f"  Nhân với (x - {x[j]:.4f}), mẫu *= ({x[i]:.4f} - {x[j]:.4f}) = {x[i]-x[j]:.4f}")

        Li = Li / mau
        print(f"  L_{i}(x) = ({' + '.join([f'{Li[k]:.4f}x^{n-1-k}' if n-1-k > 1 else (f'{Li[k]:.4f}x' if n-1-k == 1 else f'{Li[k]:.4f}') for k in range(len(Li))])}) / {mau:.4f}")

        coeffs += y[i] * Li

    print("\n--- ĐA THỨC NỘI SUY LAGRANGE ---")
    print("P(x) = ", end="")
    terms = []
    for i in range(len(coeffs)):
        power = len(coeffs) - 1 - i
        if abs(coeffs[i]) > 1e-10:
            if power == 0:
                terms.append(f"{coeffs[i]:.6f}")
            elif power == 1:
                terms.append(f"{coeffs[i]:.6f}x")
            else:
                terms.append(f"{coeffs[i]:.6f}x^{power}")
    print(" + ".join(terms))

    return coeffs

def bang_ti_hieu(x, y):
    """Tính bảng tỉ hiệu Newton"""
    n = len(x)
    # Ma trận tỉ hiệu
    F = np.zeros((n, n))
    F[:, 0] = y

    print("\n--- BẢNG TỈ HIỆU NEWTON ---")

    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i+1, j-1] - F[i, j-1]) / (x[i+j] - x[i])
            print(f"  f[x_{i},...,x_{i+j}] = (F[{i+1},{j-1}] - F[{i},{j-1}]) / (x_{i+j} - x_{i})")
            print(f"                      = ({F[i+1,j-1]:.6f} - {F[i,j-1]:.6f}) / ({x[i+j]:.4f} - {x[i]:.4f})")
            print(f"                      = {F[i,j]:.6f}")

    # In bảng
    print("\nBảng tỉ hiệu:")
    header = f"{'i':>3} | {'x_i':>10} | {'f[x_i]':>12}"
    for j in range(1, n):
        header += f" | {'f[...]':>12}"
    print(header)
    print("-" * len(header))

    for i in range(n):
        row = f"{i:>3} | {x[i]:>10.4f} | {F[i,0]:>12.6f}"
        for j in range(1, n - i):
            row += f" | {F[i,j]:>12.6f}"
        print(row)

    return F[0, :]  # Hệ số trên đường chéo

def noi_suy_newton(x, y):
    """Nội suy Newton tiến với bảng tỉ hiệu"""
    n = len(x)

    print("\n================ NỘI SUY NEWTON ================")
    print(f"Số điểm: {n}")
    print("Các điểm dữ liệu:")
    for i in range(n):
        print(f"  (x_{i}, y_{i}) = ({x[i]:.4f}, {y[i]:.4f})")

    print("\nCông thức Newton:")
    print("  P(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1) + ...")

    coeffs_ti_hieu = bang_ti_hieu(x, y)

    print("\n--- ĐA THỨC NỘI SUY NEWTON ---")
    print("P(x) = ", end="")
    terms = []
    for i in range(n):
        if abs(coeffs_ti_hieu[i]) > 1e-10:
            if i == 0:
                terms.append(f"{coeffs_ti_hieu[i]:.6f}")
            else:
                factor = ""
                for j in range(i):
                    factor += f"(x - {x[j]:.4f})"
                terms.append(f"{coeffs_ti_hieu[i]:.6f}·{factor}")
    print(" + ".join(terms))

    # Chuyển về dạng chuẩn ax^n + ...
    result = np.array([coeffs_ti_hieu[0]])
    for i in range(1, n):
        # Nhân với (x - x_{i-1})
        temp = np.convolve(result, [1, -x[i-1]])
        # Thêm hệ số mới
        if len(temp) < n:
            temp = np.pad(temp, (0, n - len(temp)))
        temp[-1] += coeffs_ti_hieu[i]  # Thêm vào hệ số tự do... 
        # Cách đúng: cộng đa thức
        new_term = np.zeros(n)
        new_term[n-1-i] = coeffs_ti_hieu[i]
        for j in range(i):
            # Nhân các (x - x_k)
            pass
        # Đơn giản: dùng np.polyfit để kiểm tra

    # Dùng cách đơn giản hơn
    from numpy.polynomial import polynomial as P
    # Chuyển về dạng chuẩn
    standard_coeffs = np.zeros(n)

    def multiply_poly(p1, p2):
        return np.convolve(p1, p2)

    current = np.array([1.0])
    standard_coeffs[0] = coeffs_ti_hieu[0]  # Hệ số bậc cao nhất

    # Tính từng phần
    product = np.array([1.0])
    for i in range(n):
        if i > 0:
            product = np.convolve(product, [1, -x[i-1]])
        # Cộng vào
        if len(product) > len(standard_coeffs):
            standard_coeffs = np.pad(standard_coeffs, (0, len(product) - len(standard_coeffs)))
        standard_coeffs += coeffs_ti_hieu[i] * np.pad(product, (len(standard_coeffs) - len(product), 0))

    # Đảo ngược để bậc cao nhất đầu tiên
    standard_coeffs = standard_coeffs[::-1]

    print("\nDạng chuẩn:")
    print("P(x) = ", end="")
    terms = []
    for i in range(len(standard_coeffs)):
        power = len(standard_coeffs) - 1 - i
        if abs(standard_coeffs[i]) > 1e-10:
            if power == 0:
                terms.append(f"{standard_coeffs[i]:.6f}")
            elif power == 1:
                terms.append(f"{standard_coeffs[i]:.6f}x")
            else:
                terms.append(f"{standard_coeffs[i]:.6f}x^{power}")
    print(" + ".join(terms))

    return coeffs_ti_hieu, standard_coeffs

def danh_gia_sai_so(x, y, coeffs, x_test):
    """Đánh giá sai số nội suy tại điểm x_test"""
    # Giá trị đúng (nếu có)
    # Giá trị nội suy
    P_test = np.polyval(coeffs, x_test)
    print(f"\nGiá trị nội suy tại x = {x_test}: P({x_test}) = {P_test:.6f}")

    # Sai số cận trên (công thức Lagrange)
    n = len(x)
    omega = 1.0
    for xi in x:
        omega *= (x_test - xi)

    print(f"  ω(x) = Π(x - x_i) = {omega:.6e}")
    print(f"  Sai số |f(x) - P(x)| ≤ |ω(x)| · M_{n} / {math.factorial(n)}")
    print(f"  (với M_{n} = max|f^{n}(ξ)| trên khoảng chứa các điểm)")

    return P_test

if __name__ == "__main__":
    import math

    print("="*80)
    print(" NỘI SUY NEWTON VÀ LAGRANGE")
    print("="*80)

    x, y = read_points()

    print("\nChọn phương pháp:")
    print("1. Nội suy Lagrange")
    print("2. Nội suy Newton (bảng tỉ hiệu)")
    print("3. Cả hai")

    choice = input("Chọn (1, 2 hoặc 3): ").strip()

    if choice in ['1', '3']:
        coeffs_lag = noi_suy_lagrange(x, y)

    if choice in ['2', '3']:
        coeffs_newton, standard = noi_suy_newton(x, y)

    # Đánh giá tại điểm bất kỳ
    print("\n--- ĐÁNH GIÁ TẠI ĐIỂM BẤT KỲ ---")
    x_test = float(input("Nhập x cần tính giá trị nội suy: ").strip())

    if choice in ['1', '3']:
        P_lag = np.polyval(coeffs_lag, x_test)
        print(f"Lagrange: P({x_test}) = {P_lag:.6f}")

    if choice in ['2', '3']:
        P_new = np.polyval(standard, x_test)
        print(f"Newton: P({x_test}) = {P_new:.6f}")
