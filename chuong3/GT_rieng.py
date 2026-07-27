import numpy as np

def in_ma_tran(M, ten=""):
    """Hàm in ma trận căn lề đẹp mắt để chép thi"""
    if ten:
        print(f"{ten} =")
    hang, cot = M.shape
    for i in range(hang):
        print("  [ " + "  ".join([f"{M[i,j]:8.4f}" for j in range(cot)]) + " ]")
    print()

def in_phuong_trinh_dac_trung(p_coeffs):
    """Hàm in phương trình đặc trưng P(lambda) = 0"""
    n = len(p_coeffs) - 1
    terms = []
    for i, p in enumerate(p_coeffs):
        mu = n - i
        if abs(p) < 1e-10:
            continue
        
        sign = "-" if p < 0 else "+"
        val_str = f"{abs(p):.4f}" if i > 0 else ""
        
        if mu == 0:
            terms.append(f"{sign} {abs(p):.4f}")
        elif mu == 1:
            terms.append(f"{sign} {val_str}λ")
        else:
            terms.append(f"{sign} {val_str}λ^{mu}")
            
    # Xử lý dấu cộng ở đầu (nếu có)
    pt_str = " ".join(terms).strip()
    if pt_str.startswith("+ "):
        pt_str = pt_str[2:]
        
    return pt_str + " = 0"

def faddeev_leverrier_chi_tiet():
    print("🌸 GIẢI GIÁ TRỊ RIÊNG BẰNG FADDEEV - LEVERRIER CỰC CHI TIẾT 🌸")
    print("-" * 75)
    
    while True:
        try:
            n = int(input("Nhập kích thước ma trận vuông n (ví dụ: 6): "))
            if n <= 0:
                print("Kích thước phải lớn hơn 0 nha!")
                continue
            break
        except ValueError:
            print("Cậu nhập số nguyên giúp tớ nhé!")

    print(f"\nNhập từng hàng của ma trận A ({n}x{n}), các số cách nhau bởi khoảng trắng:")
    A_list = []
    for i in range(n):
        while True:
            try:
                row_input = input(f"Hàng {i+1}: ").replace('−', '-').replace('–', '-')
                row = list(map(float, row_input.strip().split()))
                if len(row) != n:
                    print(f"Vui lòng nhập đúng {n} số!")
                    continue
                A_list.append(row)
                break
            except ValueError:
                print("Dữ liệu nhập vào phải là số!")
                
    A = np.array(A_list)
    
    print("\n" + "🌟"*10 + " BÀI LÀM CHI TIẾT " + "🌟"*10)
    in_ma_tran(A, "Ma trận A ban đầu")
    print("Áp dụng thuật toán Faddeev - Leverrier để tìm đa thức đặc trưng.")
    print("Công thức: B_k = A * C_{k-1}; p_k = -(1/k) * Tr(B_k); C_k = B_k + p_k * I")
    print("-" * 75)
    
    I = np.eye(n)
    C_k_minus_1 = I.copy()
    p_coeffs = [1.0] # Hệ số của lambda^n luôn là 1
    
    for k in range(1, n + 1):
        print(f"\n--- BƯỚC k = {k} ---")
        
        # 1. Tính B_k
        if k == 1:
            B_k = A.copy()
            print("1) B_1 = A")
        else:
            B_k = np.dot(A, C_k_minus_1)
            print(f"1) B_{k} = A * C_{k-1}")
        in_ma_tran(B_k, f"B_{k}")
        
        # 2. Tính vết (Trace) và p_k
        tr_B = np.trace(B_k)
        p_k = -tr_B / k
        p_coeffs.append(p_k)
        print(f"2) Tính vết: Tr(B_{k}) = {tr_B:.4f}")
        print(f"   => Hệ số p_{k} = -Tr(B_{k}) / {k} = {p_k:.4f}")
        
        # 3. Tính C_k (bỏ qua bước này ở lần lặp cuối vì không cần thiết)
        if k < n:
            C_k = B_k + p_k * I
            C_k_minus_1 = C_k.copy()
            print(f"3) C_{k} = B_{k} + p_{k} * I")
            in_ma_tran(C_k, f"C_{k}")
            
    print("\n" + "="*60)
    print("KẾT LUẬN PHƯƠNG TRÌNH ĐẶC TRƯNG:")
    print("="*60)
    pt_dac_trung = in_phuong_trinh_dac_trung(p_coeffs)
    print(f"Đa thức đặc trưng của ma trận A là:\n  P(λ) = {pt_dac_trung}")
    
    print("\nGiải phương trình đặc trưng P(λ) = 0, ta được các giá trị riêng:")
    
    # Tìm nghiệm của đa thức
    roots = np.roots(p_coeffs)
    
    # In tất cả các nghiệm
    for i, root in enumerate(roots):
        if abs(root.imag) > 1e-10:
            print(f"  Nghiệm {i+1}: λ = {root.real:.4f} + {root.imag:.4f}i (Nghiệm phức)")
        else:
            print(f"  Nghiệm {i+1}: λ = {root.real:.4f} (Nghiệm thực)")
            
    print("\n🎯 ĐÁP ÁN CUỐI CÙNG: CÁC GIÁ TRỊ RIÊNG THỰC LÀ:")
    real_roots = sorted([root.real for root in roots if abs(root.imag) < 1e-10])
    
    if real_roots:
        for i, val in enumerate(real_roots):
            print(f"  λ_{i+1} = {val:.4f}")
    else:
        print("  (Ma trận không có giá trị riêng thực nào!)")

if __name__ == "__main__":
    faddeev_leverrier_chi_tiet()