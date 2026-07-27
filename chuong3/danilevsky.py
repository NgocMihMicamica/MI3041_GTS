import numpy as np

def read_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("Vui lòng nhập một số nguyên hợp lệ!")

def read_matrix(n, m, name):
    print(f"--- Nhập ma trận {name} ({n}x{m}) ---")
    matrix = []
    for i in range(n):
        while True:
            try:
                # Lọc dấu trừ phake từ PDF
                raw_input = input(f"Nhập hàng {i+1} (các số cách nhau khoảng trắng): ")
                clean_input = raw_input.replace('−', '-').replace('–', '-')
                row = list(map(float, clean_input.split()))
                if len(row) != m:
                    print(f"Hàng phải có đúng {m} phần tử. Vui lòng nhập lại!")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Dữ liệu nhập vào không hợp lệ!")
    return np.array(matrix, dtype=float)

def bieu_dien_ma_tran(matrix, ten=""):
    if ten: print(f"{ten} = ")
    for row in matrix:
        print("  [ " + " ".join([f"{x:9.4f}" for x in row]) + " ]")
    print()

def danilevsky_verbose(A_input):
    n = A_input.shape[0]
    A = A_input.copy()
    
    print("\n================ BẮT ĐẦU PHƯƠNG PHÁP DANILEVSKY ================")
    print("Mục tiêu: Đưa ma trận A về dạng Frobenius để tìm đa thức đặc trưng.")
    print("Sử dụng biến đổi đồng dạng: A_new = M^-1 * A * M\n")
    
    # Bước k chạy ngược từ dòng n-1 về 1 (trong code index là n-1 về 1)
    for k in range(n - 1, 0, -1):
        print(f"--- BƯỚC {n - k}: Xử lý để dòng {k+1} có dạng [0, ..., 1, 0, ..., 0] ---")
        
        pivot = A[k, k - 1]
        
        # Xử lý trường hợp bị "Nerf" (phần tử chốt bằng 0)
        if abs(pivot) < 1e-9:
            print(f"🚨 CẢNH BÁO: Phần tử chốt A[{k+1}, {k}] = 0. Cần tìm cột để hoán vị!")
            swap_ok = False
            for j in range(k - 2, -1, -1):
                if abs(A[k, j]) > 1e-9:
                    # Hoán vị cột k-1 và cột j
                    A[:, [k - 1, j]] = A[:, [j, k - 1]]
                    # Hoán vị dòng k-1 và dòng j
                    A[[k - 1, j], :] = A[[j, k - 1], :]
                    print(f"-> Đã hoán vị đồng thời CỘT {k} với CỘT {j+1} và DÒNG {k} với DÒNG {j+1}.")
                    pivot = A[k, k - 1]
                    swap_ok = True
                    break
            
            if not swap_ok:
                print(f"🚨 LỖI: Toàn bộ các phần tử bên trái A[{k+1}, {k}] đều bằng 0.")
                print("Ma trận phân rã thành các khối độc lập. Thuật toán Danilevsky cơ bản dừng tại đây.")
                return None

        print(f"Phần tử chốt (pivot) = {pivot:.4f}")
        
        # 1. Lập ma trận nghịch đảo M_inv (Cực kỳ dễ nhớ: Thay dòng k-1 bằng dòng k của A)
        M_inv = np.eye(n)
        M_inv[k - 1, :] = A[k, :]
        bieu_dien_ma_tran(M_inv, f"Ma trận M_{k}^-1 (Lấy dòng {k+1} của A lắp vào dòng {k})")
        
        # 2. Lập ma trận M (Áp dụng công thức Danilevsky)
        M = np.eye(n)
        for j in range(n):
            if j == k - 1:
                M[k - 1, j] = 1.0 / pivot
            else:
                M[k - 1, j] = -A[k, j] / pivot
        bieu_dien_ma_tran(M, f"Ma trận M_{k}")
        
        # 3. Tính A_new = M_inv * A * M
        # Đi thi bạn chỉ cần nhân A * M trước, sau đó lấy M_inv nhân với kết quả
        A_temp = np.dot(A, M)
        A = np.dot(M_inv, A_temp)
        
        # Khử sai số phẩy động cực nhỏ để ma trận đẹp
        A[np.abs(A) < 1e-9] = 0.0
        
        bieu_dien_ma_tran(A, f"Ma trận A sau bước {n - k} (A = M_{k}^-1 * A * M_{k})")

    print("--- KẾT LUẬN ---")
    print("Ma trận đã được đưa về dạng Frobenius (F).")
    print("Các hệ số p_1, p_2, ..., p_n của đa thức đặc trưng nằm ở DÒNG 1 của ma trận F.")
    
    # Rút trích hệ số từ dòng 1
    p_coeffs = A[0, :]
    
    poly_terms = [f"λ^{n}"]
    for i, p in enumerate(p_coeffs):
        power = n - (i + 1)
        if abs(p) > 1e-9:
            dau = " - " if p > 0 else " + "
            val = abs(p)
            term = f"{val:.4f}"
            if power > 1: term += f"λ^{power}"
            elif power == 1: term += "λ"
            poly_terms.append(f"{dau}{term}")
            
    print(f"=> Đa thức đặc trưng: P(λ) = {''.join(poly_terms)} = 0")
    print("================ KẾT THÚC ================\n")

if __name__ == "__main__":
    print("=== PHƯƠNG PHÁP DANILEVSKY (CHUẨN CHỈNH) ===")
    print("Chuyên dùng: Đưa ma trận về dạng Frobenius để tìm đa thức đặc trưng.")
    n = read_int("Nhập cấp ma trận vuông n: ")
    A = read_matrix(n, n, "A")
    danilevsky_verbose(A)