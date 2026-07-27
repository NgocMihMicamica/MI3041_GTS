import numpy as np

def read_matrix_interactive(rows, cols, name="A"):
    print(f"Nhập các hàng của ma trận {name} ({rows}x{cols}), các số cách nhau bằng khoảng trắng:")
    matrix = []
    for i in range(rows):
        while True:
            try:
                line = input(f"Hàng {i+1}: ").replace('−', '-').replace('–', '-').split()
                if len(line) != cols:
                    print(f"Vui lòng nhập đúng {cols} phần tử!")
                    continue
                matrix.append([float(x) for x in line])
                break
            except ValueError:
                print("Dữ liệu nhập vào phải là số!")
    return np.array(matrix, dtype=float)

def read_vector_interactive(size, name="b"):
    while True:
        try:
            print(f"Nhập vector {name} gồm {size} phần tử (cách nhau bởi khoảng trắng):")
            line = input().replace('−', '-').replace('–', '-').split()
            if len(line) != size:
                print(f"Vui lòng nhập đúng {size} phần tử!")
                continue
            return np.array([float(x) for x in line], dtype=float)
        except ValueError:
            print("Dữ liệu nhập vào phải là số!")

def bieu_dien_ma_tran(matrix, ten=""):
    if ten:
        print(f"{ten} = ")
    for row in matrix:
        print("  [ " + " ".join([f"{x:9.4f}" for x in row]) + " ]")
    print()

def bieu_dien_vector(vector, ten=""):
    if ten:
        print(f"{ten} = ")
    print("  [ " + " ".join([f"{x:9.4f}" for x in vector]) + " ]^T")
    print()

def kiem_tra_cheo_troi_hang(A):
    n = len(A)
    for i in range(n):
        tong_hang = sum(abs(A[i, j]) for j in range(n) if i != j)
        if abs(A[i, i]) <= tong_hang:
            return False
    return True

def chuan_vo_cung_ma_tran(B):
    return max(sum(abs(x) for x in row) for row in B)

def format_num(val):
    """Hàm phụ trợ để in số nguyên thì bỏ đuôi .0, số thực thì giữ nguyên"""
    if val.is_integer():
        return str(int(val))
    return f"{val:g}"

def seidel_chinh_tac_verbose(B, d, x0, k_buoc):
    n = len(d)
    x = np.array(x0, dtype=float)
    
    arr_str = "\n    ".join([format_num(val) for val in x])
    print(f"\nChọn vectơ giả thiết ban đầu:\n\n         X^(0) = [ {arr_str} ]^T")
    
    for buoc in range(1, k_buoc + 1):
        print(f"\nBước lặp k = {buoc}:")
        if buoc == 1:
            print("Thay các giá trị vào công thức lặp, ta có:")
        elif buoc == 2:
            print("Tiếp tục sử dụng các nghiệm vừa tìm được ở bước 1:")
        else:
            print("Tương tự, ta tính được:")
            
        for i in range(n):
            # Tạo chuỗi biểu thức biến
            terms = []
            for j in range(n):
                val = B[i, j]
                if val == 0: continue
                # Seidel: j < i dùng mũ hiện tại (buoc), j >= i dùng mũ trước đó (buoc - 1)
                sup = buoc if j < i else buoc - 1
                
                val_str = format_num(abs(val))
                if not terms:
                    sign = "-" if val < 0 else ""
                    terms.append(f"{sign}{val_str}x_{j+1}^({sup})")
                else:
                    sign = "-" if val < 0 else "+"
                    terms.append(f"{sign} {val_str}x_{j+1}^({sup})")
            
            sum_str = " ".join(terms) if terms else "0"
            
            # Tính toán kết quả thực sự
            tong = d[i]
            for j in range(n):
                tong += B[i, j] * x[j]
            x[i] = tong
            
            d_str = format_num(d[i])
            print(f"  x_{i+1}^({buoc}) = {d_str} + [ {sum_str} ] = {x[i]:.4f}")
            
        arr_str = "  ".join([f"{val:.4f}" for val in x])
        print(f"=> X^({buoc}) = [ {arr_str} ]^T")

def seidel_standard_verbose(A, b, x0, k_buoc):
    n = len(b)
    x = np.array(x0, dtype=float)
    
    arr_str = "\n    ".join([format_num(val) for val in x])
    print(f"\nChọn vectơ giả thiết ban đầu:\n\n         X^(0) = [ {arr_str} ]^T")
    
    for buoc in range(1, k_buoc + 1):
        print(f"\nBước lặp k = {buoc}:")
        if buoc == 1:
            print("Thay các giá trị vào công thức lặp, ta có:")
        elif buoc == 2:
            print("Tiếp tục sử dụng các nghiệm vừa tìm được ở bước 1:")
        else:
            print("Tương tự, ta tính được:")
            
        for i in range(n):
            if A[i, i] == 0:
                print(f"Lỗi: A[{i+1},{i+1}] = 0, không thể chia cho 0.")
                return
            
            # Tạo chuỗi biểu thức biến cho phép trừ
            terms = []
            for j in range(n):
                if i == j: continue
                val = A[i, j]
                if val == 0: continue
                
                sup = buoc if j < i else buoc - 1
                val_str = format_num(abs(val))
                
                if not terms:
                    sign = "-" if val < 0 else ""
                    terms.append(f"{sign}{val_str}x_{j+1}^({sup})")
                else:
                    sign = "-" if val < 0 else "+"
                    terms.append(f"{sign} {val_str}x_{j+1}^({sup})")
            
            sum_str = " ".join(terms) if terms else "0"
            
            # Tính toán kết quả thực sự
            tong = b[i]
            for j in range(n):
                if i != j:
                    tong -= A[i, j] * x[j]
            tong /= A[i, i]
            x[i] = tong
            
            b_str = format_num(b[i])
            aii_str = format_num(A[i,i])
            
            print(f"  x_{i+1}^({buoc}) = (1/{aii_str}) * [ {b_str} - ({sum_str}) ] = {x[i]:.4f}")
            
        arr_str = "  ".join([f"{val:.4f}" for val in x])
        print(f"=> X^({buoc}) = [ {arr_str} ]^T")

if __name__ == "__main__":
    print("=====================================================")
    print(" CHƯƠNG TRÌNH SEIDEL IN KẾT QUẢ ĐỂ CHÉP THI (FULL) ")
    print("=====================================================")
    print("Chọn dạng bài toán gốc trong đề thi:")
    print("1. Hệ có dạng chính tắc : x = Bx + d")
    print("2. Hệ có dạng tiêu chuẩn: Ax = b")
    
    while True:
        choice = input("Nhập lựa chọn của bạn (1 hoặc 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Lựa chọn không hợp lệ!")

    m = int(input("Nhập số HÀNG (số phương trình) của ma trận: "))
    n = int(input("Nhập số CỘT (số ẩn) của ma trận: "))
    k_buoc = int(input("Nhập số lần lặp k yêu cầu: "))

    print("\nĐề bài có cho trước vector khởi tạo x0 không cậu nhỉ? (y/n - mặc định là vector 0)")
    if input().strip().lower() == 'y':
        x0 = read_vector_interactive(n, "x0")
    else:
        x0 = np.zeros(n)
        print("Tớ đã tự động gán x0 là vector không rùi nhé!")

    if choice == '1':
        print("\n--- NHẬP LIỆU BÀI TOÁN x = Bx + d ---")
        B = read_matrix_interactive(m, n, "B")
        d = read_vector_interactive(m, "d")
        
        print("\n=== LẬP LUẬN ĐI THI KHÂU ĐẦU VÀO ===")
        if m == n:
            chuan_B = chuan_vo_cung_ma_tran(B)
            if chuan_B < 1:
                print(f"Nhận xét: Ma trận B vuông và ||B|| = {chuan_B:.4f} < 1.")
                print("=> Hội tụ. Tiến hành thuật toán Seidel dạng chính tắc.")
                seidel_chinh_tac_verbose(B, d, x0, k_buoc)
            else:
                print(f"Nhận xét: Ma trận B vuông nhưng CHƯA CHÍNH TẮC (||B|| = {chuan_B:.4f} >= 1).")
                print("Đưa về dạng Ax = b với A = (I - B), sau đó nhân cả 2 vế với A^T để đảm bảo hội tụ!")
                I = np.eye(n)
                A = I - B
                A_new = np.dot(A.T, A)
                d_new = np.dot(A.T, d)
                
                print("\n--- CHI TIẾT HỆ HỘI TỤ MỚI ---")
                print("Hệ hội tụ mới: A_new * x = d_new (với A_new = A^T * A và d_new = A^T * d)")
                bieu_dien_ma_tran(A_new, "A_new")
                bieu_dien_vector(d_new, "d_new")
                
                seidel_standard_verbose(A_new, d_new, x0, k_buoc)
        else:
            print(f"Nhận xét: Ma trận B không vuông ({m}x{n}). Đưa về (I - B)x = d rồi áp dụng Bình phương tối tiểu:")
            I = np.zeros((m, n))
            np.fill_diagonal(I, 1)
            A = I - B
            A_new = np.dot(A.T, A)
            d_new = np.dot(A.T, d)
            
            print("\n--- CHI TIẾT HỆ VUÔNG MỚI ---")
            print("Hệ hội tụ mới: A_new * x = d_new (với A_new = A^T * A và d_new = A^T * d)")
            bieu_dien_ma_tran(A_new, "A_new")
            bieu_dien_vector(d_new, "d_new")
            
            seidel_standard_verbose(A_new, d_new, x0, k_buoc)

    elif choice == '2':
        print("\n--- NHẬP LIỆU BÀI TOÁN Ax = b ---")
        A = read_matrix_interactive(m, n, "A")
        b = read_vector_interactive(m, "b")
        
        print("\n=== LẬP LUẬN ĐI THI KHÂU ĐẦU VÀO ===")
        if m == n:
            if kiem_tra_cheo_troi_hang(A):
                print("Nhận xét: Ma trận A là MA TRẬN VUÔNG và CHÉO TRỘI HÀNG.")
                print("=> Đảm bảo hội tụ. Tiến hành giải trực tiếp.")
                seidel_standard_verbose(A, b, x0, k_buoc)
            else:
                print("Nhận xét: Ma trận A vuông nhưng CHƯA CHÍNH TẮC (chưa chéo trội).")
                print("=> Phải nhân cả 2 vế với ma trận chuyển vị A^T để đưa về hệ đối xứng xác định dương (đảm bảo hội tụ)!")
                A_new = np.dot(A.T, A)
                b_new = np.dot(A.T, b)
                
                print("\n--- CHI TIẾT HỆ HỘI TỤ MỚI ---")
                print("Hệ hội tụ mới: A_new * x = b_new (với A_new = A^T * A và b_new = A^T * b)")
                bieu_dien_ma_tran(A_new, "A_new")
                bieu_dien_vector(b_new, "b_new")
                
                seidel_standard_verbose(A_new, b_new, x0, k_buoc)
        else:
            print(f"Nhận xét: Ma trận A không vuông ({m}x{n}). Áp dụng Bình phương tối tiểu:")
            A_new = np.dot(A.T, A)
            b_new = np.dot(A.T, b)
            
            print("\n--- CHI TIẾT HỆ VUÔNG MỚI ---")
            print("Hệ hội tụ mới: A_new * x = b_new (với A_new = A^T * A và b_new = A^T * b)")
            bieu_dien_ma_tran(A_new, "A_new")
            bieu_dien_vector(b_new, "b_new")
            
            seidel_standard_verbose(A_new, b_new, x0, k_buoc)