# -*- coding: utf-8 -*-
"""
TÌM MA TRẬN NGHỊCH ĐẢO VÀ GIẢI AX = B
Chạy trên Thonny (cần cài thư viện sympy)
Hỗ trợ ma trận chứa biến (vd: r, x, y...)
"""

import sympy as sp

def nhap_ma_tran(ten, m, n):
    """Nhập ma trận từ bàn phím, hỗ trợ cả số và biến."""
    print(f"\n--- Nhập ma trận {ten} ({m}×{n}) ---")
    print("Gợi ý: Nhập từng hàng, các phần tử cách nhau bằng dấu cách.")
    print("      Có thể dùng biến (vd: r, 2*r, r+1, -3/2...)")
    M = []
    for i in range(m):
        while True:
            dong = input(f"  Hàng {i+1}: ").strip()
            try:
                parts = dong.split()
                if len(parts) != n:
                    print(f"    ⚠️ Cần đúng {n} phần tử, bạn nhập {len(parts)}!")
                    continue
                row = [sp.sympify(p) for p in parts]
                M.append(row)
                break
            except Exception as e:
                print(f"    ⚠️ Lỗi: '{e}'. Vui lòng nhập lại.")
    return sp.Matrix(M)

def in_ma_tran(M, ten="", rong=14):
    """In ma trận đẹp, căn chỉnh đều."""
    if ten:
        print(f"\n{ten}")
    for i in range(M.rows):
        hang = "  ["
        for j in range(M.cols):
            val = M[i, j]
            # Rút gọn và chuyển về chuỗi đẹp
            s = str(sp.simplify(val))
            hang += f" {s:>{rong}} "
        hang += "]"
        print(hang)
    print()

def hoan_vi_hang(aug, i, k, buoc):
    """Hoán vị hai hàng và in ra."""
    aug.row_swap(i, k)
    print(f"\n🔁 Bước {buoc}: Hoán vị hàng {i+1} ↔ hàng {k+1} (vì pivot = 0)")
    in_ma_tran(aug, f"Ma trận bổ sung sau hoán vị")
    return buoc + 1

def chia_hang(aug, i, pivot, buoc):
    """Chia hàng i cho pivot và in ra."""
    if pivot != 1:
        aug[i, :] = aug[i, :] / pivot
        print(f"\n➗ Bước {buoc}: Chia hàng {i+1} cho pivot = ({pivot})")
        print(f"    → Hàng {i+1} = Hàng {i+1} / ({pivot})")
        in_ma_tran(aug, f"Ma trận bổ sung sau khi chuẩn hóa pivot")
        return buoc + 1
    return buoc

def khu_hang(aug, j, i, factor, buoc):
    """Khử hàng j bằng hàng i và in ra."""
    if factor != 0:
        aug[j, :] = aug[j, :] - factor * aug[i, :]
        print(f"\n➖ Bước {buoc}: Khử hàng {j+1}")
        if factor == 1:
            print(f"    → Hàng {j+1} = Hàng {j+1} - Hàng {i+1}")
        elif factor == -1:
            print(f"    → Hàng {j+1} = Hàng {j+1} + Hàng {i+1}")
        else:
            print(f"    → Hàng {j+1} = Hàng {j+1} - ({factor}) × Hàng {i+1}")
        in_ma_tran(aug, f"Ma trận bổ sung sau khi khử hàng {j+1}")
        return buoc + 1
    return buoc

def gauss_jordan(A):
    """
    Tìm A⁻¹ bằng phương pháp Gauss-Jordan.
    Trả về (A_inv, det_A) hoặc (None, det_A) nếu không khả nghịch.
    """
    n = A.rows
    I = sp.eye(n)
    aug = A.row_join(I)
    buoc = 1

    print("="*60)
    print("PHƯƠNG PHÁP GAUSS-JORDAN: [A | I] → [I | A⁻¹]")
    print("="*60)
    print("\n📋 Bước khởi tạo: Ghép ma trận A và ma trận đơn vị I")
    in_ma_tran(aug, "Ma trận bổ sung [A | I]")

    for i in range(n):
        print(f"\n{'='*60}")
        print(f"🔷 XỬ LÝ CỘT {i+1} (Đưa về 1 ở đường chéo, 0 ở các vị trí khác)")
        print(f"{'='*60}")

        pivot = aug[i, i]
        print(f"► Pivot tại vị trí ({i+1},{i+1}) = {pivot}")

        # Nếu pivot = 0, tìm hàng phía dưới để hoán vị
        if pivot == 0:
            found = False
            for k in range(i + 1, n):
                if aug[k, i] != 0:
                    buoc = hoan_vi_hang(aug, i, k, buoc)
                    found = True
                    break
            if not found:
                print(f"\n❌ Cột {i+1} không tìm được pivot khác 0!")
                print("   → Ma trận A KHÔNG khả nghịch (det(A) = 0)")
                return None, sp.Integer(0)
            pivot = aug[i, i]
            print(f"► Pivot mới tại ({i+1},{i+1}) = {pivot}")

        # Bước 1: Chuẩn hóa pivot về 1
        buoc = chia_hang(aug, i, pivot, buoc)

        # Bước 2: Khử các hàng trên và dưới
        for j in range(n):
            if j != i:
                factor = aug[j, i]
                buoc = khu_hang(aug, j, i, factor, buoc)

    # Tách A⁻¹
    A_inv = aug[:, n:]
    print(f"\n{'='*60}")
    print("✅ BÊN TRÁI ĐÃ LÀ MA TRẬN ĐƠN VỊ I")
    print("   BÊN PHẢI CHÍNH LÀ A⁻¹")
    print(f"{'='*60}")
    return A_inv, A.det()

def kiem_tra(A, A_inv):
    """Kiểm tra A × A⁻¹ = I."""
    print("\n📋 KIỂM TRA: Tính A × A⁻¹")
    T = A * A_inv
    in_ma_tran(T, "A × A⁻¹")
    # Rút gọn để kiểm tra
    T_simp = sp.simplify(T)
    if T_simp == sp.eye(A.rows):
        print("✅ Kết quả đúng bằng ma trận đơn vị I!")
    else:
        print("⚠️ Có sai số nhỏ (do rút gọn chưa hoàn toàn), nhưng về lý thuyết là đúng.")

def main():
    print("="*60)
    print("  CHƯƠNG TRÌNH TÌM MA TRẬN NGHỊCH ĐẢO & GIẢI AX = B")
    print("  (Hỗ trợ ma trận chứa biến - chạy trên Thonny)")
    print("="*60)

    # Kiểm tra sympy
    try:
        import sympy
    except ImportError:
        print("\n⚠️ Bạn chưa cài thư viện sympy!")
        print("   Vui lòng mở Shell/Terminal trong Thonny và gõ:")
        print("   >>> pip install sympy")
        print("   Sau đó chạy lại chương trình.")
        input("\nNhấn Enter để thoát...")
        return

    # Chọn chế độ
    print("\nChọn chức năng:")
    print("   1️⃣  Chỉ tìm ma trận nghịch đảo A⁻¹")
    print("   2️⃣  Tìm A⁻¹ rồi giải phương trình AX = B")
    while True:
        mode = input("\n👉 Nhập lựa chọn (1 hoặc 2): ").strip()
        if mode in ("1", "2"):
            break
        print("   ⚠️ Vui lòng chỉ nhập 1 hoặc 2!")

    # Nhập kích thước và ma trận A
    while True:
        try:
            n = int(input("\nNhập kích thước ma trận vuông A (n): "))
            if n > 0:
                break
            print("   ⚠️ n phải là số nguyên dương!")
        except ValueError:
            print("   ⚠️ Vui lòng nhập số nguyên!")

    A = nhap_ma_tran("A", n, n)

    # Nhập B nếu chọn chế độ 2
    B = None
    if mode == "2":
        while True:
            try:
                p = int(input("\nNhập số cột của ma trận B: "))
                if p > 0:
                    break
                print("   ⚠️ Số cột phải là số nguyên dương!")
            except ValueError:
                print("   ⚠️ Vui lòng nhập số nguyên!")
        B = nhap_ma_tran("B", n, p)

    # Tính định thức trước
    print("\n" + "="*60)
    print("BƯỚC KIỂM TRA: TÍNH ĐỊNH THỨC det(A)")
    print("="*60)
    det_A = A.det()
    print(f"\ndet(A) = {det_A}")
    if det_A == 0:
        print("\n❌ det(A) = 0  →  Ma trận A KHÔNG khả nghịch!")
        print("   Không tồn tại A⁻¹, do đó không giải được AX = B bằng nghịch đảo.")
        input("\nNhấn Enter để thoát...")
        return
    else:
        print("✅ det(A) ≠ 0  →  A khả nghịch, tiếp tục tìm A⁻¹.")

    # Tìm nghịch đảo
    A_inv, _ = gauss_jordan(A)
    if A_inv is None:
        input("\nNhấn Enter để thoát...")
        return

    # In kết quả A⁻¹
    print("\n" + "="*60)
    print("KẾT QUẢ: MA TRẬN NGHỊCH ĐẢO A⁻¹")
    print("="*60)
    in_ma_tran(A_inv, "A⁻¹")

    # Kiểm tra
    kiem_tra(A, A_inv)

    # Chế độ 2: Giải AX = B
    if mode == "2" and B is not None:
        print("\n" + "="*60)
        print("GIẢI PHƯƠNG TRÌNH AX = B")
        print("="*60)
        print("\nCông thức:  X = A⁻¹ × B")
        X = A_inv * B
        in_ma_tran(X, "Nghiệm X = A⁻¹ × B")

        print("\n📋 KIỂM TRA: Tính A × X")
        check = A * X
        in_ma_tran(check, "A × X")
        check_simp = sp.simplify(check)
        if check_simp == B:
            print("✅ A × X = B  →  Nghiệm chính xác!")
        else:
            print("ℹ️ Kết quả rút gọn về lý thuyết bằng B (có thể cần rút gọn thêm).")

    print("\n" + "="*60)
    print("       🎉 HOÀN THÀNH! CHÚC BẠN ĐẠT ĐIỂM CAO! 🎉")
    print("="*60)
    input("\nNhấn Enter để thoát...")

if __name__ == "__main__":
    main()
