import math

EPS = 1e-12

def sign(x):
    if x > 0: return 1
    elif x < 0: return -1
    return 0

def nhap_so_thuc(thong_bao):
    while True:
        try: 
            return float(input(thong_bao).strip())
        except ValueError: 
            print("=> Cậu nhập số thực hợp lệ nha (≧◡≦)")

# =====================================================================
# HÀM PHÂN TÍCH BIỂU THỨC THÔNG MINH
# =====================================================================
def tao_ham(bieu_thuc):
    # Cắt bỏ đuôi "= 0" nếu người dùng gõ cả phương trình
    if '=' in bieu_thuc:
        bieu_thuc = bieu_thuc.split('=')[0].strip()

    # Xử lý ký hiệu mũ phổ thông toán học thành cú pháp Python
    bieu_thuc = bieu_thuc.replace('^', '**')

    # Tạo môi trường tính toán giàu toán học
    boi_canh = {
        'math': math, 'sin': math.sin, 'cos': math.cos, 
        'tan': math.tan, 'exp': math.exp, 'log': math.log, 
        'ln': math.log, 'sqrt': math.sqrt, 'pi': math.pi,
        'e': math.e
    }

    def f_x(x_val):
        boi_canh['x'] = float(x_val)
        return eval(bieu_thuc, boi_canh)

    return f_x

# =====================================================================
# HÀM ĐỔI BIẾN x -> -x (CHO TÌM NGHIỆM ÂM)
# =====================================================================
def tao_ham_doi_bien_am(f_goc):
    """Tạo hàm mới: g(x) = f(-x), để tìm nghiệm âm của f(x)"""
    def g_x(x_val):
        return f_goc(-x_val)
    return g_x

# =====================================================================
# RADAR QUÉT TÌM TẤT CẢ CÁC KHOẢNG PHÂN LY NGHIỆM
# =====================================================================
def quet_tat_ca_khoang_nghiem(f, start=-10, end=10, step=0.1):
    khoang_nghiem = []
    x_truoc = start
    try:
        f_truoc = f(x_truoc)
    except:
        f_truoc = 0

    so_mau = int((end - start) / step)
    for i in range(1, so_mau + 1):
        x_hien_tai = start + i * step
        try:
            f_hien_tai = f(x_hien_tai)
            if sign(f_truoc) * sign(f_hien_tai) < 0:
                khoang_nghiem.append((x_truoc, x_hien_tai))
            x_truoc = x_hien_tai
            f_truoc = f_hien_tai
        except:
            continue
    return khoang_nghiem

# =====================================================================
# LÕI PHƯƠNG PHÁP CHIA ĐÔI (TÁCH RIÊNG ĐỂ TÁI SỬ DỤNG)
# =====================================================================
def chia_doi_core(f, a_ban_dau, b_ban_dau, ep, sign_a, sign_b, nghiem_am_mode=False, ten_nghiem="x"):
    """Lõi chia đôi, có thể dùng cho cả nghiệm dương và âm (sau đổi biến)"""

    a, b = a_ban_dau, b_ban_dau

    # --- TÍNH TIÊN NGHIỆM ĐẦY ĐỦ CÁC BƯỚC VÀO BÀI LÀM TỰ LUẬN ---
    gia_tri_log = math.log2(b - a) - math.log2(ep)
    n_tien_nghiem = math.ceil(gia_tri_log - 1)
    n_tien_nghiem = max(0, n_tien_nghiem)

    print("\n" + "🌸"*38)
    if nghiem_am_mode:
        print("📝 BÀI LÀM TỰ LUẬN: PHƯƠNG PHÁP CHIA ĐÔI (TÌM NGHIỆM ÂM)")
    else:
        print("📝 BÀI LÀM TỰ LUẬN THAM KHẢO: PHƯƠNG PHÁP CHIA ĐÔI")
    print("🌸"*38)
    print(f"Khoảng phân ly nghiệm ban đầu: [{a}, {b}]")
    print(f"\n👉 Đánh giá tiên nghiệm số bước lặp lý thuyết:")
    print(f"   - Công thức: n >= log2((b - a) / ε) - 1")
    print(f"   - Thay số:   n >= log2(({b} - {a}) / {ep}) - 1")
    print(f"   - Kết quả:   n >= {gia_tri_log - 1:.4f}  => Cần thực hiện tối thiểu n = {n_tien_nghiem} bước lặp.")

    # --- IN BẢNG SỐ LIỆU CHUẨN ĐẸP ---
    print("\n" + "="*115)
    header = f"{'n':<4} | {'a_n':<12} | {'b_n':<12} | {'x_n':<12} | {'Dấu f(x_n)':<12} | {'Sai số bước lặp Δ = (b-a)/2^(n+1)':<25}"
    print(header)
    print("-" * 115)

    n = 0
    while True:
        root = (a + b) / 2
        try:
            f_root = f(root)
        except Exception as e:
            print(f"\n❌ LỖI phát sinh khi tính f(x) tại x = {root}: {e}")
            break

        delta = (b_ban_dau - a_ban_dau) / (2 ** (n + 1))
        dau_f_root = "+" if f_root > 0 else ("-" if f_root < 0 else "0")

        row = f"{n:<4} | {a:<12.5f} | {b:<12.5f} | {root:<12.5f} | {dau_f_root:^12} | {delta:<25.5f}"
        print(row)

        if abs(f_root) < EPS:
            print("-" * 115)
            if nghiem_am_mode:
                print(f"🌟 ĐẠT NGHIỆM CHÍNH XÁC TUYỆT ĐỐI: t = {root:.8f}  =>  {ten_nghiem} = -t = {-root:.8f}")
            else:
                print(f"🌟 ĐẠT NGHIỆM CHÍNH XÁC TUYỆT ĐỐI: {ten_nghiem} = {root:.8f}")
            break

        if sign_a * sign(f_root) > 0:
            a = root
        else:
            b = root

        # Điều kiện dừng kiểm tra tại bước n
        if delta <= ep:
            print("-" * 115)
            if nghiem_am_mode:
                print(f"🌟 KẾT LUẬN CUỐI CÙNG: Nghiệm gần đúng t = {root:.5f}  =>  {ten_nghiem} = -t = {-root:.5f}")
            else:
                print(f"🌟 KẾT LUẬN CUỐI CÙNG: Nghiệm gần đúng của phương trình là {ten_nghiem} ≈ {root:.5f}")
            print(f"\n👉 Biện luận điều kiện dừng tại bước n = {n}:")
            print(f"   - Công thức: Δ = (b_ban_dau - a_ban_dau) / 2^(n + 1)")
            print(f"   - Thay số:   Δ = ({b_ban_dau} - {a_ban_dau}) / 2^({n} + 1)")
            print(f"   - Kết quả:   Δ = {delta:.5f} <= {ep} (Thỏa mãn tiêu chuẩn sai số thực tế!)")
            break

        n += 1

    return root

# =====================================================================
# CHƯƠNG TRÌNH CHÍNH
# =====================================================================
def haunghiem():
    print("\n🌸 Nhập biểu thức f(x) (Cứ gõ dấu ^ cho phép mũ, VD: x^3 - x - 5 = 0)")
    fx_str = input("Nhập f(x) = ").strip()

    f = tao_ham(fx_str)

    # TRƯỚC KHI CHẠY: Kiểm tra lỗi cú pháp toán học ngay lập tức
    try:
        f(1.0)
    except (SyntaxError, NameError) as e:
        print(f'\n❌ LỖI CÚ PHÁP: Biểu thức hàm số cậu nhập chưa chuẩn Python!')
        print(f'   Chi tiết lỗi: {e}')
        print('💡 Gợi ý: Nhớ dùng dấu * cho phép nhân (VD: 2*x chứ không gõ 2x) và dùng toán học chuẩn nha.')
        return
    except Exception:
        pass

    # --- CHỌN CHẾ ĐỘ ---
    print("\n👉 CHỌN CHẾ ĐỘ TÌM NGHIỆM:")
    print("   [1] Tìm nghiệm thông thường (dương hoặc bất kỳ)")
    print("   [2] Tự nhập tay khoảng phân ly [a, b]")
    print("   [3] Tìm NGHIỆM ÂM lớn nhất / nhỏ nhất (đổi biến x -> -x)")
    lua_chon = input("Nhập lựa chọn của cậu (1, 2 hoặc 3): ").strip()

    # --- CHẾ ĐỘ 3: TÌM NGHIỆM ÂM BẰNG ĐỔI BIẾN ---
    if lua_chon == '3':
        print("\n" + "="*80)
        print("📐 PHƯƠNG PHÁP TÌM NGHIỆM ÂM BẰNG ĐỔI BIẾN x -> -x")
        print("="*80)
        print("   Lý thuyết:")
        print("   ─────────────────────────────────────────────────────────────")
        print("   Muốn tìm nghiệm âm của f(x) = 0, ta đặt x = -t (t > 0).")
        print("   Khi đó: f(x) = f(-t) = 0.")
        print("   Đặt g(t) = f(-t). Nghiệm t > 0 của g(t) = 0")
        print("   tương ứng với nghiệm âm x = -t của f(x) = 0.")
        print("")
        print("   Ví dụ: Nghiệm âm lớn nhất của f(x) = 0")
        print("          → Tìm nghiệm dương NHỎ NHẤT của g(t) = f(-t) = 0")
        print("          → Sau đó x = -t.")
        print("="*80)

        # Tạo hàm g(t) = f(-t)
        g = tao_ham_doi_bien_am(f)

        print("\n🔎 Đang quét tìm nghiệm dương của g(t) = f(-t) trong [0, 10]...")
        khoang_duong = quet_tat_ca_khoang_nghiem(g, start=0, end=10, step=0.1)

        if len(khoang_duong) > 0:
            print("=> Đã tìm thấy các khoảng phân ly nghiệm dương của g(t):")
            for i, (a_sug, b_sug) in enumerate(khoang_duong):
                print(f"   Khoảng {i+1}: t ∈ [{a_sug:.1f}, {b_sug:.1f}]  =>  x = -t ∈ [{-b_sug:.1f}, {-a_sug:.1f}]")
            print("\n💡 LỜI KHUYÊN:")
            print("   • Nghiệm âm LỚN NHẤT (gần 0 nhất): Chọn khoảng có |x| nhỏ nhất (gần 0)")
            print("   • Nghiệm âm NHỎ NHẤT (xa 0 nhất): Chọn khoảng có |x| lớn nhất")
        else:
            print("=> Không tìm thấy nghiệm dương nào của g(t) trong [0, 10].")
            print("   Có thể thử quét rộng hơn hoặc phương trình không có nghiệm âm.")

        print("\n" + "-"*60)
        print("NHẬP KHOẢNG CHO g(t) = f(-t) (tìm nghiệm DƯƠNG của g)")
        print("-"*60)
        a_t = nhap_so_thuc("Nhập điểm a (cho t, tìm nghiệm dương nên a ≥ 0): ")
        b_t = nhap_so_thuc("Nhập điểm b (cho t): ")
        ep = nhap_so_thuc("Nhập sai số Epsilon (VD: 1e-3 hoặc 0.001): ")

        if a_t > b_t:
            print("\n=> Đã tự động đảo lại đoạn [a, b] cho chuẩn hướng tăng dần nha!")
            a_t, b_t = b_t, a_t

        try:
            sign_a = sign(g(a_t))
            sign_b = sign(g(b_t))
        except Exception as e:
            print(f"\n❌ LỖI: Không tính được giá trị hàm số tại biên đầu mút: {e}")
            return

        if sign_a * sign_b > 0:
            print(f"\n❌ CẢNH BÁO: g({a_t}) và g({b_t}) cùng dấu!")
            tuy_chon = input("Cậu có muốn ép máy chạy tiếp không? (y/n): ").strip().lower()
            if tuy_chon != 'y':
                return

        # Chạy chia đôi trên g(t), sau đó đổi về x = -t
        t_nghiem = chia_doi_core(g, a_t, b_t, ep, sign_a, sign_b, nghiem_am_mode=True, ten_nghiem="x")

        print("\n" + "="*80)
        print("📐 PHẦN GIẢI THÍCH ĐỔI BIẾN (Chép vào bài)")
        print("="*80)
        print(f"   • Đặt x = -t, khi đó f(x) = f(-t) = 0.")
        print(f"   • Đặt g(t) = f(-t), tìm nghiệm dương t của g(t) = 0.")
        print(f"   • Tìm được t ≈ {t_nghiem:.5f} (nghiệm dương của g).")
        print(f"   • Suy ra nghiệm âm của f(x): x = -t ≈ {-t_nghiem:.5f}.")
        print("="*80)
        return

    # --- CHẾ ĐỘ 1 & 2: TÌM NGHIỆM THÔNG THƯỜNG ---
    # --- KHỞI ĐỘNG RADAR QUÉT NGHIỆM ---
    print("\n" + "🔍 "*15)
    print("ĐANG KHỞI ĐỘNG RADAR TÁCH NGHIỆM...")
    danh_sach_khoang = quet_tat_ca_khoang_nghiem(f)

    if len(danh_sach_khoang) > 0:
        print("=> Đã tìm thấy các khoảng phân ly nghiệm (đổi dấu) sau đây:")
        for i, (a_sug, b_sug) in enumerate(danh_sach_khoang):
            print(f"   Khoảng {i+1}: [{a_sug:.1f}, {b_sug:.1f}]")
        print("\n💡 LỜI KHUYÊN: Hãy đọc kỹ đề bài thi để chọn khoảng [a, b] phù hợp nhất bên dưới nhé!")
    else:
        print("=> Radar không tìm thấy nghiệm nào trong khoảng [-10, 10]. Cậu có thể chủ động nhập tay khoảng rộng hơn ở dưới!")
    print("🔍 "*15 + "\n")

    if lua_chon == '2':
        a_ban_dau = nhap_so_thuc("Nhập điểm a cậu chọn: ")
        b_ban_dau = nhap_so_thuc("Nhập điểm b cậu chọn: ")
    else:
        a_ban_dau = nhap_so_thuc("Nhập điểm a cậu chọn: ")
        b_ban_dau = nhap_so_thuc("Nhập điểm b cậu chọn: ")

    ep = nhap_so_thuc("Nhập sai số Epsilon (VD: 1e-3 hoặc 0.001): ")

    if a_ban_dau > b_ban_dau:
        print("\n=> Đã tự động đảo lại đoạn [a, b] cho chuẩn hướng tăng dần nha!")
        a_ban_dau, b_ban_dau = b_ban_dau, a_ban_dau

    a, b = a_ban_dau, b_ban_dau

    try:
        sign_a = sign(f(a))
        sign_b = sign(f(b))
    except Exception as e:
        print(f"\n❌ LỖI: Không tính được giá trị hàm số tại biên đầu mút: {e}")
        return

    if sign_a * sign_b > 0:
        print(f"\n❌ CẢNH BÁO: f({a}) và f({b}) cùng dấu! Khoảng này không đảm bảo chứa nghiệm định lý Bolzano-Cauchy.")
        tuy_chon = input("Cậu có muốn ép máy chạy tiếp không? (y/n): ").strip().lower()
        if tuy_chon != 'y':
            return

    # Chạy chia đôi thông thường
    chia_doi_core(f, a_ban_dau, b_ban_dau, ep, sign_a, sign_b, nghiem_am_mode=False, ten_nghiem="x")

if __name__ == "__main__":
    print("✨ TOOL AUTO-PILOT PHƯƠNG PHÁP CHIA ĐÔI (CHUẨN TỰ LUẬN) ✨")
    print("   ✨ BẢN MỚI: Thêm chế độ tìm nghiệm âm bằng đổi biến x -> -x ✨")
    haunghiem()
