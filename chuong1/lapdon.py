import math

def lap_don_tu_dong_hoan_hao():
    print('=== PHÂN TÍCH TỰ LUẬN: PHƯƠNG PHÁP LẶP ĐƠN (TỰ ĐỘNG TÌM KHOẢNG) ===')

    # 1. NHẬP HÀM LẶP VÀ SAI SỐ
    print('💡 CÚ PHÁP: Nhập vế phải phi(x). Dùng * để nhân, ** để mũ, math.cos(x)...')
    print('   Ví dụ: (x + 1)**(1/3)  hoặc  x = (x + 1)**(1/3)')
    chuoi_phix = input('Nhập hàm lặp phi(x): ').strip()
    ep = float(input('Nhập sai số epsilon: '))

    # Tự động xử lý nếu người dùng gõ cả cụm "x = ..." hoặc "phi(x) = ..."
    if '=' in chuoi_phix:
        cac_phan = chuoi_phix.split('=')
        if cac_phan[0].strip().lower() in ['x', 'phi(x)', 'phi', 'φ(x)']:
            chuoi_phix = cac_phan[1].strip()
        else:
            chuoi_phix = cac_phan[0].strip()

    boi_canh = {
        'math': math, 'sin': math.sin, 'cos': math.cos, 
        'tan': math.tan, 'exp': math.exp, 'log': math.log, 
        'sqrt': math.sqrt, 'pi': math.pi
    }

    def phi(x_val):
        boi_canh['x'] = float(x_val)
        return eval(chuoi_phix, boi_canh)

    def f_goc(x_val):
        # Phương trình gốc f(x) = phi(x) - x = 0
        return phi(x_val) - x_val

    def phi_phay_approx(x_val, h=1e-5):
        return (phi(x_val + h) - phi(x_val - h)) / (2 * h)

    # Kiểm tra thử hàm số xem có lỗi cú pháp không
    try:
        phi(1.0)
    except Exception as e:
        print(f'\n❌ LỖI CÚ PHÁP: Biểu thức hàm số phi(x) chưa đúng chuẩn Python! (Chi tiết: {e})')
        return

    # 2. LỰA CHỌN PHƯƠNG THỨC XÁC ĐỊNH KHOẢNG PHÂN LY
    print('\n👉 CHỌN CHẾ ĐỘ XÁC ĐỊNH KHOẢNG PHÂN LY NGHIỆM:')
    print('   [1] Máy tự động quét khoảng thích hợp từ [-100, 100]')
    print('   [2] Tự nhập tay khoảng phân ly [a, b] theo đề bài')
    lua_chon = input('Nhập lựa chọn của cậu (1 hoặc 2): ').strip()

    if lua_chon == '1':
        print('\n🔎 Đang tự động quét khoảng phân ly nghiệm dựa trên f(x) = phi(x) - x = 0...')
        khoang_tim_duoc = []
        for i in range(-100, 100):
            try:
                # Tìm nơi hàm f(x) đổi dấu
                if f_goc(i) * f_goc(i + 1) < 0:
                    khoang_tim_duoc.append((float(i), float(i + 1)))
            except:
                continue

        if not khoang_tim_duoc:
            print('❌ Máy không tự tìm thấy khoảng phân ly nào trong phạm vi [-100, 100].')
            print('💡 Hãy chuyển sang Option 2 để tự nhập khoảng thủ công nhé.')
            return

        a, b = khoang_tim_duoc[0]
        x0 = a  # Mặc định lấy luôn đầu mút làm điểm khởi đầu
        print(f'✅ Đã tìm thấy khoảng phân ly thích hợp: [{a:.0f}, {b:.0f}]')
        print(f'👉 Chọn luôn điểm bắt đầu lặp x0 = {x0:.1f}')
    else:
        a = float(input("Nhập đầu mút a: "))
        b = float(input("Nhập đầu mút b: "))
        x0 = float(input("Nhập điểm khởi đầu x0: "))

    # 3. BIỆN LUẬN ĐIỀU KIỆN HỘI TỤ (TÌM HỆ SỐ CO Q CHÍNH XÁC)
    print(f"\n👉 Biện luận điều kiện hội tụ trên đoạn [{a:.2f}, {b:.2f}]:")
    print(f"   - Hàm lặp tương ứng: φ(x) = {chuoi_phix}")

    # Chia nhỏ đoạn thành 100 điểm quét tìm max |phi'(x)| tránh bỏ sót
    q = 0.0
    steps = 100
    for i in range(steps + 1):
        diem_quet = a + i * (b - a) / steps
        try:
            q = max(q, abs(phi_phay_approx(diem_quet)))
        except:
            continue

    print(f"   - Hệ số co q = max|φ'(x)| trên [{a:.2f}, {b:.2f}] xấp xỉ: {q:.5f}")

    if q >= 1:
        print(f"❌ KẾT LUẬN: Hệ số co q = {q:.5f} >= 1. Hàm lặp không đảm bảo hội tụ co!")
        return
    else:
        print(f"✅ KẾT LUẬN (Chép vào bài): Vì q = {q:.5f} < 1, hàm lặp thỏa mãn điều kiện co.")
        print("   -> Dãy lặp chắc chắn hội tụ về nghiệm duy nhất trên đoạn đã cho.")

    # 3.5. TÍNH SAI SỐ TIÊN NGHIỆM (THÊM MỚI)
    print("\n" + "="*80)
    print("📐 PHẦN ĐÁNH GIÁ SAI SỐ TIÊN NGHIỆM (Tính trước khi lặp)")
    print("="*80)
    print("   Công thức tiên nghiệm:")
    print("   ─────────────────────────────────────────────────────────────")
    print("        |xₙ - x*| ≤ qⁿ/(1-q) · |x₁ - x₀|")
    print("")
    print("   Trong đó:")
    print(f"     • q = {q:.5f} (hệ số co đã tính ở trên)")
    print(f"     • x₀ = {x0:.5f} (điểm khởi tạo)")
    print(f"     • |x₁ - x₀| cần tính sau bước lặp đầu tiên")
    print("")
    print("   Ý nghĩa: Tiên nghiệm cho phép ước lượng sai số TRƯỚC KHI lặp,")
    print("            dựa trên số bước n và hệ số co q.")
    print("="*80)

    # 4. IN BẢNG TRÌNH BÀY CHI TIẾT (CÔNG THỨC = THAY SỐ)
    print('\n' + '='*125)
    print(f"{'n':<3} | {'Công thức & Biểu thức thay số tường minh':<58} | {'Kết quả x_n':<14} | {'Sai số |x_n - x_{n-1}|':<20}")
    print('-' * 125)

    x_cu = x0
    x1_minus_x0 = None  # Lưu |x₁ - x₀| để tính tiên nghiệm

    for k in range(1, 101):
        try:
            x_moi = phi(x_cu)
            delta = abs(x_moi - x_cu)

            # Lưu |x₁ - x₀| tại bước k=1
            if k == 1:
                x1_minus_x0 = delta

            # Thay thế biến x thành con số thực tế bước trước để tạo chuỗi "thay số"
            chuoi_thay_so = chuoi_phix.replace('x', f"({x_cu:.5f})")
            cong_thuc_thay_so = f"x_{k} = φ(x_{k-1}) = {chuoi_thay_so}"

            if len(cong_thuc_thay_so) > 56:
                cong_thuc_thay_so = cong_thuc_thay_so[:53] + "..."

            print(f"{k:<3} | {cong_thuc_thay_so:<58} | {x_moi:<14.5f} | {delta:<20.5f}")

            # Tính sai số hậu nghiệm chuẩn lý thuyết phương pháp co
            sai_so_chat = (q / (1 - q)) * delta

            if delta <= ep:
                print('-' * 125)
                print(f'\n👉 Đánh giá sai số và điều kiện dừng ở bước n = {k}:')
                print(f'   - Hiệu hai bước lặp liên tiếp: |x_{k} - x_{k-1}| = {delta:.5f}')
                print(f'   - Sai số hậu nghiệm đầy đủ:    (q/(1-q)) * |x_{k} - x_{k-1}| = ({q:.4f}/(1-{q:.4f})) * {delta:.5f} = {sai_so_chat:.5f}')

                # THÊM: In sai số tiên nghiệm tại bước dừng
                if x1_minus_x0 is not None and x1_minus_x0 > 0:
                    tien_nghiem_k = (q**k / (1 - q)) * x1_minus_x0
                    print(f'\n   - Sai số TIÊN NGHIỆM tại bước n={k}:')
                    print(f'        |x_{k} - x*| ≤ q^{k}/(1-q) · |x₁ - x₀|')
                    print(f'                     ≤ ({q:.4f})^{k}/(1-{q:.4f}) · {x1_minus_x0:.5f}')
                    print(f'                     ≤ {q**k:.6f}/{1-q:.4f} · {x1_minus_x0:.5f}')
                    print(f'                     ≤ {tien_nghiem_k:.8f}')

                print(f'   - Kết luận điều kiện dừng:    {delta:.5f} <= {ep} (Đạt chỉ tiêu đề bài!)')
                print(f'\n✅ KẾT LUẬN CUỐI CÙNG: Nghiệm gần đúng của phương trình là x ≈ {x_moi:.5f}')
                break

            x_cu = x_moi
        except Exception as e:
            print(f'\n❌ LỖI phát sinh tại bước {k}: {e}')
            break

if __name__ == '__main__':
    lap_don_tu_dong_hoan_hao()
