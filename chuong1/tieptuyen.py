import math

def tiep_tuyen_tu_luan_chuan_muc():
    print('=== PHÂN TÍCH TỰ LUẬN: PHƯƠNG PHÁP TIẾP TUYẾN (NEWTON) ===')
    
    # 1. NHẬP HÀM SỐ VÀ SAI SỐ
    print('💡 CÚ PHÁP: Dùng * để nhân, ** để mũ, math.exp(x) để tính e^x, math.cos(x) cho lượng giác...')
    chuoi_fx = input('Nhập hàm f(x): ').strip()
    ep = float(input('Nhập sai số epsilon: '))
    
    # Tự động cắt bỏ đuôi "=0" nếu người dùng nhập toàn bộ phương trình
    if '=' in chuoi_fx:
        chuoi_fx = chuoi_fx.split('=')[0].strip()
    
    # Môi trường chứa các hàm toán học để eval đọc được
    boi_canh = {
        'math': math, 'sin': math.sin, 'cos': math.cos, 
        'tan': math.tan, 'exp': math.exp, 'log': math.log, 
        'sqrt': math.sqrt, 'pi': math.pi
    }
    
    # Hàm bọc eval để tính giá trị f(x) công thức
    def f(x_val):
        boi_canh['x'] = float(x_val)
        return eval(chuoi_fx, boi_canh)
        
    def f1_approx(x_val, h=1e-5):
        return (f(x_val + h) - f(x_val - h)) / (2 * h)
        
    def f2_approx(x_val, h=1e-5):
        return (f(x_val + h) - 2 * f(x_val) + f(x_val - h)) / (h ** 2)

    # TRƯỚC KHI CHẠY: Kiểm tra xem hàm nhập vào có lỗi cú pháp Python không
    try:
        f(1.0)
    except (SyntaxError, NameError) as e:
        print(f'\n❌ LỖI CÚ PHÁP: Biểu thức hàm số cậu nhập chưa chuẩn Python!')
        print(f'   Chi tiết: {e}')
        print('💡 Gợi ý: Hãy chắc chắn cậu dùng ** để mũ (ví dụ x**5) và có dấu * khi nhân (ví dụ 4*math.cos(x)).')
        return
    except Exception:
        # Các lỗi toán học khác như chia cho 0 tại x=1 tạm thời bỏ qua để vào vòng quét rộng hơn
        pass

    # 2. XÁC ĐỊNH KHOẢNG PHÂN LY NGHIỆM [a, b]
    print('\n👉 CHỌN CHẾ ĐỘ XÁC ĐỊNH KHOẢNG PHÂN LY NGHIỆM:')
    print('   [1] Máy tự động quét khoảng thích hợp từ [-100, 100]')
    print('   [2] Tự nhập tay khoảng phân ly [a, b] theo đề bài')
    lua_chon = input('Nhập lựa chọn của cậu (1 hoặc 2): ').strip()
    
    if lua_chon == '1':
        print('\n🔎 Đang quét tìm khoảng phân ly nghiệm thích hợp...')
        khoang_tim_duoc = []
        for i in range(-100, 100):
            try:
                if f(i) * f(i + 1) < 0:
                    khoang_tim_duoc.append((float(i), float(i + 1)))
            except:
                continue

        if not khoang_tim_duoc:
            print('❌ Không tìm thấy khoảng phân ly nghiệm nào trong phạm vi [-100, 100].')
            print('💡 Gợi ý: Hãy kiểm tra lại xem hàm số có nghiệm thực trong khoảng này không nhé.')
            return
            
        a, b = khoang_tim_duoc[0]
        print(f'✅ Đã tìm thấy khoảng phân ly: [{a:.0f}, {b:.0f}]')
    else:
        a = float(input("Nhập đầu mút a: "))
        b = float(input("Nhập đầu mút b: "))
        print(f'✅ Đã ghi nhận khoảng phân ly: [{a:.2f}, {b:.2f}]')

    # 3. BIỆN LUẬN ĐIỀU KIỆN FOURIER TÌM x0 (TRÌNH BÀY GIẤY THI)
    print(f'\n👉 Biện luận điều kiện hội tụ Fourier trên đoạn [{a:.2f}, {b:.2f}]:')
    print('   - Công thức lý thuyết: Chọn x0 sao cho f(x0) * f\'\'(x0) > 0')
    
    try:
        f_a, f2_a = f(a), f2_approx(a)
        tich_a = f_a * f2_a
        
        f_b, f2_b = f(b), f2_approx(b)
        tich_b = f_b * f2_b
        
        print(f'   - Xét đầu mút a = {a:.2f}: f({a:.2f}) = {f_a:.5f}, f\'\'({a:.2f}) ≈ {f2_a:.5f}  => Tích = {tich_a:.5f}')
        print(f'   - Xét đầu mút b = {b:.2f}: f({b:.2f}) = {f_b:.5f}, f\'\'({b:.2f}) ≈ {f2_b:.5f}  => Tích = {tich_b:.5f}')
        
        if tich_a > 0:
            x0 = a
            print(f'\n✅ KẾT LUẬN (Chép vào bài): Vì f({a:.2f})·f\'\'({a:.2f}) > 0 nên chọn điểm khởi đầu x0 = {x0:.5f}')
        elif tich_b > 0:
            x0 = b
            print(f'\n✅ KẾT LUẬN (Chép vào bài): Vì f({b:.2f})·f\'\'({b:.2f}) > 0 nên chọn điểm khởi đầu x0 = {x0:.5f}')
        else:
            # Nếu cả 2 đầu mút không thỏa mãn, chọn x0 là điểm trung điểm hoặc mút có tích lớn hơn gần 0 để chạy tiếp
            x0 = b if abs(tich_b) > abs(tich_a) else a
            print(f'\n⚠️ CẢNH BÁO: Điều kiện Fourier không tuyệt đối ở mút, tạm chọn x0 = {x0:.5f} để tính toán.')
    except Exception as e:
        print(f'❌ Lỗi khi tính đạo hàm biện luận: {e}')
        return

    # 4. IN BẢNG TRÌNH BÀY CHI TIẾT CÔNG THỨC THAY SỐ (Y NHƯ GIẤY THI)
    print('\n' + '='*115)
    print(f"{'n':<3} | {'Các giá trị thành phần':<35} | {'Công thức thay số chi tiết':<45} | {'Kết quả x_n':<12}")
    print('-' * 115)
    
    x_cu = x0
    for k in range(1, 101):
        try:
            fx = f(x_cu)
            f1x = f1_approx(x_cu)
            
            if abs(f1x) < 1e-12:
                print('\n❌ LỖI: Đạo hàm f\'(x) quá gần 0. Tiếp tuyến bị song song trục hoành!')
                break
                
            x_moi = x_cu - fx / f1x
            delta = abs(x_moi - x_cu)
            
            # Khớp nội dung thành phần và phân số thay số để chép tay
            thanh_phan = f"f(x)={fx:.5f}, f'(x)={f1x:.5f}"
            thay_so = f"{x_cu:.5f} - ({fx:.5f}) / ({f1x:.5f})"
            
            print(f"{k:<3} | {thanh_phan:<35} | {thay_so:<45} | {x_moi:<12.5f}")
            
            # Đánh giá điều kiện dừng
            if delta <= ep:
                print('-' * 115)
                print(f'\n👉 Đánh giá sai số và điều kiện dừng ở bước n = {k}:')
                print(f'   - Công thức hậu nghiệm: |x_{k} - x_{k-1}| <= epsilon')
                print(f'   - Thay số kiểm tra:     |{x_moi:.5f} - {x_cu:.5f}| = {delta:.5f}')
                print(f'   - Kết luận:             {delta:.5f} <= {ep} (Thỏa mãn yêu cầu!)')
                print(f'\n✅ KẾT LUẬN CUỐI CÙNG: Nghiệm gần đúng của phương trình là x ≈ {x_moi:.5f}')
                break
                
            x_cu = x_moi
        except Exception as e:
            print(f'\n❌ LỖI phát sinh tại bước {k}: {e}')
            break

if __name__ == '__main__':
    tiep_tuyen_tu_luan_chuan_muc()