import math

def day_cung_tu_luan_linh_hoat():
    print('=== PHÂN TÍCH TỰ LUẬN: PHƯƠNG PHÁP DÂY CUNG ===')
    
    # 1. NHẬP HÀM SỐ VÀ SAI SỐ
    print('💡 CÚ PHÁP: Dùng * để nhân, ** để mũ, math.exp(x) để tính e^x, math.cos(x) cho lượng giác...')
    chuoi_fx = input('Nhập hàm f(x): ').strip()
    ep = float(input('Nhập sai số epsilon: '))
    
    # Tự động cắt bỏ đuôi "=0" nếu người dùng gõ cả phương trình
    if '=' in chuoi_fx:
        chuoi_fx = chuoi_fx.split('=')[0].strip()
    
    # Môi trường tính toán cho hàm eval
    boi_canh = {
        'math': math, 'sin': math.sin, 'cos': math.cos, 
        'tan': math.tan, 'exp': math.exp, 'log': math.log, 
        'sqrt': math.sqrt, 'pi': math.pi
    }
    
    def f(x_val):
        boi_canh['x'] = float(x_val)
        return eval(chuoi_fx, boi_canh)
    
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
        print(f'✅ Đã tìm thấy khoảng phân ly nghiệm: [{a:.0f}, {b:.0f}]')
    else:
        a = float(input("Nhập đầu mút a: "))
        b = float(input("Nhập đầu mút b: "))
        print(f'✅ Đã ghi nhận khoảng phân ly: [{a:.2f}, {b:.2f}]')

    # 3. TỰ ĐỘNG BIỆN LUẬN ĐIỀU KIỆN FOURIER ĐỂ CHỌN ĐIỂM CỐ ĐỊNH
    print(f'\n👉 Biện luận điều kiện Fourier trên đoạn [{a:.2f}, {b:.2f}]:')
    print('   - Nguyên tắc: Điểm cố định d thỏa mãn f(d) * f\'\'(d) > 0')
    
    try:
        f_a, f2_a = f(a), f2_approx(a)
        tich_a = f_a * f2_a
        
        f_b, f2_b = f(b), f2_approx(b)
        tich_b = f_b * f2_b
        
        print(f'   - Tại a = {a:.2f}: f(a) = {f_a:.5f}, f\'\'(a) ≈ {f2_a:.5f} => Tích = {tich_a:.5f}')
        print(f'   - Tại b = {b:.2f}: f(b) = {f_b:.5f}, f\'\'(b) ≈ {f2_b:.5f} => Tích = {tich_b:.5f}')
        
        if tich_a > 0:
            d, x0 = a, b
            print(f'\n✅ KẾT LUẬN: Chọn điểm cố định d = a = {d:.5f}, điểm bắt đầu lặp x0 = b = {x0:.5f}')
        elif tich_b > 0:
            d, x0 = b, a
            print(f'\n✅ KẾT LUẬN: Chọn điểm cố định d = b = {d:.5f}, điểm bắt đầu lặp x0 = a = {x0:.5f}')
        else:
            # Fallback nếu điều kiện Fourier không lý tưởng ở biên
            d, x0 = a, b
            print('\n⚠️ CẢNH BÁO: Khoảng biên không thỏa mãn Fourier tuyệt đối, mặc định chọn d = a để chạy tính toán.')
    except Exception as e:
        print(f'❌ Lỗi khi tính đạo hàm biện luận Fourier: {e}')
        return

    # 4. IN BẢNG SỐ LIỆU RÚT GỌN (THEO PHƯƠNG PHÁP KHÔNG ĐỔI ĐẦU MÚT)
    print('\n' + '='*75)
    print(f"{'n':<3} | {'x_{n-1}':<12} | {'f(x_{n-1})':<15} | {'x_n':<12} | {'|x_n - x_{n-1}|':<15}")
    print('-' * 75)
    
    f_d = f(d) 
    x_cu = x0
    
    for k in range(1, 101):
        try:
            fx_cu = f(x_cu)
            mau = fx_cu - f_d
            
            if abs(mau) < 1e-12:
                print('❌ LỖI: Mẫu số bằng 0 (Đường dây cung song song hoặc trùng trục hoành)!')
                break
                
            x_moi = x_cu - fx_cu * (x_cu - d) / mau
            delta = abs(x_moi - x_cu)
            
            # In dòng dữ liệu thuần số gọn gàng như cậu mong muốn
            print(f"{k:<3} | {x_cu:<12.5f} | {fx_cu:<15.5f} | {x_moi:<12.5f} | {delta:<15.5f}")
            
            if delta <= ep:
                print('-' * 75)
                print(f'\n✅ KẾT LUẬN CUỐI CÙNG: Nghiệm gần đúng x ≈ {x_moi:.5f}')
                print(f'   (Thỏa mãn điều kiện dừng do hiệu hai bước lặp {delta:.5f} <= {ep})')
                break
                
            x_cu = x_moi
        except Exception as e:
            print(f'\n❌ LỖI phát sinh tại bước {k}: {e}')
            break

if __name__ == '__main__':
    day_cung_tu_luan_linh_hoat()