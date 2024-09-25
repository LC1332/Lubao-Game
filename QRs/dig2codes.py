import cv2
import qrcode
import numpy as np
from typing import List, Tuple

# number2codes 函数
def number2codes(s1: str, s2: str) -> str:
    return f"CN{s1},{s2.zfill(12)},0000,"

# 生成二维码并循环展示
def display_qr_codes(data_list: List[Tuple[str, str]]):
    qr_data_list = [number2codes(s1, s2) for s1, s2 in data_list]
    idx = 0

    while True:
        qr_data = qr_data_list[idx]

        # 生成二维码
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)

        # 转换成图像
        img = qr.make_image(fill='black', back_color='white').convert('RGB')
        img_array = np.array(img)

        # 使用 opencv 展示二维码
        cv2.imshow(f"QR Code: {qr_data}", img_array)

        # 按下 'q' 键退出，按下空格键切换到下一个二维码
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 空格切换下一个
            idx = (idx + 1) % len(qr_data_list)

    # 释放所有窗口
    cv2.destroyAllWindows()

# 主函数
if __name__ == "__main__":
    # 初始化一个list of tuple
    data_list = [("15", "14001304"), ("06", "14008136"), ("00", "14003046")]

    # 转换并展示二维码
    display_qr_codes(data_list)
