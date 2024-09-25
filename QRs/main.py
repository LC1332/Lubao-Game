import cv2
from pyzbar.pyzbar import decode

# 存储已打印过的二维码内容
printed_codes = set()

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    while True:
        # 读取摄像头画面
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 解码二维码
        codes = decode(frame)
        
        for code in codes:
            # 将二维码内容转为字符串
            data = code.data.decode('utf-8')
            
            # 如果二维码内容未打印过，则打印并添加到已打印的集合中
            if data not in printed_codes:
                print(f"QR Code content: {data}")
                printed_codes.add(data)
        
        # 展示摄像头画面
        cv2.imshow("QR Code Scanner", frame)
        
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
