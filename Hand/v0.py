
import cv2
import mediapipe as mp
import numpy as np
from classify_hand import classify_hand

# 初始化MediaPipe手势模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

# 用于绘制关键点和分类结果的函数
def draw_landmarks(image, hand_landmarks, handedness, classification_result=None):
    # 绘制关键点
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    
    # 在图像上显示手势分类结果
    if classification_result is not None:
        cv2.putText(image, f'Gesture: {classification_result}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# 主程序
def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # 左右镜像翻转
        image = cv2.flip(image, 1)
        original_image = image.copy()

        # 处理图像
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        max_hand = None
        classification_result = None

        # 如果检测到手势
        if results.multi_hand_landmarks:
            # 找到最大的手
            max_hand = max(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
            handedness = results.multi_handedness[results.multi_hand_landmarks.index(max_hand)].classification[0].label

            # 提取关键点并归一化坐标
            landmarks = [landmark.x for landmark in max_hand.landmark] +                         [landmark.y for landmark in max_hand.landmark] +                         [landmark.z for landmark in max_hand.landmark]

            # 将x, y, z归一化处理
            x_coords = landmarks[:21]
            y_coords = landmarks[21:42]
            z_coords = landmarks[42:]

            x_centered = [x - np.mean(x_coords) for x in x_coords]
            y_centered = [y - np.mean(y_coords) for y in y_coords]
            z_centered = [z - np.mean(z_coords) for z in z_coords]

            # 拼接成归一化后的63维坐标
            normalized_coordinates = x_centered + y_centered + z_centered

            # 使用 classify_hand 进行手势分类
            classification_result = classify_hand(normalized_coordinates)

            # 绘制关键点及分类结果
            draw_landmarks(image, max_hand, handedness, classification_result)

        # 显示图像
        cv2.imshow('MediaPipe Hands with Classification', image)

        # 按键处理
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
