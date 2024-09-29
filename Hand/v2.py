import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from classify_hand import classify_hand, predict_hand_prob

# 初始化MediaPipe手势模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

# 定义screen_height函数
def screen_height(image, desired_height):
    aspect_ratio = image.shape[1] / image.shape[0]
    width = int(desired_height * aspect_ratio)
    resized_image = cv2.resize(image, (width, desired_height))
    return resized_image, width

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

# 绘制概率条形图（横向，显示在左侧）
def draw_probabilities(image, probabilities, screen_height):
    bar_height = int(screen_height / 10)
    bar_width = int(0.15 * screen_height)
    
    # Create a blank area on the left side
    bars_image = np.zeros((screen_height, bar_width, 3), dtype=np.uint8)

    # Draw horizontal bars for each class probability
    for i, prob in enumerate(probabilities):
        start_point = (0, i * bar_height)
        end_point = (int(prob * bar_width), (i + 1) * bar_height - 2)
        cv2.rectangle(bars_image, start_point, end_point, (0, 255, 0), -1)

        # Add class labels (0-9) on the bars
        cv2.putText(bars_image, f'{i}', (5, (i + 1) * bar_height - bar_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Concatenate the bars image with the original image
    image_with_bars = np.hstack((bars_image, image))
    return image_with_bars

# 主程序
def main():
    cap = cv2.VideoCapture(0)
    desired_screen_height = 600
    csv_file = 'hand_record_data/hand_data.csv'
    img_dir = 'hand_record_data/imgs'
    os.makedirs(img_dir, exist_ok=True)

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
        probabilities = None

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

            # 获取手势分类概率
            probabilities = predict_hand_prob(normalized_coordinates)

            # 绘制关键点及分类结果
            draw_landmarks(image, max_hand, handedness, classification_result)

        # 调整图像大小
        resized_image, new_width = screen_height(image, desired_screen_height)

        # 如果有分类概率，显示在左侧
        if probabilities is not None:
            resized_image = draw_probabilities(resized_image, probabilities, desired_screen_height)
        else:
            zero_probabilities = [0.1 for _ in range(10)]
            resized_image = draw_probabilities(resized_image, zero_probabilities, desired_screen_height)

        # 显示图像
        cv2.imshow('MediaPipe Hands with Classification', resized_image)

        # 按键处理
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key != -1 and max_hand is not None:
            # 记录数据
            data = [chr(key)] + landmarks
            df = pd.DataFrame([data])
            df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

            # 打印df的行数
            rows = sum(1 for line in open(csv_file))
            print(f"Saved with { rows } rows")

            # 保存图片
            timestamp = int(time.time())
            img_name = f"{chr(key)}_{timestamp}.jpg"
            img_path = os.path.join(img_dir, img_name)
            cv2.imwrite(img_path, original_image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()