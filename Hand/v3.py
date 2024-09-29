
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from classify_hand import classify_hand, predict_hand_prob
from visualize import draw_landmarks, draw_probabilities
from CalcuManager import CalcuManager

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


# 主程序
def main():
    cap = cv2.VideoCapture(0)
    desired_screen_height = 600
    csv_file = 'hand_record_data/hand_data.csv'
    img_dir = 'hand_record_data/imgs'
    os.makedirs(img_dir, exist_ok=True)

    calcu_manager = CalcuManager()

    current_digit = -1
    new_digit = -1
    new_digit_time = None

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

            # 判断当前手势是否稳定
            if classification_result == current_digit:
                new_digit = -1  # 已经稳定在 current_digit
            else:
                if new_digit == -1:
                    new_digit = classification_result
                    new_digit_time = time.time()
                elif new_digit == classification_result:
                    if time.time() - new_digit_time > 2:  # 稳定时间超过2秒
                        current_digit = new_digit
                        new_digit = -1
                        print(f"手势变为 {current_digit}")
                        eqn = calcu_manager.sample_eqn_with_ans(current_digit)
                        # return a dict with {'question': '2-2', 'answer': 0, 'score': 20, 'inputs': [2, 2]}
                        # TODO: display eqn at right of screen
                else:
                    new_digit = classification_result
                    new_digit_time = time.time()

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
