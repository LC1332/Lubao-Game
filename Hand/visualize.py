
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
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