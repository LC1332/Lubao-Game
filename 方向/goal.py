import pygame
import random
from settings import SCREEN_WIDTH, SCREEN_HEIGHT

goal_rect = pygame.Rect(100, 100, 100, 100)
RED = (255, 0, 0)

def reinit_box(car_rect, score, if_increase=False):
    global RED
    if if_increase:
        score += 1

    # 初始化goal的位置
    goal_rect.x = random.randint(0, SCREEN_WIDTH - 100)
    goal_rect.y = random.randint(0, SCREEN_HEIGHT - 100)

    global RED
    RED = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    if sum(RED) < 100:
        RED = (random.randint(0, 225)+30, random.randint(0, 225)+30, random.randint(0, 225)+30)

    count = 0
    while count < 5 and (car_rect.colliderect(goal_rect) or
                         abs(car_rect.x - goal_rect.x) < SCREEN_HEIGHT // 4 and 
                         abs(car_rect.y - goal_rect.y) < SCREEN_HEIGHT // 4):
        goal_rect.x = random.randint(0, SCREEN_WIDTH - 100)
        goal_rect.y = random.randint(0, SCREEN_HEIGHT - 100)
        count += 1

    return score
