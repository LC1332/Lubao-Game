import pygame

# 初始化pygame
pygame.init()

# 获取当前屏幕的分辨率
SCREEN_INFO = pygame.display.Info()
SCREEN_WIDTH = 800 # SCREEN_INFO.current_w
SCREEN_HEIGHT = 600 # SCREEN_INFO.current_h

# 颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 设置全屏模式
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# 速度
carScale = 7
speed = SCREEN_HEIGHT / 10
car_width = SCREEN_HEIGHT // carScale
