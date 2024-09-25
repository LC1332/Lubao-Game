import pygame
import os
import random

car_imgs_folder = "./cars"

pygame.mixer.init()
up_sound = pygame.mixer.Sound("./audio/up.wav")
left_sound = pygame.mixer.Sound("./audio/left.wav")
down_sound = pygame.mixer.Sound("./audio/down.wav")
right_sound = pygame.mixer.Sound("./audio/right.wav")

# 初始化pygame
pygame.init()

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 设置窗口大小
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
carScale = 5
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# 设置速度
speed = SCREEN_HEIGHT/10

def reinit_car():
    car_img_file = random.choice(os.listdir(car_imgs_folder))
    car_img_path = os.path.join(car_imgs_folder, car_img_file)
    car_image = pygame.image.load(car_img_path).convert_alpha()
    car_image = pygame.transform.scale(car_image, (SCREEN_HEIGHT // carScale, SCREEN_HEIGHT // carScale))
    return car_image

# 加载图片
car_image = reinit_car()

# 获取图片位置
car_rect = car_image.get_rect()
car_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)




import time

key_sound_time = 0

up_button = ['2', '3', '4', '5', '6', '7', '8', '9', '0', \
'-', '=', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'F1', 'F2', 'F3',\
 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12']
 
left_button = ['tab', '`', 'q', 'w', 'a', 's', 'd', 'f', 'g', \
'Caps_Lock', 'left-shift', 'z', 'x', 'c', 'left-ctrl', 'KP_Left']

down_button = ['v', 'b', 'n', 'm', ',', '.', 'space', 'KP_Down','，','。']

right_button = ['、','/','j', 'k', 'l', '；',';','’', "'", 'return', '【','[',\
'】', ']','p', '\\', 'right-shift', 'right-ctrl', 'KP_Right']


move_cooldown_time = 0
cooldown_time = 0.3
up_direction = False
left_direction = False
right_direction = False
down_direction = False

done = False
while not done:
    # 处理游戏事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            # done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                # done = True
            elif event.unicode in up_button:
                up_direction = True
            elif event.key == pygame.K_UP:
                up_direction = True
            elif event.unicode in left_button:
                left_direction = True
            elif event.key == pygame.K_LEFT or event.key == 9:
                left_direction = True
            elif event.unicode in down_button:
                down_direction = True
            elif event.key == pygame.K_DOWN:
                down_direction = True
            elif event.unicode in right_button or event.key == 13:
                right_direction = True
            elif event.key == pygame.K_RIGHT or event.key == 127\
             or event.key ==8:
                right_direction = True
            elif event.key == 32:
                down_direction = True
            else:
                print(event.unicode)
                print(event.key)
                
    if time.time() - move_cooldown_time > cooldown_time:
        update_flag = True
        if up_direction:
            car_rect.move_ip(0, -speed)
            up_sound.play()
        elif left_direction:
            car_rect.move_ip(-speed, 0)
            left_sound.play()
        elif down_direction:
            car_rect.move_ip(0, speed)
            down_sound.play()
        elif right_direction:
            right_sound.play()
            car_rect.move_ip(speed, 0)
        else:
            update_flag = False

        if update_flag:
            move_cooldown_time = time.time()

    up_direction = False
    left_direction = False
    right_direction = False
    down_direction = False

    if car_rect.left < 0 or car_rect.right > SCREEN_WIDTH or car_rect.top < 0 or car_rect.bottom > SCREEN_HEIGHT:
        # 加载图片
        car_image = reinit_car()

        # 获取图片位置
        car_rect = car_image.get_rect()
        car_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

    # 填充屏幕
    screen.fill(BLACK)

    # 绘制图片
    screen.blit(car_image, car_rect)

    # 更新屏幕
    pygame.display.flip()

# 退出pygame
pygame.quit()
