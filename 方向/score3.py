import pygame
import time
from settings import screen, BLACK, SCREEN_HEIGHT
from sounds import load_sounds
from car import reinit_car
from goal import reinit_box, goal_rect, RED
from controls import handle_input
from utils import display_score
from wall import Wall  # 导入Wall类
import pygame
import random
from settings import SCREEN_WIDTH, SCREEN_HEIGHT
import os
import random
import pygame
from settings import SCREEN_HEIGHT, carScale, car_width

car_imgs_folder = "./cars"


def reinit_car():
    global car_image
    car_img_file = random.choice(os.listdir(car_imgs_folder))
    car_img_path = os.path.join(car_imgs_folder, car_img_file)
    car_image = pygame.image.load(car_img_path).convert_alpha()
    car_image = pygame.transform.scale(car_image, (car_width, car_width))
    return car_image

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

def main():

    # 设置速度
    speed = SCREEN_HEIGHT / 10

    score = 0
    sounds = load_sounds()
    car_image = reinit_car()
    car_rect = car_image.get_rect()
    car_rect.center = (screen.get_width() // 2, screen.get_height() // 2)

    directions = None

    running = True
    move_cooldown_time = 0
    cooldown_time = 0.3

    # 初次生成Wall
    wall = Wall(car_rect, goal_rect)

    # 记录汽车的上一次位置
    previous_position = car_rect.topleft

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            direction = handle_input(event, directions)
            if direction:
                if time.time() - move_cooldown_time > cooldown_time:
                    # 记录汽车的上一次位置
                    previous_position = car_rect.topleft

                    if direction == 'up':
                        car_rect.move_ip(0, -speed)
                        sounds["up"].play()
                    elif direction == 'left':
                        car_rect.move_ip(-speed, 0)
                        sounds["left"].play()
                    elif direction == 'down':
                        car_rect.move_ip(0, speed)
                        sounds["down"].play()
                    elif direction == 'right':
                        car_rect.move_ip(speed, 0)
                        sounds["right"].play()

                    move_cooldown_time = time.time()

        # 碰撞检测红色方块
        if car_rect.colliderect(goal_rect):
            sounds["good"].play()
            score = reinit_box(car_rect, score, True)
            # 每次重新生成goal时，重新生成wall
            wall = Wall(car_rect, goal_rect)
            car_image = reinit_car()

        # 碰撞检测Wall
        if wall.check_collision(car_rect):
            print("汽车碰到墙壁！")
            # 找到汽车贴近墙的位置
            for segment in wall.wall_segments:
                if segment.colliderect(car_rect):
                    if wall.orientation == 'vertical':
                        # 如果是竖直墙，调整汽车的x位置
                        if previous_position[0] < segment.left:
                            car_rect.right = segment.left
                        else:
                            car_rect.left = segment.right
                    elif wall.orientation == 'horizontal':
                        # 如果是水平墙，调整汽车的y位置
                        if previous_position[1] < segment.top:
                            car_rect.bottom = segment.top
                        else:
                            car_rect.top = segment.bottom

        # 将汽车矩形限制在屏幕边缘
        car_rect.clamp_ip(screen.get_rect())

        screen.fill(BLACK)
        screen.blit(car_image, car_rect)
        global RED
        pygame.draw.rect(screen, RED, goal_rect)
        wall.draw(screen)  # 绘制Wall
        display_score(screen, score)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
