'''
score3.py给出了只有一个墙的迷宫代码，其中包含了按键交互、声音和小汽车图案的使用方式

maze_example.py给出了pyamaze生成迷宫的例子代码

我希望重构一个新的迷宫游戏，使用pyamaze生成迷宫，并且使用score3.py的交互方案
'''
import pygame
import time
from settings import screen, BLACK, SCREEN_HEIGHT, SCREEN_WIDTH, carScale, car_width
from sounds import load_sounds
from car import reinit_car
from goal import reinit_box, goal_rect, RED
from controls import handle_input
from utils import display_score
from pyamaze import maze
import os
import random

car_imgs_folder = "./cars"

block_num = 5

m = maze(block_num, block_num)
m.CreateMaze()
maze_map = m.maze_map

cell_width = SCREEN_WIDTH // m.rows
cell_height = SCREEN_HEIGHT // m.cols

cell_size = min(cell_width, cell_height)
wall_width = max(1,cell_size // 20)


def reinit_car():
    global car_image
    car_img_file = random.choice(os.listdir(car_imgs_folder))
    car_img_path = os.path.join(car_imgs_folder, car_img_file)
    car_image = pygame.image.load(car_img_path).convert_alpha()
    car_width = int(cell_size - 8 * wall_width)
    car_image = pygame.transform.scale(car_image, (car_width, car_width))
    # 随机选择一个整数格点
    row = random.randint(1, m.rows)
    col = random.randint(1, m.cols)
    car_rect = car_image.get_rect()
    car_rect.x = (col - 1) * cell_size
    car_rect.y = (row - 1) * cell_size
    return car_image, car_rect


def reinit_box(car_rect, score, if_increase=False):
    global RED
    if if_increase:
        score += 1
    # 随机选择一个不同的整数格点
    while True:
        row = random.randint(1, m.rows)
        col = random.randint(1, m.cols)
        goal_x = (col - 1) * cell_size
        goal_y = (row - 1) * cell_size
        goal_rect.x = goal_x
        goal_rect.y = goal_y
        if not car_rect.colliderect(goal_rect):
            break
    RED = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    if sum(RED) < 100:
        RED = (random.randint(0, 225)+30, random.randint(0, 225)+30, random.randint(0, 225)+30)
    return score


def draw_maze(screen):
    cell_size = min(cell_width, cell_height )
    for row in range(1, m.rows + 1):
        for col in range(1, m.cols + 1):
            x = (col - 1) * cell_size
            y = (row - 1) * cell_size
            # 修正单引号转义问题
            if maze_map[(row, col)]['N'] == 0:
                pygame.draw.line(screen, (255, 255, 255), (x, y), (x + cell_size, y), wall_width)
            if maze_map[(row, col)]['S'] == 0:
                pygame.draw.line(screen, (255, 255, 255), (x, y + cell_size), (x + cell_size, y + cell_size), wall_width)
            if maze_map[(row, col)]['E'] == 0:
                pygame.draw.line(screen, (255, 255, 255), (x + cell_size, y), (x + cell_size, y + cell_size), wall_width)
            if maze_map[(row, col)]['W'] == 0:
                pygame.draw.line(screen, (255, 255, 255), (x, y), (x, y + cell_size), wall_width)
            goal_size = cell_size - 8 * wall_width
            goal_rect = pygame.Rect(0, 0, goal_size, goal_size)


def main():
    # 设置速度
    speed = SCREEN_HEIGHT / 10

    score = 0
    sounds = load_sounds()
    car_image, car_rect = reinit_car()
    # car_rect.center = (screen.get_width() // 2, screen.get_height() // 2)

    directions = None

    running = True
    move_cooldown_time = 0
    cooldown_time = 0.3

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
                    # 修正单引号转义问题
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
            car_image, car_rect = reinit_car()

        # 确保汽车在迷宫边界内移动
        car_rect.clamp_ip(screen.get_rect())

        screen.fill(BLACK)
        draw_maze(screen)
        screen.blit(car_image, car_rect)
        pygame.draw.rect(screen, RED, goal_rect)
        display_score(screen, score)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

