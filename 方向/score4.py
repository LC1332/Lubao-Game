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
from pyamaze import maze
import os
import random



car_imgs_folder = "./cars"


MIN_BLOCK_NUM = 3
MAX_BLOCK_NUM = 16

block_num = 5

m = maze(block_num, block_num)
m.CreateMaze()
maze_map = m.maze_map

cell_width = SCREEN_WIDTH // m.rows
cell_height = SCREEN_HEIGHT // m.cols

cell_size = min(cell_width, cell_height)
cell_width = cell_size
cell_height = cell_size
wall_width = max(1,cell_size // 20)



def display_score(screen, score):
    font = pygame.font.Font(None, 56)
    text = font.render(str(score), True, (255, 255, 255))
    # 计算右上角位置：屏幕宽度 - 文本宽度 - 右边距（20像素），y坐标设为20像素上边距
    text_x = SCREEN_WIDTH - text.get_width() - 20
    text_y = 20
    screen.blit(text, (text_x, text_y))


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
    # 计算居中位置：格子左上角 + (格子尺寸 - 汽车尺寸)/2
    car_rect.x = (col - 1) * cell_size + (cell_size - car_width) // 2
    car_rect.y = (row - 1) * cell_size + (cell_size - car_width) // 2  # 汽车是正方形，宽高相同
    return car_image, car_rect


def reinit_box(car_rect, score, if_increase=False):
    global RED
    if if_increase:
        score += 1
    # 计算目标方块尺寸（与汽车尺寸保持一致）
    goal_size = cell_size - 8 * wall_width
    # 设置目标方块尺寸（关键修正：之前未设置宽高导致居中不准）
    goal_rect.width = goal_size
    goal_rect.height = goal_size
    # 随机选择一个不同的整数格点
    while True:
        row = random.randint(1, m.rows)
        col = random.randint(1, m.cols)
        # 计算居中位置：格子左上角 + (格子尺寸 - 目标尺寸)/2
        goal_offset = (cell_size - goal_size) // 2
        goal_x = (col - 1) * cell_size + goal_offset
        goal_y = (row - 1) * cell_size + goal_offset
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
    # 删除原错误的局部goal_rect创建（之前错误覆盖了全局goal_rect）


def reset_game_state(score, if_increase=False):
    """
    重置游戏状态（生成新迷宫/调整单元格尺寸/重新初始化小车和目标）
    :param score: 当前分数
    :param if_increase: 是否增加分数（用于碰撞目标时）
    :return: (新m对象, 新maze_map, 新cell_width, 新cell_height, 新cell_size, 新wall_width, 新car_image, 新car_rect, 新score)
    """
    # 重新生成迷宫
    new_m = maze(block_num, block_num)
    new_m.CreateMaze()
    new_maze_map = new_m.maze_map
    
    # 重新计算单元格尺寸
    new_cell_width = SCREEN_WIDTH // new_m.rows
    new_cell_height = SCREEN_HEIGHT // new_m.cols
    new_cell_size = min(new_cell_width, new_cell_height)
    new_cell_width = new_cell_size
    new_cell_height = new_cell_size
    new_wall_width = max(1, new_cell_size // 20)
    
    # 重新初始化小车和目标位置
    new_car_image, new_car_rect = reinit_car()
    new_score = reinit_box(new_car_rect, score, if_increase)
    
    return new_m, new_maze_map, new_cell_width, new_cell_height, new_cell_size, new_wall_width, new_car_image, new_car_rect, new_score


def main():
    # 删除原speed设置，改为直接使用cell尺寸
    # 设置速度（原speed = SCREEN_HEIGHT / 10 已删除）
    global m, maze_map, cell_width, cell_height, cell_size, wall_width
    global block_num

    score = 0
    sounds = load_sounds()
    car_image, car_rect = reinit_car()
    score = reinit_box(car_rect, score, False)

    directions = None

    running = True
    move_cooldown_time = 0
    cooldown_time = 0.3

    previous_position = car_rect.topleft

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:  # 添加R键处理逻辑
                    # 调用重置函数（不增加分数）
                    m, maze_map, cell_width, cell_height, cell_size, wall_width, car_image, car_rect, score = reset_game_state(score, False)
                elif event.key == pygame.K_q:
                    block_num = max(MIN_BLOCK_NUM, block_num - 1)  # 不小于最小值
                    # 调整后重新生成游戏状态
                    m, maze_map, cell_width, cell_height, cell_size, wall_width, car_image, car_rect, score = reset_game_state(score, False)
                elif event.key == pygame.K_w:
                    
                    block_num = min(MAX_BLOCK_NUM, block_num + 1)  # 不超过最大值
                    # 调整后重新生成游戏状态
                    m, maze_map, cell_width, cell_height, cell_size, wall_width, car_image, car_rect, score = reset_game_state(score, False)

            direction = handle_input(event, directions)
            if direction: 
                if time.time() - move_cooldown_time > cooldown_time: 
                    previous_position = car_rect.topleft 
                    # 获取当前所在的格子坐标（pyamaze行列从1开始）
                    current_col = (car_rect.x) // cell_size + 1
                    current_row = (car_rect.y) // cell_size + 1
                    print(f"当前所在格子坐标：({current_row}, {current_col})")
                    can_move = False  # 默认不能移动

                    # 根据移动方向检测对应方向的墙是否存在（0有墙，1无墙）
                    if direction == 'up': 
                        if maze_map[(current_row, current_col)]['N'] == 1:  # 北边无墙
                            can_move = True
                    elif direction == 'left': 
                        if maze_map[(current_row, current_col)]['W'] == 1:  # 西边无墙
                            can_move = True
                    elif direction == 'down': 
                        if maze_map[(current_row, current_col)]['S'] == 1:  # 南边无墙
                            can_move = True
                    elif direction == 'right': 
                        if maze_map[(current_row, current_col)]['E'] == 1:  # 东边无墙
                            can_move = True
                    
                    if can_move:  # 只有无墙时才允许移动
                        # 调整移动距离为对应方向的格子尺寸
                        if direction == 'up': 
                            car_rect.move_ip(0, -cell_height)  # 纵向移动使用cell_height
                            sounds["up"].play() 
                        elif direction == 'left': 
                            car_rect.move_ip(-cell_width, 0)  # 横向移动使用cell_width
                            sounds["left"].play() 
                        elif direction == 'down': 
                            car_rect.move_ip(0, cell_height)  # 纵向移动使用cell_height
                            sounds["down"].play() 
                        elif direction == 'right': 
                            car_rect.move_ip(cell_width, 0)  # 横向移动使用cell_width
                            sounds["right"].play() 
                        
                        move_cooldown_time = time.time() 

        # 碰撞检测红色方块
        if car_rect.colliderect(goal_rect):
            sounds["good"].play()
            # 调用重置函数（增加分数）
            m, maze_map, cell_width, cell_height, cell_size, wall_width, car_image, car_rect, score = reset_game_state(score, True)

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

