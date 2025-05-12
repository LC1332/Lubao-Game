'''
我需要维护一个 feasible_map list of list

表示当前所有还没有走到的格子，

并且每走一步合法的步的时候，调用

find_solution(feasible_map, current, end)

来判断从当前点出发，是否还有可行解
'''

import pygame
import random
import json
import os
from generate_map import generate_map
from generate_map import find_solution

# 初始化参数
m, n, k = 5, 5, 2
screen_height = 600
screen_width = 800
padding = 30

# 初始化pygame
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("One Stroke Game")
clock = pygame.time.Clock()

# 加载地图
def load_map():
    # 查找匹配的json文件
    files = [f for f in os.listdir('data') if f.startswith(f'{m}_{n}_{k}')]
    if not files:
        # 生成新地图
        for _ in range(10):
            generate_map(m, n, k)
        files = [f for f in os.listdir('data') if f.startswith(f'{m}_{n}_{k}')]
    
    # 随机选择一个文件
    selected_file = random.choice(files)
    with open(f'data/{selected_file}') as f:
        data = json.load(f)
    return data['map'], data['start'], data['end']



# 绘制地图
def draw_map(map_data, start, end, path, feasible_path):
    # 计算格子大小
    cell_size = min((screen_height - 2 * padding) / m, (screen_width - 2 * padding) / n)
    
    # 绘制网格
    for i in range(m):
        for j in range(n):
            x = padding + j * cell_size
            y = padding + i * cell_size
            # 绘制格子
            pygame.draw.rect(screen, (255, 255, 255), (x, y, cell_size, cell_size), 1)
            # 绘制障碍物
            if map_data[i][j] == 1:
                pygame.draw.line(screen, (255, 0, 0), (x, y), (x + cell_size, y + cell_size), 3)
                pygame.draw.line(screen, (255, 0, 0), (x + cell_size, y), (x, y + cell_size), 3)
    
    # 绘制起点和终点
    start_x = padding + start[1] * cell_size + cell_size / 2
    start_y = padding + start[0] * cell_size + cell_size / 2
    pygame.draw.circle(screen, (0, 255, 0), (int(start_x), int(start_y)), int(cell_size / 4))
    
    end_x = padding + end[1] * cell_size + cell_size / 2
    end_y = padding + end[0] * cell_size + cell_size / 2
    pygame.draw.circle(screen, (0, 0, 255), (int(end_x), int(end_y)), int(cell_size / 4))
    
    # 绘制路径
    if len(path) > 1:
        for i in range(1, len(path)):
            x1 = padding + path[i-1][1] * cell_size + cell_size / 2
            y1 = padding + path[i-1][0] * cell_size + cell_size / 2
            x2 = padding + path[i][1] * cell_size + cell_size / 2
            y2 = padding + path[i][0] * cell_size + cell_size / 2
            
            # 判断当前线段是否在feasible_path中
            if i <= len(feasible_path):
                color = (0, 255, 0)  # 绿色
            else:
                color = (255, 0, 0)  # 红色
                
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), 3)
    
    # 绘制当前点
    if path:
        current = path[-1]
        current_x = padding + current[1] * cell_size + cell_size / 2
        current_y = padding + current[0] * cell_size + cell_size / 2
        pygame.draw.circle(screen, (255, 255, 0), (int(current_x), int(current_y)), int(cell_size / 4 * 0.6))

def check_new_position(map_data, new_pos, path, start):
    # 检查新位置是否超出边界
    if new_pos[0] < 0 or new_pos[0] >= m or new_pos[1] < 0 or new_pos[1] >= n:
        return False
    # 检查新位置是否是障碍物
    if map_data[new_pos[0]][new_pos[1]] == 1:
        return False
    # 检查新位置是否已经在路径上
    if new_pos in path:
        return False
    if new_pos == start:
        return False
    return True

# 主游戏循环
def main():
    map_data, start, end = load_map()
    current = start
    end = (end[0], end[1])
    path = [(start[0], start[1])]
    feasible_path = path
    # 初始化feasible_map
    feasible_map = [[1 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if map_data[i][j] == 0:
                feasible_map[i][j] = 0
    feasible_map[start[0]][start[1]] = 1

    feasible_path = []
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                x, y = current
                if event.key == pygame.K_UP and x > 0:
                    new_pos = (x - 1, y)
                elif event.key == pygame.K_DOWN and x < m - 1:
                    new_pos = (x + 1, y)
                elif event.key == pygame.K_LEFT and y > 0:
                    new_pos = (x, y - 1)
                elif event.key == pygame.K_RIGHT and y < n - 1:
                    new_pos = (x, y + 1)
                else:
                    continue
                
                # 检查新位置是否可行
                if check_new_position(map_data, new_pos, path, start):
                    # 更新feasible_map
                    
                    current = new_pos
                    path.append(current)
                    # print("start = ",current, "end = ", end)
                    # for row in feasible_map:
                    #     print(row)
                    # 调用find_solution进行验证
                    solution = find_solution(feasible_map, current, end)
                    if solution is None:
                        print("No solution from current position!")
                    else:
                        print("still feasible")
                        feasible_path.append(current)

                    feasible_map[current[0]][current[1]] = 1

                elif new_pos == path[-2]:  # 回到上一个位置
                    # 更新feasible_map
                    feasible_map[current[0]][current[1]] = 0
                    current = new_pos
                    path = path[:-1]
                    if len(path) < len(feasible_path):
                        feasible_path.pop()
        
        # 绘制
        screen.fill((0, 0, 0))
        draw_map(map_data, start, end, path, feasible_path)
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == '__main__':
    main()

