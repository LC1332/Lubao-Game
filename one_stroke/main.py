
import pygame
import random
import json
import os
from generate_map import generate_map
from generate_map import find_solution
import time

# 初始化参数
m, n, k = 3, 3, 1
screen_height = 700
screen_width = 900
padding = 60

# 初始化pygame
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((screen_width, screen_height)) # , pygame.FULLSCREEN
pygame.display.set_caption("One Stroke Game")
clock = pygame.time.Clock()

# 在全局变量部分添加
last_wrong_play_time = 0
COOLDOWN = 120  # 2分钟冷却时间

current_map_name = None

# 加载地图
def load_map(map_name = None):
    # 查找匹配的json文件
    files = [f for f in os.listdir('data') if f.startswith(f'{m}_{n}_{k}')]
    if not files:
        global m, n, k
        if k == 0:
            m, n, k = 4, 4, 1
            return load_map()
        else:
            k = 0
            return load_map()
        # 生成新地图
        # for _ in range(10):
        #     generate_map(m, n, k)
        # files = [f for f in os.listdir('data') if f.startswith(f'{m}_{n}_{k}')]
    
    if map_name is not None and os.path.exists(f'data/{map_name}'):
        selected_file = map_name
    else:
        # 随机选择一个文件
        selected_file = random.choice(files)
        pygame.mixer.music.load('audios/new_problem.mp3')
        pygame.mixer.music.play()
    
    global current_map_name
    current_map_name = selected_file
    
    with open(f'data/{selected_file}') as f:
        data = json.load(f)
    start = data['start']
    start = (start[0], start[1])
    end = data['end']
    end = (end[0], end[1])
    

    global last_wrong_play_time
    # 在全局变量部分添加
    last_wrong_play_time = 0
    COOLDOWN = 120  # 2分钟冷却时间

    return data['map'], start, end



# 绘制地图
def draw_map(map_data, start, end, path, feasible_path, if_win):
    # 显示M N K
    font = pygame.font.Font(None, 36)
    text = font.render(f'M: {m} N: {n} K: {k}', True, (255, 255, 255))
    screen.blit(text, (10, 10))
    
    # 显示胜利信息
    if if_win:
        win_text = font.render('WIN!', True, (15, 255, 10))
        font = pygame.font.Font(None, 100)
        screen.blit(win_text, (screen_width // 2 - 50, screen_height - 50))
        
    
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
            if i < len(feasible_path):
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

def get_feasible_map(m, n, map_data, start):
    feasible_map = [[1 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if map_data[i][j] == 0:
                feasible_map[i][j] = 0
    feasible_map[start[0]][start[1]] = 1
    return feasible_map


# 主游戏循环
def main():
    global m, n, k
    map_data, start, end = load_map()
    current = start
    
    path = [(start[0], start[1])]
    feasible_path = path.copy()
    # 初始化feasible_map
    feasible_map = get_feasible_map(m, n, map_data, start)
    

    # feasible_path = []
    
    running = True

    if_win = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                new_pos = None

                x, y = current
                if event.key == pygame.K_UP and x > 0:
                    new_pos = (x - 1, y)
                elif event.key == pygame.K_DOWN and x < m - 1:
                    new_pos = (x + 1, y)
                elif event.key == pygame.K_LEFT and y > 0:
                    new_pos = (x, y - 1)
                elif event.key == pygame.K_RIGHT and y < n - 1:
                    new_pos = (x, y + 1)
                elif event.key in [pygame.K_r, pygame.K_t]:
                    if event.key == pygame.K_r:
                        map_data, start, end = load_map(current_map_name)
                    else:
                        map_data, start, end = load_map()
                    if_win = False
                    current = start
                    path = [(start[0], start[1])]
                    feasible_path = path.copy()
                    feasible_map = get_feasible_map(m, n, map_data, start)
                elif event.key in (pygame.K_q, pygame.K_a, pygame.K_w, pygame.K_s, pygame.K_e, pygame.K_d):
                    if event.key == pygame.K_q and m > 3:
                        m -= 1
                    elif event.key == pygame.K_a and m < 7:
                        m += 1
                    elif event.key == pygame.K_w and n > 3:
                        n -= 1
                    elif event.key == pygame.K_s and n < 7:
                        n += 1
                    elif event.key == pygame.K_e and k > 0:
                        k -= 1
                    elif event.key == pygame.K_d and k < 7:
                        if k * 5 <= m * n:
                            k += 1
                    map_data, start, end = load_map()
                    if_win = False
                    current = start
                    path = [(start[0], start[1])]
                    feasible_path = path.copy()
                    feasible_map = get_feasible_map(m, n, map_data, start)
                
                if new_pos is None:
                    pass
                # 检查新位置是否可行
                elif check_new_position(map_data, new_pos, path, start):

                    print("path:", path)
                    # 更新feasible_map
                    print("new_pos = ", new_pos)
                    current = new_pos
                    path.append(current)
                    # print("start = ",current, "end = ", end)
                    # for row in feasible_map:
                    #     print(row)
                    # 调用find_solution进行验证
                    solution = find_solution(feasible_map, current, end)
                    if solution is None:
                        print("No solution from current position!", current, end)
                        for row in feasible_map:
                            print(row)
                        
                        global last_wrong_play_time
                        # 修改load_map函数中的播放逻辑
                        if pygame.mixer.music.get_busy() == False and time.time() - last_wrong_play_time > COOLDOWN:
                            pygame.mixer.music.load('audios/wrong.mp3')
                            pygame.mixer.music.play()
                            last_wrong_play_time = time.time()
                    else:
                        
                        feasible_path.append(current)
                        total_cells = sum(1 for row in feasible_map for cell in row if cell == 0)
                        if total_cells == 1:
                            print("Congratulations! You have completed the game!")
                            if_win = True
                            pygame.mixer.music.load('audios/win.mp3')
                            pygame.mixer.music.play()
                        else:
                            print("still feasible")

                    print("len_path = ", len(path), " len_feasible_path = ", len(feasible_path))
                    print("path : ", path)
                    print("feasible_path : ", feasible_path)

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
        draw_map(map_data, start, end, path, feasible_path, if_win)
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == '__main__':
    main()

