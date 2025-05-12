'''
我希望实现一个一笔画的关卡生成函数 generate_map

输入 边长m 边长n 石头的个数 k

输出是一个tuple，有三个元素
map : m*n的list of list 表示地图 ， 0 表示没有石头， 1 表示有石头(不能走到)
start: tuple (x,y) 表示起点
end: tuple (x,y) 表示终点

输出一个合法的地图tuple组合，并且同时用json的格式，以data/{m}_{n}_{k}.json的形式保存到data文件夹下

这里我希望这个地图是有feasible的solution的。

所以需要先实现一个函数 find_solution( map, start, end) ， 

输入一个地图和起点终点，判断是否有solution，如果有，以list of tuple的形式输出一个solution，不然则返回none

并且，我们需要这个路径能够访问地图上所有非石头的格子。

可以考虑使用一个深度优先搜索+格子计数的方法来进行实现

对于generate_map函数，会尝试至少10次，直到找到一个可行的地图，然后返回这个地图。找不到则返回None

name == main的时候， 用m = 4, n = 4, k = 3来进行测试
'''


def find_solution(map, start, end):
    m, n = len(map), len(map[0])
    visited = [[False for _ in range(n)] for _ in range(m)]
    path = []
    
    # 计算非石头格子的总数
    total_cells = sum(1 for row in map for cell in row if cell == 0)

    # print("total_cells = ", total_cells)

    # print(total_cells)

    def dfs(x, y):
        if x < 0 or x >= m or y < 0 or y >= n or map[x][y] == 1 or visited[x][y]:
            return False
        visited[x][y] = True
        path.append((x, y))
        if (x, y) == end:
            # 检查路径长度是否等于非石头格子总数
            if len(path) >= total_cells:
                return True
            else:
                path.pop()
                visited[x][y] = False
                return False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if dfs(x + dx, y + dy):
                return True
        path.pop()
        visited[x][y] = False
        return False
    
    if dfs(start[0], start[1]):
        return path
    return None


import random
import json
import os

def generate_map(m, n, k):
    for _ in range(10):
        map_res = [[0 for _ in range(n)] for _ in range(m)]
        stones = random.sample(range(m * n), k)
        for stone in stones:
            x, y = divmod(stone, n)
            map_res[x][y] = 1
        
        # 确保起点和终点都在边缘
        edge_positions = []
        for i in range(m):
            for j in range(n):
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                    edge_positions.append((i, j))
        
        start = random.choice(edge_positions)
        while map_res[start[0]][start[1]] == 1:
            start = random.choice(edge_positions)
        
        end = random.choice(edge_positions)
        while map_res[end[0]][end[1]] == 1 or end == start:
            end = random.choice(edge_positions)
        
        if find_solution(map_res, start, end):
            if not os.path.exists('data'):
                os.makedirs('data')
            suffix = random.randint(0, 1000000)
            with open(f'data/{m}_{n}_{k}_{suffix}.json', 'w') as f:
                json.dump({'map': map_res, 'start': start, 'end': end}, f)
            return map_res, start, end
    return None

if __name__ == '__main__':
    result = generate_map(4, 4, 2)
    if result:
        map_res, start, end = result
        map_res[start[0]][start[1]] = 'S'
        map_res[end[0]][end[1]] = 'E'
        print("Generated map:", )
        print("Start:", start)
        print("End:", end)
        for row in map_res:
            print(row)
    else:
        print("No valid map found after 10 attempts.")

    meaningful_tuple = [(3,4,1), (4,4,2),(4,4,3), (5,5,2)]
    from tqdm import tqdm

    # for m, n, k in meaningful_tuple:
    #     for _ in tqdm(range(10)):
    #         result = generate_map(m, n, k)

    for m in range(3, 8):
        for n in range(3, 8):
            for k in range(0, 8):
                if k * 5 > m * n:
                    continue
                for _ in tqdm(range(5)):
                    result = generate_map(m, n, k)
                    if result:
                        map_res, start, end = result

    

