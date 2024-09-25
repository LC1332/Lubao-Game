import pygame
import random
import math
import pygame
import sys

# 初始化Pygame
pygame.init()
screen_width = 1000
screen_height = 650

class Polygon:
    def __init__(self, radius, N, is_regular=True):
        self.screen_width, self.screen_height = screen_width, screen_height
        self.x, self.y = self.screen_width // 2, self.screen_height // 2  # 屏幕中心
        self.radius = radius
        self.N = N
        self.is_regular = is_regular
        self.vertices = []
        self.set_speed(5, 5)
        self.generate_polygon()
        self.current_edge_render_index = 0  # 当前渲染的边索引
        self.update_count = 0  # 用于控制边的渲染更新
        self.M = 30  # 每 M 次更新更新一次边的渲染

    def generate_polygon(self):
        if self.is_regular:
            # 生成正多边形
            angle_step = 2 * math.pi / self.N
            self.vertices = [(self.x + self.radius * math.cos(i * angle_step), 
                              self.y + self.radius * math.sin(i * angle_step)) for i in range(self.N)]
        else:
            # 生成凸多边形，使用极角排序法
            points = self.generate_random_points(self.N)
            convex_hull = self.generate_convex_hull(points)
            self.vertices = self.adjust_to_radius(convex_hull)

    # 生成随机点
    def generate_random_points(self, n):
        return [(random.uniform(self.x - self.radius, self.x + self.radius),
                 random.uniform(self.y - self.radius, self.y + self.radius)) for _ in range(n)]

    # 计算两点之间的极角
    def polar_angle(self, p0, p1=None):
        if p1 is None:
            p1 = (self.x, self.y)
        y_span = p0[1] - p1[1]
        x_span = p0[0] - p1[0]
        return math.atan2(y_span, x_span)

    # 计算凸包
    def generate_convex_hull(self, points):
        centroid = self.calculate_centroid(points)
        points = sorted(points, key=lambda p: self.polar_angle(p, centroid))
        return points

    # 计算质心
    def calculate_centroid(self, points):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        centroid_x = sum(x_coords) / len(points)
        centroid_y = sum(y_coords) / len(points)
        return (centroid_x, centroid_y)

    # 调整顶点到中心的距离为指定的半径
    def adjust_to_radius(self, points):
        centroid = self.calculate_centroid(points)
        adjusted_points = []
        for p in points:
            # 计算从质心到该点的向量
            vector_x = p[0] - centroid[0]
            vector_y = p[1] - centroid[1]
            # 计算该点到质心的距离
            distance = math.sqrt(vector_x ** 2 + vector_y ** 2)
            # 调整距离到指定的半径
            if distance != 0:
                factor = self.radius / distance
                adjusted_x = self.x + vector_x * factor
                adjusted_y = self.y + vector_y * factor
                adjusted_points.append((adjusted_x, adjusted_y))
        return adjusted_points

    def set_speed(self, v_x, v_y):
        self.v_x, self.v_y = v_x, v_y

    def get_speed(self):
        return self.v_x, self.v_y

    def update(self):
        self.x += self.v_x
        self.y += self.v_y

        # 更新顶点位置
        for i, (vx, vy) in enumerate(self.vertices):
            self.vertices[i] = (vx + self.v_x, vy + self.v_y)

        # 如果多边形的中心超出屏幕边界，速度反向
        if self.x - self.radius < 0 or self.x + self.radius > self.screen_width:
            self.v_x = -self.v_x
        if self.y - self.radius < 0 or self.y + self.radius > self.screen_height:
            self.v_y = -self.v_y

        # 每 M 次更新current_edge_render_index
        self.update_count += 1
        if self.update_count >= self.M:
            self.current_edge_render_index = (self.current_edge_render_index + 1) % self.N
            self.update_count = 0

    def render(self, screen):
        pygame.draw.polygon(screen, (0, 255, 0), self.vertices)
        # 渲染多边形的每一条边
        for i in range(self.N):
            next_i = (i + 1) % self.N
            color = (255, 0, 0) if i == self.current_edge_render_index else (0, 255, 0)  # 蓝色或绿色
            pygame.draw.line(screen, color, self.vertices[i], self.vertices[next_i], 8)


# 主程序设置
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Polygon Game')
clock = pygame.time.Clock()

# 设置字体
font = pygame.font.Font(None, 36)  # 使用默认字体，字体大小为36

# 游戏相关参数
radius = 100
sides = 3
is_regular = True
polygon = Polygon(radius, sides, is_regular)

# 速度调节
speed_change = 5
max_speed = 30

# 游戏主循环
while True:
    screen.fill((0, 0, 0))  # 填充黑色背景

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # 按键事件处理
        if event.type == pygame.KEYDOWN:
            if event.key in range(pygame.K_3, pygame.K_9 + 1):
                sides = event.key - pygame.K_0
                polygon = Polygon(radius, sides, is_regular)
                polygon.set_speed(5, 5)

            if event.key == pygame.K_SPACE:
                is_regular = not is_regular
                polygon = Polygon(radius, sides, is_regular)

            if event.key == pygame.K_RETURN:
                radius *= 1.1
                if radius > screen_width // 2:
                    radius = 100
                polygon = Polygon(radius, sides, is_regular)

            if event.key == pygame.K_UP:
                polygon.set_speed(polygon.v_x, max(-max_speed, polygon.v_y - speed_change))
            if event.key == pygame.K_DOWN:
                polygon.set_speed(polygon.v_x, min(max_speed, polygon.v_y + speed_change))
            if event.key == pygame.K_LEFT:
                polygon.set_speed(max(-max_speed, polygon.v_x - speed_change), polygon.v_y)
            if event.key == pygame.K_RIGHT:
                polygon.set_speed(min(max_speed, polygon.v_x + speed_change), polygon.v_y)

    # 更新多边形位置
    polygon.update()

    # 渲染多边形
    polygon.render(screen)

    # 创建显示边数的文本
    sides_text = font.render(f"Sides: {sides}", True, (255, 255, 255))  # 白色文本
    screen.blit(sides_text, (10, 10))  # 将文本渲染到左上角

    # 更新显示
    pygame.display.flip()

    # 控制帧率
    clock.tick(40)
