import pygame
import random
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, car_width, WHITE

class Wall:
    def __init__(self, car_rect, goal_rect):
        self.wall_segments = []
        self.valid = self.create_wall(car_rect, goal_rect)

    def create_wall(self, car_rect, goal_rect):
        # 根据车和方块的距离判断墙的方向
        if abs(car_rect.x - goal_rect.x) > abs(car_rect.y - goal_rect.y):
            # 竖直墙
            x_center = (car_rect.x + goal_rect.x) // 2
            self.orientation = 'vertical'
            hole_position = random.randint(0, SCREEN_HEIGHT - car_width * 2)

            # 上部分墙
            top_rect = pygame.Rect(x_center - 10, 0, 20, hole_position)
            # 下部分墙
            bottom_rect = pygame.Rect(x_center - 10, hole_position + car_width * 2, 20, SCREEN_HEIGHT - (hole_position + car_width * 2))

            self.wall_segments = [top_rect, bottom_rect]
        else:
            # 水平墙
            y_center = (car_rect.y + goal_rect.y) // 2
            self.orientation = 'horizontal'
            hole_position = random.randint(0, SCREEN_WIDTH - car_width * 2)

            # 左部分墙
            left_rect = pygame.Rect(0, y_center - 10, hole_position, 20)
            # 右部分墙
            right_rect = pygame.Rect(hole_position + car_width * 2, y_center - 10, SCREEN_WIDTH - (hole_position + car_width * 2), 20)

            self.wall_segments = [left_rect, right_rect]

        # 检查是否与汽车或方块碰撞
        for segment in self.wall_segments:
            if segment.colliderect(car_rect) or segment.colliderect(goal_rect):
                print("墙生成错误")
                return False

        return True

    def draw(self, screen):
        if self.valid:
            for segment in self.wall_segments:
                pygame.draw.rect(screen, WHITE, segment)

    def check_collision(self, car_rect):
        if not self.valid:
            return False
        for segment in self.wall_segments:
            if segment.colliderect(car_rect):
                return True
        return False
