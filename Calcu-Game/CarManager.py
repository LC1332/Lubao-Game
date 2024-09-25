import pygame
import os
import time

class Car:
    def __init__(self, img_path, x, y, size, speed=5):
        self.image = pygame.image.load(img_path)
        self.image = pygame.transform.scale(self.image, (size, size))
        self.x = x
        self.y = y
        self.size = size
        self.speed = speed
        self.target_x = x
        self.target_y = y

    def set_target(self, t_x, t_y):
        self.target_x = t_x
        self.target_y = t_y

    def update(self):
        # 水平方向移动
        if abs(self.target_x - self.x) > self.speed:
            self.x += self.speed if self.target_x > self.x else -self.speed
        else:
            self.x = self.target_x
        
        # 垂直方向移动
        if abs(self.target_y - self.y) > self.speed:
            self.y += self.speed if self.target_y > self.y else -self.speed
        else:
            self.y = self.target_y

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))


class CarManager:
    def __init__(self, scale):
        self.cars = []
        self.car_images = []
        self.scale = scale
        self.current_increase_id = 0

        # 扫描cars目录下的jpg文件
        car_dir = "cars"
        for filename in os.listdir(car_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(car_dir, filename)
                self.car_images.append(img_path)

    def add_car(self):
        if self.current_increase_id >= 27:  # 最多显示27辆车
            # return
            self.cars = self.cars[:3]
            self.current_increase_id = 3

        col = self.current_increase_id % 3
        row = self.current_increase_id // 3

        # 新车的目标位置
        target_x = 7 * self.scale + col * self.scale
        target_y = row * self.scale

        # 新车的初始位置
        start_x = target_x
        start_y = 9 * self.scale

        car_image_path = self.car_images[self.current_increase_id % len(self.car_images)]
        new_car = Car(car_image_path, start_x, start_y, size=self.scale, speed=10)
        new_car.set_target(target_x, target_y)
        self.cars.append(new_car)

        self.current_increase_id += 1

    def update(self):
        for car in self.cars:
            car.update()

    def render(self, screen):
        for car in self.cars:
            car.render(screen)



def test_car_manager():
    # 初始化 Pygame
    pygame.init()

    # 游戏窗口设置
    scale = 70
    screen_width, screen_height = 10 * scale, 9 * scale
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Car Manager Test")

    # 颜色设置
    BLACK = (0, 0, 0)

    # 初始化 CarManager
    car_manager = CarManager(scale)

    running = True
    last_key_time = 0
    cooldown_time = 0.2  # 防止过快响应按键

    while running:
        screen.fill(BLACK)

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # 按空格键时增加一辆车
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if time.time() - last_key_time > cooldown_time:
                    car_manager.add_car()
                    last_key_time = time.time()

        # 更新和渲染小汽车
        car_manager.update()
        car_manager.render(screen)

        # 更新屏幕
        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    test_car_manager()