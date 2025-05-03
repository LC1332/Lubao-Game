import os
import random
import pygame
from settings import SCREEN_HEIGHT, carScale, car_width

car_imgs_folder = "./cars"

def reinit_car():
    car_img_file = random.choice(os.listdir(car_imgs_folder))
    car_img_path = os.path.join(car_imgs_folder, car_img_file)
    car_image = pygame.image.load(car_img_path).convert_alpha()
    car_image = pygame.transform.scale(car_image, (car_width, car_width))
    return car_image
