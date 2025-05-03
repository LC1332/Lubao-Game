import pygame

def load_sounds():
    pygame.mixer.init()
    sounds = {
        "up": pygame.mixer.Sound("./audio/up.wav"),
        "left": pygame.mixer.Sound("./audio/left.wav"),
        "down": pygame.mixer.Sound("./audio/down.wav"),
        "right": pygame.mixer.Sound("./audio/right.wav"),
        "good": pygame.mixer.Sound("./audio/good.wav")
    }
    return sounds
