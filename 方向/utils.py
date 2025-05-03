import pygame

def display_score(screen, score):
    font = pygame.font.Font(None, 56)
    text = font.render(str(score), True, (255, 255, 255))
    screen.blit(text, (100, 100))
