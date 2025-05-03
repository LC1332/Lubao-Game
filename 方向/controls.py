import time
import pygame

# up_button = ['2', '3', '4', '5', '6', '7', '8', '9', '0', \
# '-', '=', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'F1', 'F2', 'F3',\
#  'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12']
 
# left_button = ['tab', '`', 'q', 'w', 'a', 's', 'd', 'f', 'g', \
# 'Caps_Lock', 'left-shift', 'z', 'x', 'c', 'left-ctrl', 'KP_Left']

# down_button = ['v', 'b', 'n', 'm', ',', '.', 'space', 'KP_Down','，','。']

# right_button = ['、','/','j', 'k', 'l', '；',';','’', "'", 'return', '【','[',\
# '】', ']','p', '\\', 'right-shift', 'right-ctrl', 'KP_Right']

default_directions = {
    "up": ['2', '3', '4', '5', '6', '7', '8', '9', '0', \
'-', '=', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'F1', 'F2', 'F3',\
 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12'],
    "left": ['tab', '`', 'q', 'w', 'a', 's', 'd', 'f', 'g', \
'Caps_Lock', 'left-shift', 'z', 'x', 'c', 'left-ctrl', 'KP_Left'],
    "down": ['v', 'b', 'n', 'm', ',', '.', 'space', 'KP_Down','，','。'],
    "right": ['、','/','j', 'k', 'l', '；',';','’', "'", 'return', '【','[',\
'】', ']','p', '\\', 'right-shift', 'right-ctrl', 'KP_Right']
}



def handle_input(event, directions = None):
    if directions is None:
        directions = default_directions
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
            return False
        # 检查方向键（上下左右）
        elif event.key == pygame.K_UP:
            return 'up'
        elif event.key == pygame.K_LEFT:
            return 'left'
        elif event.key == pygame.K_DOWN:
            return 'down'
        elif event.key == pygame.K_RIGHT:
            return 'right'
        # 检查自定义的键位组合
        elif event.unicode in directions['up']:
            return 'up'
        elif event.unicode in directions['left']:
            return 'left'
        elif event.unicode in directions['down']:
            return 'down'
        elif event.unicode in directions['right']:
            return 'right'
    return None
