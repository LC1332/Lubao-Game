import pygame
import time
from pygame.locals import *
from CalcuManager import CalcuManager
from CarManager import CarManager

# 初始化 Pygame
pygame.init()

# 游戏窗口设置
scale = 80
screen_width, screen_height = 10 * scale, 9 * scale
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("数学早教游戏")

# 字体设置
font = pygame.font.SysFont(None, 120)

# 颜色设置
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 按键映射表
key_map = {K_0: 0, K_1: 1, K_2: 2, K_3: 3, K_4: 4, K_5: 5, K_6: 6, K_7: 7, K_8: 8, K_9: 9, K_q: 10}


from play_audio import play_audio, play_audios


# 实现 foo_render_ball
def foo_render_ball(current_eqn):
    print(f"调用公式渲染: {current_eqn['question']}")

# 初始化题目
def initialize_eqn(manager, history = []):
    current_eqn = manager.sample_eqn(history)
    current_co_eqn = manager.find_co_eqn(current_eqn)
    print(current_eqn)
    print(current_co_eqn)

    # 播放音频提示
    play_audios(["鲁宝,", current_eqn['question'], "和", current_co_eqn['question'], "都等于几啊?"])
    
    return current_eqn, current_co_eqn

# 渲染题目
def render_eqn(current_eqn, current_co_eqn, is_answer = False):
    # screen.fill(BLACK)
    if not is_answer:
        eqn1_text = f"{current_eqn['question']} = ?"
        eqn2_text = f"{current_co_eqn['question']} = ?"
    else:
        eqn1_text = f"{current_eqn['question']} = {current_eqn['answer']}"
        eqn2_text = f"{current_co_eqn['question']} = {current_co_eqn['answer']}"
    
    eqn1_surface = font.render(eqn1_text, True, WHITE)
    eqn2_surface = font.render(eqn2_text, True, WHITE)
    
    screen.blit(eqn1_surface, (scale, scale))
    screen.blit(eqn2_surface, (scale, 4 * scale))
    
    # pygame.display.update()

# 主游戏循环
def main_game_loop():

    is_answer = False
    manager = CalcuManager(max_ans=10)
    manager.load()

    current_eqn, current_co_eqn = initialize_eqn(manager)
    if_first_correct = True
    render_eqn(current_eqn, current_co_eqn)

    car_manager = CarManager(scale)
    
    running = True
    cooldown_time = 0.3
    last_key_time = 0

    restart_eqn = False
    restart_timer = 0
    
    while running:

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            # 键盘事件处理
            if event.type == KEYDOWN and (time.time() - last_key_time > cooldown_time):
                last_key_time = time.time()
                
                if event.key in key_map:
                    answer = key_map[event.key]

                    if answer == current_eqn['answer']:
                        audios = ["答对了,", current_eqn['question'], "和", current_co_eqn['question'], f"都是{current_eqn['answer']}。"]
                        is_answer = True
                        if if_first_correct:
                            print("一次答对")
                            audios.append("一次就答对了，奖励你看一辆新的小汽车。")
                        play_audios(audios)
                        foo_render_ball(current_eqn)

                        restart_eqn = True
                        restart_timer = time.time()

                        if if_first_correct:
                            # TODO add car render here
                            car_manager.add_car()
                            manager.add_score(current_eqn['question'])
                    else:
                        if_first_correct = False
                        wrong_eqn = manager.find_eqn_with_wa(current_eqn, current_eqn, wrong_answer=answer)
                        if wrong_eqn:
                            play_audios([wrong_eqn["question"], f"才是{wrong_eqn['answer']}哦。", "再想想，", current_eqn["question"], "和", current_co_eqn["question"], "都等于几啊?"])

        # 处理5秒后更新题目
        if restart_eqn and time.time() - restart_timer > 10 and pygame.mixer.music.get_busy() == 0:
            history = [current_eqn['answer'], current_co_eqn['answer']]
            current_eqn, current_co_eqn = initialize_eqn(manager, history=history)
            is_answer = False
            if_first_correct = True
            restart_eqn = False
            render_eqn(current_eqn, current_co_eqn)

        
        screen.fill(BLACK)

        render_eqn(current_eqn, current_co_eqn,is_answer)

        
        # 更新并渲染小汽车
        car_manager.update()
        car_manager.render(screen)

        # control time
        pygame.time.Clock().tick(60)

        pygame.display.flip()

    pygame.quit()

# 启动游戏
if __name__ == "__main__":
    main_game_loop()
