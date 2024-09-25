from VolcanoTTS import get_audio
import pygame
import time
import threading

def play_audio(text):
    text = text.replace('-', '减')
    # 获取音频文件路径
    audio_file_path = get_audio(text)
    
    # 初始化 pygame.mixer
    pygame.mixer.init()
    
    # 停止当前播放的音频（如果有）
    pygame.mixer.music.stop()

    # 加载音频文件
    pygame.mixer.music.load(audio_file_path)
    
    # 播放音频
    pygame.mixer.music.play()
    
    # 阻塞模式，等待音频播放完成
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# def play_audios(texts):
#     def play_audio_sequence():
#         for text in texts:
#             play_audio(text)
    
#     # 启动新线程播放音频
#     audio_thread = threading.Thread(target=play_audio_sequence)
#     audio_thread.start()
# 全局锁和当前播放的线程引用
audio_lock = threading.Lock()
current_audio_thread = None
stop_flag = False  # 停止播放的标志

def stop_current_audio():
    """ 请求停止当前正在播放的音频序列 """
    global stop_flag, current_audio_thread
    stop_flag = True  # 设置标志，请求停止音频播放

    if current_audio_thread and current_audio_thread.is_alive():
        current_audio_thread.join()  # 等待线程安全退出
        current_audio_thread = None  # 清空当前播放线程的引用

def play_audios(texts):
    """ 播放多个音频文本，保证播放时不会被打断 """
    global stop_flag, current_audio_thread

    def play_audio_sequence():
        global stop_flag
        stop_flag = False  # 每次新播放时重置标志
        for text in texts:
            if stop_flag:  # 如果 stop_flag 为 True，则停止播放
                break
            play_audio(text)

    # 获取全局锁，确保当前的播放序列可以安全停止
    with audio_lock:
        stop_current_audio()  # 停止之前的播放
        # 启动新线程播放音频
        current_audio_thread = threading.Thread(target=play_audio_sequence)
        current_audio_thread.start()

# 示例用法
if __name__ == "__main__":
    pygame.init()

    from CalcuManager import CalcuManager
    manager = CalcuManager()
    current_eqn = manager.sample_eqn()
    current_co_eqn = manager.find_co_eqn(current_eqn)

    texts = ["鲁宝,", current_eqn['question'], "和", current_co_eqn['question'], "都等于几啊?"]

    play_audios(texts)

    # 保证主线程不会立刻退出
    while threading.active_count() > 1:
        time.sleep(0.1)

    pygame.quit()