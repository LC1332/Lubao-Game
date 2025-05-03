
import os
# 设置代理环境变量
# os.environ['HTTP_PROXY'] = 'http://localhost:8234'
# os.environ['HTTPS_PROXY'] = 'http://localhost:8234'

from zhipuai import ZhipuAI

import tiktoken


def get_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

zhipu_api_key = get_api_key("data/zhipu_apikey.txt")  # 读取API密钥

def count_tokens(text: str, model_name: str = 'gpt-3.5-turbo') -> int:
    # 加载指定模型的编码器
    encoder = tiktoken.encoding_for_model(model_name)
    
    # 对输入文本进行 token 化并计算 token 数量
    tokens = encoder.encode(text)
    return len(tokens)


def truncate_text_from_start(text: str, max_tokens: int, max_lines: int, model_name: str = 'gpt-3.5-turbo') -> str:
    encoder = tiktoken.encoding_for_model(model_name)
    lines = text.splitlines()
    selected_lines = []
    token_count = 0
    
    for line in lines[:max_lines]:
        line_tokens = encoder.encode(line)
        if token_count + len(line_tokens) > max_tokens:
            break
        selected_lines.append(line)
        token_count += len(line_tokens)
    
    return "\n".join(selected_lines)

def truncate_text_from_end(text: str, max_tokens: int, max_lines: int, model_name: str = 'gpt-3.5-turbo') -> str:
    encoder = tiktoken.encoding_for_model(model_name)
    lines = text.splitlines()
    selected_lines = []
    token_count = 0
    
    for line in reversed(lines[-max_lines:]):
        line_tokens = encoder.encode(line)
        if token_count + len(line_tokens) > max_tokens:
            break
        selected_lines.insert(0, line)
        token_count += len(line_tokens)
    
    return "\n".join(selected_lines)


# 看看79个金色案例的


# golden_data = []

# # 读入 string/Q_A_without_answer.jsonl
# with open("string/79_strong_match.jsonl", "r", encoding="utf-8") as file:
#     for line in file:
#         data = json.loads(line)
#         golden_data.append(data)

# max_middle_token_len = 0
# for data in golden_data:
#     middle = data["middle"]
#     middle_token_len = count_tokens(middle)
#     if middle_token_len > max_middle_token_len:
#         max_middle_token_len = middle_token_len
#         print("max_middle_token_len:", max_middle_token_len)


import json
import time
from tqdm import tqdm

org_datas = []

# 读入 string/Q_A_without_answer.jsonl
with open("string/Q_B_without_answer.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        org_datas.append(data)

save_datas = []



# 增量写入函数
def append_to_jsonl(file_path, data):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")

# 目标文件路径
output_file = "string/GLM_B_result.jsonl"

# 如果文件已存在，则清空文件内容（可选）
with open(output_file, "w", encoding="utf-8") as file:
    pass

# 遍历数据集，带进度条
for i, data in tqdm(enumerate(org_datas), total=len(org_datas), desc="Processing"):
    prefix = data["prefix"]
    suffix = data["fim_suffix"]

    try:

        len_prefix = count_tokens(prefix)
        if len_prefix > 2500:
            prefix = truncate_text_from_end(prefix, 2500, 2500)

        len_suffix = count_tokens(suffix)
        if len_suffix > 1000:
            suffix = truncate_text_from_start(suffix, 1000, 1000)
    except:
        pass

    result = ""
    if_success = False

    # 尝试多次请求，最多 3 次
    for test_attempt in range(3):
        try:
            client = ZhipuAI(api_key=zhipu_api_key)  # 使用 API Key

            response = client.chat.completions.create(
                model="codegeex-4",
                messages=[],
                extra={
                    "target": {
                        "path": "",
                        "language": "",
                        "code_prefix": prefix,
                        "code_suffix": suffix
                    },
                    "contexts": []
                },
                temperature=0.1,
                max_tokens=256,
                stop=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|observation|>"]
            )
            result = response.choices[0].message.content
            if_success = True
            break
        except Exception as e:
            time.sleep(3)  # 等待 3 秒重试

    if if_success:
        # 保存结果到字典
        save_data = data.copy()
        save_data["middle_GLM"] = result
        save_data["if_success"] = if_success
    else:
        save_data = data.copy()
        save_data["middle_GLM"] = ""
        save_data["if_success"] = if_success

    # 增量写入到 GLM_result.jsonl
    append_to_jsonl(output_file, save_data)