import re
import json
from collections import Counter

datas = []


# 先对retain进行预搜索
from tqdm import tqdm

prefix2candidate_codes = {}

import re

def retain_alpha_numeric(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', s)



# 读入 string/part1_results.jsonl
with open("string/analysis_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        datas.append(data)

candidate_codes = []

for data in datas:
    results = data["results"]
    prefix = data["prefix"]
    retain_prefix = retain_alpha_numeric(prefix)
    for result in results:
        code = result['content']
        if code not in candidate_codes:
            candidate_codes.append(code)
            if retain_prefix not in prefix2candidate_codes:
                prefix2candidate_codes[retain_prefix] = [code]
            else:
                prefix2candidate_codes[retain_prefix].append(code)



org_datas = []

# 读入 string/Q_A_without_answer.jsonl
with open("string/Q_A_without_answer.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        org_datas.append(data)



def find_strong_and_weak_parts(text: str, direction: str) -> tuple:
    parts = text.split('\n')
    if direction == 'prefix':
        strong_part = '\n'.join(parts[-3:]) if len(parts) >= 3 else text
        weak_part = '\n'.join(parts[:-3]) if len(parts) > 3 else ""
    elif direction == 'suffix':
        strong_part = '\n'.join(parts[:3]) if len(parts) >= 3 else text
        weak_part = '\n'.join(parts[3:]) if len(parts) > 3 else ""
    return strong_part, weak_part

def score_match(weak_part: str, matched_part: str) -> int:
    weak_alpha = retain_alpha_numeric(weak_part)
    matched_alpha = retain_alpha_numeric(matched_part)
    score = 0
    for i in range(min(len(weak_alpha), len(matched_alpha))):
        if weak_alpha[i] == matched_alpha[i]:
            score += 1
        else:
            break
    return score

import re
import time

def super_match(code: str, prefix: str, suffix: str) -> str:
    # 提取 strong 和 weak 部分
    strong_prefix, weak_prefix = find_strong_and_weak_parts(prefix, 'prefix')
    strong_suffix, weak_suffix = find_strong_and_weak_parts(suffix, 'suffix')

    # 提高匹配条件
    prefix_len = len(strong_prefix)
    suffix_len = len(strong_suffix)
    
    if prefix_len < 5:
        strong_prefix = code[:min(10, len(code))] + strong_prefix
    if suffix_len < 5:
        strong_suffix = strong_suffix + code[-min(10, len(code)):]

    max_score = -1
    best_middle = None

    start_time_prefix = time.time()  # 开始计时

    # 逐个处理每个 prefix 的匹配位置
    for prefix_match in re.finditer(re.escape(strong_prefix), code):
        start = prefix_match.start()
        match_text_before = code[:start]
        prefix_score = score_match(weak_prefix, match_text_before)

        if time.time() - start_time_prefix > 15:
            print("Prefix matching time exceeded 15 seconds; exiting loop.")
            break
        
        # 设置 suffix 搜索的起点，并记录开始时间
        suffix_search_start = start + len(strong_prefix)
        start_time = time.time()  # 开始计时

        # 在每个 prefix 后寻找符合长度的 suffix
        for suffix_match in re.finditer(re.escape(strong_suffix), code[suffix_search_start:]):
            end = suffix_search_start + suffix_match.end()

            # 检查是否超出4秒限制
            if time.time() - start_time > 4:
                print("Suffix matching time exceeded 4 seconds; exiting loop.")
                break  # 跳出 suffix 的匹配循环

            # 计算并验证中间文本
            if end > start:
                middle = code[start:end]
                if 0 < len(middle) <= 600:
                    match_text_after = code[end:]
                    suffix_score = score_match(weak_suffix, match_text_after)
                    total_score = prefix_score + suffix_score - len(middle) / 100
                    
                    # 更新最大得分并记录最佳匹配
                    if total_score > max_score:
                        max_score = total_score
                        best_middle = middle

                    # 如果满足最大得分，可以选择提前返回
                    if max_score == prefix_score + suffix_score:
                        return best_middle
                else:
                    break

    return best_middle if best_middle else None


import json
from tqdm import tqdm

# 将 "string/file.csv" 读取为 long_candidate
with open("string/file.csv", "r", encoding="utf-8") as file:
    long_candidate = file.read()

target_file = "string/new_golden_match.jsonl"

existed_in_target = set()

with open(target_file, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        existed_in_target.add(data["prefix"])

# 打开文件以进行增量写入
with open(target_file, "a", encoding="utf-8") as output_file:
    for data in tqdm(org_datas):
        prefix = data["prefix"]
        prefix_retain = retain_alpha_numeric(prefix)

        if prefix_retain in existed_in_target:
            continue
        
        # 检查是否在 prefix2candidate_codes 中
        if prefix_retain not in prefix2candidate_codes:
            continue

        suffix = data["fim_suffix"]

        # 获取候选代码列表
        cands = prefix2candidate_codes[prefix_retain]

        # 根据条件将 long_candidate 添加到候选代码中
        if len(suffix) > 0 and len(prefix) > 100:
            cands.append(long_candidate)

        # 在候选代码中寻找匹配的 middle
        for code in cands:
            middle = super_match(code, prefix, suffix)
            if middle:
                # 复制数据并添加新的匹配结果
                save_result = data.copy()
                save_result["middle"] = middle
                # save_result["reference_code"] = code
                
                # 将结果写入文件，确保非 ASCII 字符也被正确写入
                output_file.write(json.dumps(save_result, ensure_ascii=False) + "\n")
                
                # 找到匹配后，跳出循环
                break