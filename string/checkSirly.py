import re
import json
from collections import Counter

datas = []

# 读入 string/part1_results.jsonl
with open("string/analysis_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        datas.append(data)

org_datas = []

# 读入 string/Q_A_without_answer.jsonl
with open("string/Q_A_without_answer.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        org_datas.append(data)

prefix2suffix = {}

for data in org_datas:
    prefix = data["prefix"].strip()
    suffix = data["fim_suffix"]
    prefix2suffix[prefix] = suffix

print(len(datas))
print(datas[0].keys())

count = 0
for data in datas:
    prefix = data["prefix"]
    if prefix.strip() in prefix2suffix:
        count += 1
        data["suffix"] = prefix2suffix[prefix.strip()]

def retain_alpha_numeric(s: str) -> str:
    # 先替换换行符为空字符串
    # s = s.replace('\\', '')
    # s = s.replace('n', '')
    # 再使用正则表达式替换非字母和数字的字符为空字符串
    return re.sub(r'[^a-zA-Z0-9]', '', s)

strong_find_both = 0
weak_find_both = 0

strong_data = []
weak_data = []

for data in datas:
    if "suffix" not in data:
        continue

    prefix = data["prefix"]
    suffix = data["suffix"]

    retain_prefix = retain_alpha_numeric(prefix)
    retain_suffix = retain_alpha_numeric(suffix)

    results = data["results"]

    for result in results:
        code = result['content']
        # print(code)
        retain_code = retain_alpha_numeric(code)
        
        strong_find_prefix = code.find(prefix) != -1
        strong_find_suffix = code.find(suffix) != -1
        weak_find_prefix = retain_code.find(retain_prefix) != -1
        weak_find_suffix = retain_code.find(retain_suffix) != -1

        if strong_find_prefix and strong_find_suffix:
            strong_find_both += 1
            save_data = data.copy()
            save_data["result"] = result
            save_data["full_code"] = code
            del save_data["results"]
            strong_data.append(save_data)

        elif weak_find_prefix and weak_find_suffix:
            weak_find_both += 1
            save_data = data.copy()
            save_data["result"] = result
            save_data["full_code"] = code
            del save_data["results"]
            weak_data.append(save_data)
    

print("strong_find_both, 无需后处理，清晰可用, " + str(strong_find_both) + "组")
print("weak_find_both, 需要后处理, " + str(weak_find_both) + "组")

for data in strong_data:
    full_code = data["full_code"]
    prefix = data["prefix"]
    suffix = data["suffix"]

    # Find the position of the prefix
    prefix_start = full_code.find(prefix)
    
    if prefix_start != -1:
        # Remove the prefix
        code_without_prefix = full_code[prefix_start + len(prefix):]
        
        # Find the position of the suffix in the remaining string
        suffix_start = code_without_prefix.find(suffix)
        
        if suffix_start != -1:
            # Extract middle part
            middle = code_without_prefix[:suffix_start]
            data["middle"] = middle  # Add middle to the data
        else:
            data["middle"] = None  # Suffix not found, set middle to None
    else:
        data["middle"] = None  # Prefix not found, set middle to None

# write to string/79_strong_match.jsonl
with open("string/79_strong_match.jsonl", "w", encoding="utf-8") as file:
    for data in strong_data:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")

# write to string/24_weak_match.jsonl
with open("string/24_weak_match.jsonl", "w", encoding="utf-8") as file:
    for data in weak_data:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")

matched_prefix = set()

for data in strong_data:
    matched_prefix.add(data["prefix"])

# write to not_match_data
with open("string/not_match_in_79_data.jsonl", "w", encoding="utf-8") as file:
    for data in org_datas:
        if data["prefix"] not in matched_prefix:
            json.dump(data, file, ensure_ascii=False)
            file.write("\n")