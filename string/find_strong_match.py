import re
import json
from collections import Counter

datas = []

# 读取 string/part1_results.jsonl
with open("string/b_github_search.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        datas.append(data)

print(len(datas))
print(datas[0].keys())

org_datas = []

# 读取 string/Q_A_without_answer.jsonl
with open("string/Q_B_without_answer.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        org_datas.append(data)

# 初始化计数和保存的列表
count_found = 0
save_datas = []

# 遍历每个数据
for i, data in enumerate(datas):
    prefix = org_datas[i]["prefix"]
    suffix = org_datas[i]["fim_suffix"]
    
    for res in data["results"]:
        content = res["content"]
        
        # 检查 prefix 和 suffix 是否都在 content 中
        if prefix in content and suffix in content:
            count_found += 1
            
            # 提取 prefix 和 suffix 之间的字符串
            prefix_index = content.index(prefix) + len(prefix)
            suffix_index = content.index(suffix)
            
            if prefix_index < suffix_index:  # 确保索引有效
                middle = content[prefix_index:suffix_index].strip()
                
                # 构建保存的数据结构
                save_data = data.copy()
                del save_data["results"]
                save_data["middle"] = middle
                
                save_datas.append(save_data)
            break

# 打印统计结果
print(f"Found {count_found} matches.")

# 可将结果保存到文件
with open("string/strong_match_B.jsonl", "w", encoding="utf-8") as file:
    for item in save_datas:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
