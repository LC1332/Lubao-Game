import re
import json
from collections import Counter

# 初始化计数器、数字列表和行列表
comma_count_freq = Counter()
numbers_list = []
merged_lines = []

# 读取文件并组合行
with open("string/file.csv", "r", encoding="utf-8") as file:
    current_line = ""
    for line in file:
        no_strip_line = line
        line = line.strip()
        
        # 检查行是否严格以数字和逗号开头，且数字不大于600
        if re.match(r'^(?:^|\n)(\d{1,3}),', line):
            match = re.match(r'^(\d+),', line)
            if match and int(match.group(1)) <= 600:
                # 如果 current_line 不为空，表示上一行是完整的，添加到 merged_lines
                if current_line:
                    merged_lines.append(current_line)
                # 重置 current_line 为新数字开头的行
                current_line = line
            else:
                # 如果行不满足数字要求，继续拼接
                current_line += "_*_*_"+ line
        else:
            # 如果不是数字开头，则拼接到 current_line
            current_line +=  "_*_*_" + line
    
    # 将最后一行添加到 merged_lines
    if current_line:
        merged_lines.append(current_line)

# 将结果转换为 list of dicts，并保存为 jsonl 格式
output_data = []
for line in merged_lines:
    # 提取开头的数字作为 id
    match = re.match(r'^(\d+),\s*(.*)', line)
    if match:
        number = int(match.group(1))
        code = match.group(2).strip()
        # 构建字典并添加到输出数据列表
        output_data.append({"id": number, "code": code})

# 将输出数据写入 JSONL 文件
with open("string/output.jsonl", "w", encoding="utf-8") as outfile:
    for entry in output_data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')

print("数据已保存到 string/output.jsonl 文件中")

original_set = []

# load string/Q_A_without_answer.jsonl
with open("string/Q_A_without_answer.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        original_set.append(data)

for i in range(len(original_set)):
    original_set[i]['id'] = i

print(original_set[0].keys())

find_prefix_count = 0
find_suffix_count = 0
both_find_count = 0

import re

import re

def retain_alpha_numeric(s: str) -> str:
    # 先替换换行符为空字符串
    # s = s.replace('\\', '')
    # s = s.replace('n', '')
    # 再使用正则表达式替换非字母和数字的字符为空字符串
    return re.sub(r'[^a-zA-Z0-9]', '', s)

both_find_data = []
find_prefix_data = []

for data in output_data:
    id = data['id']
    if id > len(original_set):
        continue
    org_data = original_set[id-1]
    prefix = retain_alpha_numeric(org_data['prefix'])
    suffix = retain_alpha_numeric(org_data['fim_suffix'])
    code = retain_alpha_numeric(data['code'])
    find_prefix_in_code = code.find(prefix)
    find_suffix_in_code = code.find(suffix)

    code = data['code']
    code = code.replace("_*_*_", "\n")
    
    data = {
        "prefix": org_data['prefix'],
        "suffix": org_data['fim_suffix'],
        "code": code,
        "find_prefix_in_code": find_prefix_in_code,
        "find_suffix_in_code": find_suffix_in_code
    }

    if find_prefix_in_code != -1:
        find_prefix_count += 1
        if find_suffix_in_code == -1:
            find_prefix_data.append(data)

    if find_suffix_in_code != -1:
        find_suffix_count += 1

    if find_prefix_in_code != -1 and find_suffix_in_code != -1:
        both_find_count += 1
        both_find_data.append(data)
        # print(suffix)

print(len(find_prefix_data))
print(len(both_find_data))


# 写入两个jsonl
with open("string/find_prefix_data.jsonl", "w", encoding="utf-8") as outfile:
    for entry in find_prefix_data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')

with open("string/both_find_data.jsonl", "w", encoding="utf-8") as outfile:
    for entry in both_find_data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"find_prefix_count: {find_prefix_count}")
print(f"find_suffix_count: {find_suffix_count}")
print(f"both_find_count: {both_find_count}")

