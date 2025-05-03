# 读取string/new_golden_match.jsonl, 删除reference_code字段再重新写回去
import json

datas = []
with open("string/new_golden_match.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        datas.append(data)

org_datas = []

# 读入 string/Q_A_without_answer.jsonl
with open("string/Q_A_without_answer.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        org_datas.append(data)

prefix2index = {}

for i, data in enumerate(org_datas):
    prefix = data["prefix"]
    if prefix in prefix2index:
        print("hello")
    prefix2index[prefix] = i

index2data_indexs = {}

save_datas = []

for i, data in enumerate(datas):
    prefix = data["prefix"]
    index = prefix2index[prefix]
    if index not in index2data_indexs:
        index2data_indexs[index] = []
    index2data_indexs[index].append(i)

count_not_visited = 0

for i in range(0, len(org_datas)):
    if i in index2data_indexs:
        for j in index2data_indexs[i]:
            datas[j]["original_question_id"] = i
            save_datas.append(datas[j])
    else:
        count_not_visited += 1

print(count_not_visited)

with open("string/rerange_golden_match.jsonl", "w", encoding="utf-8") as f:
    for data in save_datas:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")