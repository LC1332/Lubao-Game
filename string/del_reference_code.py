# 读取string/new_golden_match.jsonl, 删除reference_code字段再重新写回去
import json

datas = []
with open("string/new_golden_match.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        datas.append(data)

with open("string/new_golden_match.jsonl", "w", encoding="utf-8") as f:
    for data in datas:
        # data = json.loads(line)
        del data["reference_code"]
        f.write(json.dumps(data, ensure_ascii=False) + "\n")