import re
import json
from collections import Counter

datas = []

# 读入 string/part1_results.jsonl
with open("string/part1_results.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        datas.append(data)

print(len(datas))
