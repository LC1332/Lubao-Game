import re
import json
from collections import Counter

datas = []

# 读入 string/part1_results.jsonl
with open("string/79_strong_match.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        datas.append(data)

id = 5

data = datas[id]
print(data.keys())
prefix = data["prefix"]
suffix = data["suffix"]

def data2prompt(prefix, suffix):
    prompt_prefix = """Read the code in code_with_BLANK and infill the missing code represented by {BLANK}.

# code_with_BLANK:
```
#include <iostream>

int main() {
    int n;
    std::cout << "Enter a positive integer: ";
    std::cin >> n;

    int sum = 0;
    for (int i = 1; i <= n; ++i)
\{BLANK\}


    std::cout << "The sum of numbers from 1 to " << n << " is: " << sum << std::endl;
    return 0;
}
```
# Example output:
```
    {
        sum += i;
    }
```

Read the code in code_with_BLANK and infill the missing code represented by \{BLANK\}.

# code_with_BLANK
```
"""
    prompt_suffix = """
```

infill the missing code represented by \{BLANK\}, Let's think it step-by-step and output in a json format

- Summarize the code section before `BLANK`  
- Infer the purpose of the code within `BLANK`, and, considering the content after `BLANK`, determine where `BLANK` should end  
- Based on the previous inference, provide the code for `BLANK`
    """
    return prompt_prefix + prefix + "{BLANK}" + suffix + prompt_suffix

# print( data2prompt(prefix, suffix) )
# write result to prompt.txt
with open("string/prompt.txt", "w", encoding="utf-8") as file:
    file.write(data2prompt(prefix, suffix))

print(data["middle"])