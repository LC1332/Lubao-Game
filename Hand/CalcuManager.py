import json
import random
import os

class CalcuManager:
    def __init__(self, max_ans=10):
        self.max_ans = max_ans
        self.datas = []
        self.score_file = "question2score.jsonl"
        self._initialize_datas()

    def _initialize_datas(self):
        # 初始化所有的加法算式
        for result in range(1, self.max_ans + 1):
            for i in range(1, result):
                eqn = f"{i}+{result - i}"
                data = {
                    'question': eqn,
                    'answer': result,
                    'score': 20,
                    'inputs': [i, result - i]
                }
                self.datas.append(data)
            
            # 增加(result+1) - 1 = result
            eqn = f"{result+1}-{1}"
            data = {
                'question': eqn,
                'answer': result,
                'score': 20,
                'inputs': [result+1, 1]
            }
            self.datas.append(data)

            # 增加result - result = 0
            eqn = f"{result}-{result}"
            data = {
                'question': eqn,
                'answer': 0,
                'score': 20,
                'inputs': [result, result]
            }
            self.datas.append(data)

    def load(self, file_name=None):
        # 加载熟练度文件
        file_name = file_name or self.score_file
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                for line in f:
                    question_data = json.loads(line)
                    existing_data = next((d for d in self.datas if d['question'] == question_data['question']), None)
                    if existing_data:
                        existing_data['score'] = question_data['score']
                    else:
                        self.datas.append(question_data)

    def save(self, file_name=None):
        # 保存熟练度到文件
        file_name = file_name or self.score_file
        with open(file_name, 'w') as f:
            for data in self.datas:
                f.write(json.dumps(data) + "\n")

    def sample_eqn(self, history=[], MIN_K=10):
        # 采样未在history中的加法算式，按熟练度排序后随机选择一个
        sorted_datas = sorted(self.datas, key=lambda x: x['score'])
        candidates = [d for d in sorted_datas if d['answer'] not in history]
        if len(candidates) < MIN_K:
            MIN_K = len(candidates)
        return random.choice(candidates[:MIN_K]) if candidates else None
    
    def sample_eqn_with_ans( self, ans , MIN_K = 3):
        # 采样答案为ans的加法算式，按熟练度排序后随机选择一个
        sorted_datas = sorted(self.datas, key=lambda x: x['score'])
        candidates = [d for d in sorted_datas if d['answer'] == ans]
        if len(candidates) < MIN_K:
            MIN_K = len(candidates)
        return random.choice(candidates[:MIN_K]) if candidates else None

    def find_co_eqn(self, data):
        # 寻找相同答案但不同问题的加法算式
        same_answer_eqns = [d for d in self.datas if d['answer'] == data['answer'] and d['question'] != data['question']]
        return random.choice(same_answer_eqns) if same_answer_eqns else None

    def find_eqn_with_wa(self, data1, data2, wrong_answer):
        # 寻找输入集有重合且答案错误的加法算式
        input_set = set(data1['inputs']) | set(data2['inputs'])
        candidates = [d for d in self.datas if d['answer'] == wrong_answer and any(i in input_set for i in d['inputs'])]
        return random.choice(candidates) if candidates else None

    def add_score(self, question, delta=1):
        # 增加熟练度
        self._update_score(question, delta)

    def minus_score(self, question, delta=1):
        # 减少熟练度
        self._update_score(question, -delta)

    def _update_score(self, question, delta):
        # 更新某个算式的熟练度并保存
        data = next((d for d in self.datas if d['question'] == question), None)
        if data:
            data['score'] += delta
            data['score'] = max(0, data['score'])  # 保证分数不为负数
            self.save()

# 如果这个脚本作为主程序执行时，进行类的初始化与测试
if __name__ == "__main__":
    manager = CalcuManager(max_ans=10)
    manager.load()  # 加载之前保存的熟练度数据

    # 测试样例
    sample = manager.sample_eqn(history=['1+2', '2+2'])
    print("Sampled equation:", sample)

    co_eqn = manager.find_co_eqn(sample)
    print("Co-related equation with same answer:", co_eqn)

    wrong_eqn = manager.find_eqn_with_wa(sample, co_eqn, wrong_answer=7)
    print("Equation with wrong answer:", wrong_eqn)

    # manager.add_score(sample['question'], delta=1)
    # manager.minus_score(co_eqn['question'], delta=1)
    
    manager.save()  # 保存更新后的熟练度数据

    for ans in range(0,10):
        eqn = manager.sample_eqn_with_ans(ans)
        print("Sampled equation with answer {}: {}".format(ans, eqn))
