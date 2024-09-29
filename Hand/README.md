# 需求

预计是做个0-9的手势分类

- [ ] 可以展示每个类出现的概率
- [ ] 可以抽样展示对应的加减法在右边

# Prompt

# 1

```
hand_data_train.csv和hand_data_test.csv的格式是相同的。第一列是标签列，一共有0，1，2...,9 十种标签。后面的63列都是坐标的信息。

因为63个特征分别是21个x，21个y，21个z，先对每个数据预处理减去自己的x或y或z的21个数的平均值

帮我用hand_data.csv的数据训练一个随机森林的分类器，并且在hand_data_train.csv和hand_data_test.csv上各自测试其准确率
```

# 2

```
我希望实现一段python代码。
我希望有个独立的classify_hand.py文件，这个文件没有训练的代码。
只有模型载入和classify_hand函数

在__name__为main时，才进行训练，

- 用hand_data_train.csv的数据训练一个随机森林的分类器
- 将模型存储在model中
- 重新将模型载入为model变量
- 实现一个classify_hand( coordinates, model ) 的函数，输入的coordinates是一个63维的list of float
- 验证classify_hand函数在hand_data_test.csv上的测试其准确率
```

# 3

```
{复制record_hand_data.py的代码}

这段代码可以顺利运行，并且当按键按下时，可以记录63维的手部关键点的数据。

我希望对这段程序进行如下修改:

- 对于x,y,z，生成一个归一化后的坐标，x,y,z各自减去他们的均值。
- 将63维归一化后的坐标，以list of float的形式(21个x, 21个y, 21个z)，输入下面的classify_hand的函数中，获得0-9手势分类的结果

classify_hand函数可以 from classify_hand import classify_hand
以 result = classify_hand( coordinates, model = None )的方式调用。
具体内部的代码为 {classify_hand.py的代码}

- 将结果以英文实时显示在画面上
- 不再保留运行按键响应进行数据保存部分的代码
```

# 4

```
{复制v0的代码}

这段代码可以顺利运行，另外，我在classify_hand.py中额外实现了一个函数
def predict_hand_prob(coordinates, model=None):
    # Ensure the input coordinates is a 2D array (1 sample, 63 features)
    if model is None:
        global __model
        if __model is None:
            __model = joblib.load(default_model_path)
        model = __model

    coordinates = np.array(coordinates).reshape(1, -1)
    # Make prediction probabilities
    probabilities = model.predict_proba(coordinates)
    return probabilities[0].tolist()

我希望做下面的修改

定义一个screen_height函数，方便调整最终的显示大小

- 最终显示的时候，将画面高度调整到screen_height。
- 在右侧，增加一个高度为screen_height, 宽度为 0.15 * screen_height的区域，用一个竖向条形图的方式，展示0-9类的出现概率
```

# 5

```
{复制v1的代码}

这段代码可以顺利运行，我希望参考下面这段代码

{hand_record_data}

重新增加按下按键，可以把手势数据保存到hand_record_data文件夹中的功能
```

# 6

```
{复制v2的代码}

这段代码可以顺利运行，我希望额外能够判断当前稳定保持的手势是数字多少

一开始定义一个current_digit的变量，初始化为-1

如果判断出的数字与current_digit相同，则把new_digit记录为-1

如果判断出某个数字，并且与current_digit不同，

- 如果这时new_digit为-1 则记录在new_digit变量中，同时记录对应的时间
- 如果这是new_digit不为-1，
    - 如果判断出的数字和new_digit相同，且时间超过2秒，则更新current_digit，重置new_digit为-1，并且print("手势变为{}")
    - 如果判断出的数字和new_digit不同，则重置new_digit为新判断出的数字，并且重置时间
```

# 7

```
{复制v3的代码}

这段手势识别的代码可以顺利运行，并且我已经实现了判断稳定手势的部分

并且调用了eqn = calcu_manager.sample_eqn_with_ans(current_digit)

这个会获得一个字典{'question': '2-2', 'answer': 0, 'score': 20, 'inputs': [2, 2]}

我希望对程序进行下面的修改

初始化一个display_eqn_str = "Starting"

并且以高度screen,宽度screen*0.5,较大的字体，显示在屏幕右边

当稳定检查到手势获得eqn后，更新display_eqn_str为f"{eqn['question']} = {eqn['answer']}"
```
