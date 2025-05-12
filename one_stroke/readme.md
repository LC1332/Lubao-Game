'''
一个一笔画的pygame迷宫游戏

# 地图读入

定义m = 5, n = 5, k = 2

先检查data的文件夹下是否有 {m}_{n}_{k}开头的json文件

没有则先from generate_map import generate_map

运行10次generate_map(m, n, k)生成10个json文件

然后随机选取一个json文件

读入对应的map, start和end字段

# 地图显示

screen_height = 800, screen_width = 600, padding = 30

每个格子的边长为 min( (screen_height - 2 * padding)/m, (screen_width - 2 * padding)/n )

并且第一个格子的左上角在 (padding, padding) 开始显示，格子都用白色细线来表示

有石头的格子，使用红色的粗叉在中心显示，表示格子不能走

起点用绿色的圆圈来表示，终点用蓝色的圆圈来表示

初始化当前格子为start, path 为[start]

# 交互

当用户按下上下左右时，如果下一格子是可以走的，则append到path中，

并且用绿色的线来渲染整个线路

'''


'''
复制

删除

用一个比起点小（60%大）的圆圈表示当前的位置，check_new_position 要同步考虑new_pos不能在path上
'''


# 

'''
为当前代码增加交互

# 显示

使用全屏进行显示，自动更新screen_height和weight的值

在画面上方显示M N K 三个数字， 表示当前的地图大小和石头的个数

按下q a可以增加和减少M， 最小是3，最大是7、
按下w s可以增加和减少N， 最小是3，最大是7
按下e d可以增加和减少K， 最小是0，最大是7

m n k调整后会在data文件夹下 寻找是否有{m}_{n}_{k}开头的json文件，
如果没有，则会把k调整成0再寻找一次，如果还是没有，则不接受m n k的调整

调整后重新初始化游戏

## esc 键
按下可以退出

r键可以重新调整游戏
'''