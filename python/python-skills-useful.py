# 交换变量值
a, b = b, a

# 将列表中所有元素合成字符串
a = ["python", "i", "love"]
print(" ".join(a))

# 找出列表中频率最高的值
"""most frequent element in a list"""
a = [1, 2, 3, 1, 3, 2, 2, 2, 4, 3]
print(max(set(a), key=a.count))
"""using Counter from collections"""
from collections import Counter
cnt = Counter(a)
print(cnt.most_common(3))

# 检查两个字符串是不是由相同字母组成
from collections import Counter
Counter(str1) == Counter(str2)

# 翻转字符串
a = 'absafiiaqqgakk'
print(a[::-1])

for char in reversed(a):
    print(char)

num = 12144155
print(int(str(num)[::-1]))

# 反转列表
a = [1, 2, 3, 4, 5]
print(a[::-1])

# 转置二维数组
"""
transpose 2d array [['a', 'b'], ['c', 'd'], ['e', 'f']] into [['a', 'c', 'e'], ['b', 'd', 'f']]
"""
original = [['a', 'b'], ['c', 'd'], ['e', 'f']]
transposed = zip(*original)
print(list(transposed))

#链式比较
"""chained comparison with all kind of operators"""
b = 6
print(4 < b < 7)
print(1 == b < 20)


# 链式函数调用
def product(a, b):
    return a * b

def add(a, b):
    return a + b

b = True
print((product if b else add)(5, 7))

# 复制列表
a = [1, 2, 3, ,4, 5]
b = a # 指向同一个对象
b[0] = 10
print(a, b) #都改变
b = a[：] #重新建一个对象
b[0] = 10
print(a, b) #只有b改变

b = a.copy() # 如果数组一维则a不会随b变

from copy import deepcopy
l = [[1, 2], [3, 4]]
l2 = deepcopy(l) # 深复制，l可以多维

# 字典的get方法
d = {"a":1, "b":2}
print(d.get('c', 3))
# return None or default value when key is not in dict

# 通过key来排序字典
print(soted(d.items(), key =lambda x : x[1]))

from operator import itemgetter
print(sorted(d.items(), key = itemgetter(1)))

# sorted key by value
print(sorted(d, key=d.get))

# 转换列表为逗号分隔符形式
numbers = [1, 2, 3, 4,5]
l = ",".join(map(str, numbers))

# 合并字典
d1 = {"a" : 1}
d2 = {"b" : 2}
print({**d1, **d2})

print(dict(d1.items() | d2.items()))

d1.update(d2)
print(d1)

# 列表中最大值和最小值的索引
l = [1, 2, 3, 4]

def minIndex(l):
    return min(range(len(l)), key=l.__getitem__)

def maxIndex(l):
    return max(range(len(l)), key=l.__getitem__)

# 移除列表中的重复元素
l = [1, 2, 3, 4, 1, 2, 3]
new_l = list(set(l))

"remove dups and key order"
from collections import OrderedDict
print(list(OrderedDict.fromkeys(l).keys()))
