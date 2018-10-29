# 1.使用python实现switch-with
将数字1，2，3，4映射为Spring，Summer，Fall，Winter，而其它数字映射为Invalid Season。

```
# Java代码中可以用Switch/Case语句来实现
public static String getSeason(int season){
    String SeasonName = "",
    switch(season){
    case1:
        SeasonName = 'Spring';
        break;
    case2:
        SeasonName = 'Summer';
        break;
    case3:
        SeasonName = 'Fall';
        break;
    case1:
        SeasonName = 'Winter';
        break;
    default:
        SeasonName = 'Invalid Season';
        break;
}
return SeasonName;
}
```
Python中没有Switch/Case语句，那么该如何实现呢？
```
# 第一种是通过 if... elif... elif... else 来实现
def getSeason(season):
    """
    将season映射为字符串
    :param season:
    :return:
    """
    if season == 1:
        return 'Spring'
    elif season == 2:
        return "Summer"
    elif season == 3:
        return "Fall"
    elif season == 4:
        return "Winter"
    else:
        return "Invalid Season"

if __name__ == '__main__':
    print(getSeason(1))

# 第二种方式，也是比较好的一种方式
# 是通过字典(dict)来进行实现的：
season_dict = {
    1: "Spring",
    2: "Summer",
    3: "Fall",
    4: "Winter"
}

def getSeason(season):
    return season_dict.get(season, "Invalid Season")

if __name__ == '__main__':
    print(getSeason(1))

# 如果case中是执行不同的方法，而不是简单的返回字符串，有没有办法实现呢？
def Season1():
    return "spring"

def Season2():
    return "Summer"

def Season3():
    return "Fall"

def Season4():
    return "winter"

def Default():
    return "Invalid season"

season_dict = {
    1: Season1,
    2: Season2,
    3: Season3,
    4: Season4
}


def getSeason(season):
        fun = season_dict.get(season, Default)
        return fun()


if __name__ == '__main__':
    print(getSeason(8))

# 还有一种方式，即通过在类中定义不同的方法（方法名有一定的规则），然后通过getattr函数来进行实现

class Season():
    def Season1(self):
        return "spring"

    def Season2(self):
        return "summer"

    def Season3(self):
        return "fall"

    def Season4(self):
        return "winter"

    def Default(self):
        return "Invalid season"

    def getSeason(self, season):
        season_name = "Season" + str(season)
        fun = getattr(self, season_name, self.Default)
        return fun()


if __name__ == '__main__':
    season1 = Season()
    # 先实例化
    print(season1.getSeason(1))
```

# 2.python实现链表

```
# 单项链表包含两个域：
# 信息域，指针域
# 这些节点在逻辑上是相连的，但要知道它们在物理内存上并不相连。

# 1.Node类
class Node(object):
    def __init__(self, initdata):
        self.data = initdata
        self.next = None
    def getData(self):
        return self.data
    def getNext(self):
        return self.next
    def setData(self, newData):
        self.data = newData
    def setNext(self, newnext):
        self.next = newnext
# temp = Node(33)
# print(temp)
# print(temp.getData())

# 2.Unordered List类
# Unordered List类本身不包含任何节点对象，它只包含对链表结构中第一个节点的单个引用
class unOrderedList():
    def __init__(self):
        self.head = None

myList = unOrderedList()

# 检查是否为空链表
def isEmpty(self):
    return self.head == None

# 3.基本操作
# 1）add（）在链表前段添加元素
def add(self, item):
    temp = Node(item) #创建一个新节点
    temp.setNext(self.head) #更改新节点指向旧链表的第一个节点
    self.head = temp # 将链表的头指向新节点

# 2）size() 求链表长度
def size(self):
    current = self.head
    count = 0
    while current != None:
        count += 1
        current = current.getNext()
    return count

# 3)使用search寻找
def search(self, item):
    current = self.head
    found = False
    while current != None and not found:
        if current.getData() == item:
            found = True
        current = current.getNext()
    return found

# 4）使用remove（）删除节点
def remove(self, item):
    current = self.head
    previous = None
    found = False
    while not found and current != None:
        if current.getData() == item:
            found = True
        else:
            previous = current
            current = current.getNext()
    if previous == None:
        self.head = current.getNext()
    else:
        previous.setNext(current.getNext())
```

# 3.数字在排序数组中出现的次数
```
# 使用二分法分别找到数组中第一个和最后一个出现的值的坐标，然后相减
def get_k_counts(nums, k):
    first = get_first_k(nums, k)
    last = get_last_k(nums, k)
    if first < 0 and last < 0:
        return 0
    if first < 0 or last < 0:
        return 1
    return last - first + 1

def get_first_k(nums, k):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < k:
            if mid + 1 < len(nums) and nums[mid + 1] == k:
                return mid + 1
            left = mid + 1
        elif nums[mid] == k:
            if mid == 0 or (mid - 1 >= 0 and nums[mid - 1] < k):
                return mid
            right = mid - 1
        else:
            right = mid - 1
    return -1

def get_last_k(nums, k):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > k:
            if mid - 1 > 0 and nums[mid - 1] == k:
                return mid - 1
            right = mid - 1
        elif nums[mid] == k:
            if mid + 1 == len(nums) or (mid + 1 < len(nums) and nums[mid + 1] > k):
                return mid
            left = mid + 1
        else:
            left = mid + 1
    return -1

if __name__ == '__main__':
    print(get_k_counts([2, 3, 4, 4, 4, 4, 5], 4))
```
# 4.二叉树的深度
```
# 分别递归地求左右子树的深度
def get_depth(tree):
    if not tree:
        return 0
    if not tree.left and not tree.right:
        return 1
    return 1 + max(get_depth(tree.left), get_depth(tree.right))
```

# 5.数组中只出现一次的数字
```
# 数组中除了两个只出现一次的数字外，其他数字都出现了两次
# 按位异或，在得到的值中找到二进制最后一个1，然后把数组按照改位是0还是1分成两组
# 返回两个数字，它们都只出现了一次
def get_only_one_number(nums):
    if not nums:
        return None
    tem_ret = 0
    for n in nums:
        tem_ret ^= n
    last_one = get_bin(tem_ret)
    a, b = 0, 0
    for n in nums:
        if is_last_one(n, last_one):
            a ^= n
        else:
            b ^= n
    return [a, b]

def get_bin(num): #得到最后一个1的位置
    ret = 0
    while num & 1 == 0 and ret < 32:
        num = num >> 1
        ret += 1
    return ret

def is_last_one(num, last_one):
    num = num >> last_one
    return num & 1
```

# 6.和为s的两个数字 和 连续正数序列中和为s的两个数字
```
# 输入一个递增排序的数组和数字s，要求在数组nums中查找两个数，使之和为s
# 思路：设置头尾两个指针，和大于s则尾指针减小，和小于s则头指针增加：
def sum_to_s(nums, s):
    start, end = 0, len(nums) - 1
    while start < end:
        if nums[start] + nums[end] < s:
            start += 1
        elif nums[start] + nums[end] > s:
            end -= 1
        else:
            return [nums[start], nums[end]]
    return None
```

# 7.和为s的连续整数序列
```
# 输入一个正数s，打印出所有和为s的连续的正整数序列：
def sum_to_s(nums, s):
    start, end = 1, 2
    answer = []
    while start < s / 2 + 1:
        if sum(range(start, end + 1)) < s:
            end += 1
        elif sum(range(start, end + 1)) > s:
            start += 1
        else:
            answer.append(range(start, end + 1))
            start + 1
    return answer
```

# 8.翻转单词顺序与左旋转字符串
```
#  1）python中字符串是不可变对象，不能用书中的方法，可以直接转化成列表然后转回去：
def reverse_words(sentence):
	temp = sentence.split()
	return ''.join(temp[::-1])
# 2）把字符串的前面的若干位移到字符串的后面
def rotate_string(s, n):
	if not s:
		return ''
	n %= len(s)
	return s[n:] + s[:n]
```

# 9.n个色子的点数
```
# 求出n个骰子朝上一面之和s所有可能值出现的概率
# n出现的可能是前面n-1到n-6所有可能的和，设置两个数组，分别保存每一轮
def get_probability(n):
    if n < 1:
        return []
    data1 = [0] + [1] * 6 + [0] * 6 * (n - 1)
    data2 = [0] + [0] * 6 * n
    flag = 0
    for v in range(2, n + 1):
        if flag:
            for k in range(v, 6 * v + 1):
                data1[k] = sum([data2[k - j] for j in range(1, 7) if k > j])
            flag = 0
        else:
            for k in range(v, 6 * v + 1):
                data2[k] = sum([data1[k - j] for j in range(1, 7) if k > j])
            flag = 1
    ret = []
    total = 6 ** n
    data = data2[n:] if flag else data1[n:]
    for v in data:
        ret.append(v * 1.0 / total)
    print(data)
    return ret

if __name__ == '__main__':
    print(get_probability(1))
    print(get_probability(2))
    print(get_probability(3))
    print(get_probability(4))
    print(get_probability(5))
    print(get_probability(6))
    print(get_probability(7))
```

# 10.从扑克牌中随机抽取5张牌，判断是不是顺子，大小王可以当任意值
```
import random
def is_continus(nums, k):
	data = [random.choice(nums) for _ in range(k)]
	data.sort()
	print(data)
	zero = data.count(0)
	small, big = zero, zero + 1
	while big < k:
		if data[small] == data[big]:
			return False
		tmp = data[big] - data[small]
		if tmp > 1:
			if tmp - 1 > zero:
				return False
			else:
				zero -= tmp - 1
				small += 1
				big += 1
		else:
			small += 1
			big += 1
	return True

if __name__ == '__main__':
	nums = [x for x in range(27)] 
	print(is_continus(nums, 5))
```

# 11.圆圈中最后剩下的数字
```
# 0到n-1排成一圈，从0开始每次数m个数删除，求最后剩余的数
# 当n > 1 时：f(n,m) = [f(n-1,m) + m]%n
# 当n = 1 时：f(n,m) = 0

def f(n, m):
	if n == 1:
		return 0
	else:
		ans = (f(n - 1, m) + m) % n
	return ans

def f(n, m):
	ans = 0
	if n <= 0 or m == 0:
		return -1
	elif n == 1:
		return 0
	else:
		for i in range(2, n+1):
			ans = (ans + m) % i
	return ans

if __name__ == '__main__':
	print(f(32,4))
	print(f(5,2))
# https://www.nowcoder.com/questionTerminal/f78a359491e64a50bce2d89cff857eb6
```

# 12.求 1+2+...+n
```
# 不能使用乘除、for 、while、if、else 等
# 方法一：使用range和sum：
def get_sum1(n):
	answer = sum(range(1, n+1))
	return answer
# 方法二：使用reduce
from functools import reduce
def get_sum2(n):
	answer = reduce(lambda x,y: x + y, range(1,n+1))
	# reduce函数会对参数序列中元素进行累积。
	# function参数是一个有两个参数的函数，reduce依次从sequence中取一个元素，和上一次调用function的结果做参数再次调用function。

	# map()将函数调用映射到每个序列的对应元素上并返回一个含有所有返回值的列表
	# map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])，最后的返回结果为：[3, 7, 11, 15, 19]

	# filter函数会对指定序列执行过滤操作。
	# def is_even(x):
	# 	return x & 1 != 0
	# filter(is_even, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	# 返回结果为：[1, 3, 5, 7, 9]
	return answer

if __name__ == '__main__':
	print(get_sum1(100))
	print(get_sum2(100))
```

# 13.不用加减乘除做加法
```
# 方法一：使用位carry需要加个判断运算，python中大整数会动处理，因此对
def bit_add(n1, n2):
	carry = 1
	while carry:
		s = n1 ^ n2
		carry = 0xFFFFFFFF & ((n1 & n2) << 1)
		carry = -(~(carry - 1) & 0xFFFFFFFF) if carry > 0x7FFFFFFF else carry
		n1 = s
		n2 = carry
	return n1

# 方法二：使用sum:
def add(n1, n2):
	return sum([n1, n2])
```

# 14.把字符串转化为整数
```
# 测试用例：0，正负数，空字符，包含其他字符
# 备注：使用raise抛出异常作为非法提示
def str2int(string):
	if not string:
		raise Exception('String cannot be None',string)
	flag = 0 #判断是否第一个字符为 + 、-号
	ret = 0
	for k, s in enumerate(string):
		if s.isdigit():
			val = ord(s) - ord('0')
			ret = ret*10 + val
		else:
			if not flag:
				if s == '+' and k == 0:
					flag = 1
				elif s == '-' and k == 0:
					flag = -1
				else:
					raise Exception('digit is need', string)
			else:
				raise Exception('digit is need', string)
	if flag and len(string) == 1:
		raise Exception('digit is need', string)
	return ret if flag >= 0 else ret*(-1)
```

# 15.求普通二叉树中两个节点的最低公共祖先
```
# 方法：先求出两个节点到根节点的路径，然后从路径中找出最后一个公共节点
class Solution(object):

	def __init__(self, root, node1, node2):
		self.root = root
		self.node1 = node1
		self.node2 = node2

	@staticmethod
	def get_path(root, node, ret):
		"""获取节点的路径"""
		if not root or not node:
			return False
		ret.append(root)
		if root == node:
			return True
		left = Solution.get_path(root.left, node, ret)
		right = Solution.get_path(root.right, node, ret)
		if left or right:
			return True
		ret.pop()

	def get_last_common_node(self):
		"""获取公共节点"""
		route1 = []
		route2 = []
		ret1 = Solution.get_path(self.root, self.node1, route1)
		ret2 = Solution.get_path(self.root, self.node2, route2)
		ret = None
		if ret1 and ret2:
			length = len(route1) if len(route1) <= len(route2) else len(route2)
			index = 0
			while index < length:
				if route1[index] == route2[index]:
					ret = route1[index]
				else:
					index += 1
		return ret
```