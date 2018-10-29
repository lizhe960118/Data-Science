# 1.设原有的栈叫做栈A，此时创建一个额外的栈B，用于辅助原栈A
# 2.当第一个元素进入栈A的时候，让新元素（非下标）同时进入栈B。这个唯一的元素是栈A的当前最小值。
# 3.每当新元素进入栈A时，比较新元素和栈A当前最小值的大小，如果小于栈A当前最小值，则让新元素进入栈B，此时栈B的栈顶元素就是栈A当前的最小值。
# 4.每当栈A有元素出栈时，如果出栈元素是栈A当前最小值，则让栈B的栈顶元素也出栈。此时栈B余下的栈顶元素，是栈A当中原本第二小的元素，代替刚才的出栈元素成为了栈A的当前最小值。（备胎转正）
# 5.当调用getMin方法的时候，直接返回栈B的栈顶元素，即为栈A的最小元素。


class MinStack(object):
    """docstring for MinStack"""

    def __init__(self):
        self.data = []
        self.minValue = []

    def push(self, data):
        self.data.append(data)
        if len(self.minValue) == 0:
            self.minValue.append(data)
        else:
            if data <= self.minValue[-1]:
                self.minValue.append(data)

    def pop(self):
        if len(self.data) == 0:
            return None
        else:
            temp = self.data.pop()
            if temp == self.minValue[-1]:
                self.minValue.pop()
            return temp

    def getMin(self):
        if len(self.data) == 0:
            return None
        else:
            return self.minValue[-1]

    def show(self):
        print('stack data')
        for data in self.data:
            print(data)
        print('---------')
        print('minValue of the stack is', self.getMin())


if __name__ == '__main__':
    s = MinStack()
    s.push(2)
    s.push(1)
    s.show()
    s.push(4)
    s.push(3)
    s.push(2)
    s.show()
    s.pop()
    s.show()
