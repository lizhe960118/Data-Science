**Table of Contents**

[TOC]

# python基础
## Python基础01 HelloWorld!
```
print('Hello World!')
```

## Python基础02 基本数据类型
```
a = 10
print(a)
print(type(a))
10<class 'int'>
# 回收变量名
a = 1.3
print(a, type(a))
1.3 <class 'float'>
```

## Python基础03 序列
```
# 序列有两种：tuple（定值表； 也有翻译为元组） 和 list (表)
s1 = (2, 1.3, 'love', 5.6, 9, 12, False)         # s1是一个tuple
s2 = [True, 5, 'smile']                          # s2是一个list
print(s1, type(s1))
# <class 'tuple'>
print(s2, type(s2))
# <class 'list'>
s3 = [1, [3,4,5]]
# tuple和list的主要区别在于，一旦建立，tuple的各个元素不可再变更，而list的各个元素可以再变更。
# 元素的引用
print(s1[0])
print(s2[2])
print(s3[1][2])
# 其他引用方式
print(s1[:5]) #从开始到下标4（下标5的元素不包括在内）
print(s2[2:]) #从下标2到最后
print(s1[0:5:2]) #从下标0到下标4（下标5的元素不包括在内），每隔2取一个元素（0,2,4）
print(s1[2:0:-1]) #从下标2到下标1
print(s1[-1]) #序列最后一个元素
print(s1[-3]) #序列倒数第三个元素
# 字符串是元组
str1 = 'abcdef'
print(str1[2:4])
print(type(str1))
# tuple元素不可变，list元素可变，字符串是一种tuple
```

## Python基础04 运算

## Python基础05 缩进和选择
```
if i > 0:
	x = 1
	y = 2
```
在Python中， 去掉了i > 0周围的括号，去除了每个语句句尾的分号，表示块的花括号也消失了。多出来了if ...之后的:(冒号),
还有就是x = 1 和 y =2前面有四个空格的缩进。通过缩进，Python识别出这两个语句是隶属于if。

## Python基础07 函数
```
def square_sum(a,b):
	c = a**2 + b**2 # 这一句是函数内部进行的运算
	return c # 返回c的值，也就是输出的功能。Python的函数允许不返回值，也就是不用return。
# 当没有return, 或者return后面没有返回值时，函数将自动返回None
print(square_sum(3,4))

a = 1
def change_integer(a):
	a = a + 1
	return a
print(change_integer(a))
print(a)

b = [1,2,3]
def change_list(b):
	b[0] = b[0] + 1
	return b

print(change_list(b))
print(b)
# 对于基本数据类型的变量，变量传递给函数后，函数会在内存中复制一个新的变量，从而不影响原来的变量。（我们称此为值传递）
# 但是对于表来说，表传递给函数的是一个指针，指针指向序列在内存中的位置，在函数中对表的操作将在原有内存中进行，从而影响原有变量.(我们称此为指针传递）
```
## Python基础08 正则表达式

```
# 1、compile()
import re
pattern = re.compile('Hello')
# 使用Pattern匹配文本，获得匹配借结果，无法匹配时将返回None
mc = pattern.match('Hello Python!')

if mc:
    # 使用mc获得分组信息
    print(mc.group())
# 匹配完成后，需要通过 group() 或者 groups() 来返回结果，通常 group() 返回全部匹配的对象，groups() 返回元组

# 2.match()
# 匹配文本，无法匹配时将返回None
import re
mc = re.match("python",'i love python')
# match 是从字符串开头来进行匹配,所以匹配的结果是None
if mc is not None:
    print(mc.group())

# 3.search()
import re
mc = re.search('python', "i love python")
if mc is not None:
    print(mc.group())

# 4.findall()
import re
fstr = re.findall("python", 'i love python, python is beautiful')
print(fstr)
返回的结果是一个列表，如果匹配不到的话，则是返回一个空列表。

# 5.split()
import re
spstr = re.split('\d+', 'i am 10 years old, i love you forever')
# 按照模式串进行分割，其中模式串的意思是1个或者多个数字
print(spstr)

# 6.sub
# 按照模式串X进行替换，替换为新的字符串python
import re
substr = re.sub("X", 'python', 'i love X')
print(substr)
```

## Python基础09 面向对象的进一步拓展
```
# 调用类的其他信息
class Human(object):
	laugh = 'hahahaha'
	def show_laugh(self):
		print(self.laugh)
	def laugh_100th(self):
		for i in range(100):
			self.show_laugh()
li_lei = Human()
li_lei.show_laugh()
li_lei.laugh_100th()
# 这里有一个类属性laugh。在方法show_laugh()中，通过self.laugh，调用了该属性的值。还可以用相同的方式调用其它方法。
# 方法show_laugh()，在方法laugh_100th中()被调用。

# __init__()方法
# 如果你在类中定义了__init__()这个方法，创建对象时，Python会自动调用这个方法。这个过程也叫初始化。
class happyBird(Bird):
	def __init__(self, more_words):
		print('We are happy birds.', more_words)

summer = happyBird('Happy,happy,319')

# 对象的性质
# 在类属性的之外，可以给每个对象增添了各自特色的性质，从而能描述多样的世界。
class Human(object):
	def __init__(self, input_gender):
		self.gender = input_gender
	def printGender(self):
		print(self.gender)

li_lei = Human('male')
#这里，‘male’作为参数传递给__init__()方法的input_gender变量
print(li_lei.gender)
li_lei.printGender()
# 对象的性质也可以被其它方法调用，调用方法与类属性的调用相似，正如在printGender()方法中的调用
```

## Python基础10 反过头来看看
```
dir()  # 用来查询一个类或者对象所有属性
print(dir(list))
help()   # 用来查询的说明文档
print(help(list))
# 运算符是特殊方法
```

## Python基础11 Python的函数参数传递
```
a = 1 #不可变对象
def fun1(a):
    a = 2
fun1(a)
print(a)

a = [] # 可变对象
def fun2(a):
    a.append(1)
fun2(a)
print(a)
# [1]

# 类型是属于对象的，而不是变量。
# 而对象有两种,“可更改”（mutable）与“不可更改”（immutable）对象。
# 在python中，strings, tuples, 和numbers是不可更改的对象，而list,dict等则是可以修改的对象。
# (这就是这个问题的重点)
# 第二个例子中,函数内的引用指向的是可变对象,对它的操作就和定位了指针地址一样,在内存里进行修改.
```

## Python基础12. 两句话掌握Python最难知识点——元类
> 道生一，一生二，二生三，三生万物。
道 即是 type
一 即是 metaclass(元类，或者叫类生成器)
二 即是 class(类，或者叫实例生成器)
三 即是 instance(实例)
万物 即是 实例的各种属性与方法，我们平常使用python时，调用的就是它们。

```
# 创建一个Hello类，拥有属性say_hello -----二的起源
class Hello():
    def say_hello(self, name="world"):
        print("Hello, %s" % name)
# 从Hello类创建一个实例hello -----二生三
hello = Hello()
# 使用hello调用方法say_hello ----三生万物
hello.say_hello()

# 回到代码的第一行。class Hello其实是一个函数的“语义化简称”，只为了让代码更浅显易懂，它的另一个写法是：
# 假设我们有一个函数fn
def fn(self, name='world'):
    print("Hello, %s" % name)

#通过type创建Hello class -------神秘的道，这次我们直接从“道”生出了二
Hello = type("Hello",(object,), dict(say_hello = fn))

# 三个参数：
# 1.我是谁？在这里，我被命名为“Hello”
# 2.我从哪里来？我从父类object来
# 3.要到哪里去？在这里，我们将需要调用的方法和属性包含到一个字典里，再作为参数传入

hello = Hello()
hello.say_hello()

def Hello(object){
#class后声明“我是谁”
# 小括号内声明“我从哪里来”
# 中括号内声明“我要到哪里去”
    def say_hello(){
}
}

# 一般来说，元类均被命名后缀为Metalass。想象一下，我们需要一个可以自动打招呼的元类，它里面的类方法呢，有时需要say_Hello，有时需要say_Hi，有时又需要say_Sayolala，有时需要say_Nihao。
# 道生一
class SayMetaClass(type):
    def __new__(cls, name, bases, attrs):
        attrs['say_' + name] = lambda self, value, saying = name: print(saying + '，' + value + '!')
        # 根据类的名字创建一个类方法，然后又将类的名字作为默认参数saying，传到方法里。然后将hello方法调用时的传参作为value传进去，最后打印出来
        return type.__new__(cls, name, bases, attrs)
# 1、元类是由“type”衍生而出，所以父类需要传入type
# 2、元类的操作都在__new__中完成，它的第一个参数是将创建的类，之后的参数是：我是谁，从哪里来，到哪里去。返回值也是这三大永恒命题
# 一生二：创建类
class Hello(object, metaclass=SayMetaClass):
    pass
#二生三：创建实例
hello = Hello()
# 三生万物，调用实例的方法
hello.say_Hello("world!")
# 继续创建Sayolala,Nihao类
class Sayolala(object, metaclass=SayMetaClass):
    pass
sayolala = Sayolala()
sayolala.say_Sayolala("Japan!")
class Nihao(object, metaclass=SayMetaClass):
    pass
nihao = Nihao()
nihao.say_Nihao("China!")
```

```
class ListMetaClass(type):
    def __new__(cls, name, bases, attrs):
        attrs['add'] = lambda self, value: self.append(value)
        # 通过add方法将值绑定
        return type.__new__(cls, name, bases, attrs)

class Mylist(list,metaclass=ListMetaClass):
    # 不同点在于继承自list类
    pass

L = Mylist()
L.add(1)
L.add(2)
print(L)
# [1,2]
# 而普通的list没有add()方法
L2 = list()
L2.add(2)
print(L2)
# AttributeError: 'list' object has no attribute 'add'
```

## Python基础13.@staticmethod和@classmethod
python中其实有3个方法：静态方法，类方法，实例方法
```
def foo(x):
    print("executing foo(%s)" % x)

class A(object):
    def foo(self,x):
        print("executing foo(%s, %s)" % (self, x))

    @classmethod
    def class_foo(cls, x):
        print("executing class_foo(%s, %s)" % (cls, x))

    @staticmethod
    def static_foo(x):
        print("executing static_foo(%s)" % x)
```
> self和cls是对类或者实例的绑定，
对于实例方法,我们知道在类里每次定义方法的时候都需要绑定这个实例,就是foo(self, x),为什么要这么做呢?因为实例方法的调用离不开实例,我们需要把实例自己传给函数,调用的时候是这样的a.foo(x)(其实是foo(a, x)).
类方法一样,只不过它传递的是类而不是实例,A.class_foo(x).
对于静态方法其实和普通的方法一样,不需要对谁进行绑定,唯一的区别是调用的时候需要使用a.static_foo(x)或者A.static_foo(x)来调用.

##  Python基础14 类变量和实例变量
```
class Person:
     name = "aaa"
p1 = Person()
p2 = Person()
p1.name = "bbb"
# 实例调用了类变量，在实例的作用域把类变量的引用改变了，就变成了一个实例变量
print(p1.name)
# bbb
print(p2.name)
# aaa
print(Person.name)
# aaa

class Person:
    # if name is None:
        name = []

p1 = Person()
p2 = Person()
p1.name.append(1)
print(p1.name)
print(p2.name)
print(Person.name)
# [1]
# [1]
# [1]
# 因为person中name定义为可变数据类型，只在定义时初始化,调用p1的方法改变name实际上是改变了类的name值
```

## Python基础15 Python自省
运行时能够获得对象的类型.比如type(),dir(),getattr(),hasattr(),isinstance().

## Python基础16 字典推导式
```
d = {key: value for (key, value) in iterable}
```

## Python基础17.Python中单下划线和双下划线
```
class MyClass():
    def __init__(self):
        self.__superprivate = "Hello"
        self._semiprivate = ",world!"

mc = MyClass()
print(mc.__superprivate)
# AttributeError: 'MyClass' object has no attribute '__superprivate'
# __foo:这个有真正的意义:解析器用_classname__foo来代替这个名字,以区别和其他类相同的命名.
print(mc._semiprivate)
# _foo:一种约定,用来指定变量私有.程序员用来指定私有变量的一种方式.
print(mc.__dict__)
# {'_MyClass__superprivate': 'Helllo', '_semiprivate': ',world!'}
# __foo__:一种约定,Python内部的名字,用来区别其他用户自定义的命名,以防冲突
```

## Python基础18 字符串格式化:%和.format
```
name = (1,2,3)
# print("hi there %s" % name)
# TypeError: not all arguments converted during string formatting
# 如果name恰好是(1,2,3),它将会抛出一个TypeError异常
print('hi there %s' % (name,))
```

## Python基础19.迭代器和生成器和装饰器
http://taizilongxu.gitbooks.io/stackoverflow-about-python/content/1/README.html
### 19.1 python中的生成器
```
# 迭代器协议：对象需要提供next方法，它要么返回迭代器中的下一项，要么就引起一个StopIteration异常，以终止迭代。

# 生成器函数：
# 使用yield语句而不是return语句返回结果。
# yidld语句一次返回一个结果，在每一个结果中间，挂起函数的状态，以便下次从它离开的地方继续执行
def gensquares(N):
	for i in range(N):
		yield i ** 2

for item in gensquares(5):
	print(item,)

# 普通函数：
def gensquares(N):
	seq = []
	for i in range(N):
		seq.append(i ** 2)
	return seq

for item in gensquares(5):
	print(item,)

# 生成器表达式：
# 生成器返回按需产生结果的一个对象，而不是构建一个结果列表
squares = (x ** 2 for x in range(5))
print(type(squares))
print(next(squares))
print(next(squares))
print(list(squares))
# 生成器自动实现了迭代器协议

# 普通列表推导：一次产生所有结果
squares = [x ** 2 for x in range(5)]
print(squares)
[0, 1, 4, 9, 16]

# 生成器的好处：
# 1.延迟计算
sum([i for i in range(10000000))
sum(i for i in range(10000000))

# 2.提高代码可读性
# eg:求一段文字中，每个单词出现的起始位置。
def index_words(text):
	result = []
	if text:
		result.append(0)
	for index, letter in enumerate(text, 1):
		# enumerate(text, 1) 枚举 index 从 1 开始的 text 中的每个字母
		if letter == ' ':
			result.append(index)
	return result

print(index_words('hello world'))

def index_words(text):
	if text:
		yield 0
	for index, letter in enumerate(text, 1):
		if letter == ' ':
			yield index

print(index_words('hello world'))
# <generator object index_words at 0x00000000024CE5C8>

# 生成器的注意事项：生成器只能遍历一次
def get_num(there):
	for n in there:
		yield int(n)

gen = get_num([1, 2, 3, 4, 5])
sum = sum(gen) #生成器已经遍历
for n in gen:
	print(n / sum)
for n in gen:
	print(n / 15)
```
### 19.2 python 中的装饰器
装饰器功能：
1.引入日志
2.函数执行时间统计
3.执行函数前预备处理
4.执行函数后清理功能
5.权限校验等场景
6.缓存
```
# 多个装饰器
def makeBold(fn):
    def wrapped():
        return '<b>'+fn()+'</b>'
    return wrapped

def makeItalic(fn):
    def wrapped():
        return '<i>'+fn()+'</i>'
    return wrapped

@makeBold
def test1():
    return 'hello world - 1'

@makeItalic
def test2():
    return 'hello world - 2'

@makeBold
@makeItalic
def test3():
    return 'hello world - 3'
print(test1())
print(test2())
print(test3())

# 1.无参数的函数
from time import ctime,sleep
def timefun(func):
    def wrappedfunc():
        print('%s called at %s' % (func.__name__, ctime()))
        func()
    return wrappedfunc

@timefun
def foo():
    print('i am foo')

foo()
sleep(2)
foo()
# 上面代码理解装饰器执行行为可理解成
# foo = timefun(foo)
# foo先作为参数赋值给func后,
# foo接收指向timefun返回的wrappedfunc
# foo()
# 调用foo(),即等价调用wrappedfunc()
# 内部函数wrappedfunc被引用，所以外部函数的func变量(自由变量)并没有释放
# func里保存的是原foo函数对象

# 2.被装饰的函数有参数
from time import ctime,sleep

def timefun(func):
    def wrappedfunc(a, b):
        print('%s called at %s' % (func.__name__, ctime()))
        print(a,b)
        func(a,b)
    return wrappedfunc

@timefun
def foo(a, b):
    print(a+b)

foo(3, 5)
sleep(2)
foo(1, 2)

# 3.被装饰得的函数有不定长参数
from time import ctime,sleep

def timefun(func):
    def wrappedfunc(*args, **kwargs):
        print('%s called at %s' % (func.__name__, ctime()))
        print(args, kwargs)
        func(*args, **kwargs)
    return wrappedfunc

@timefun
def foo(a, b, c):
    print(a+b)

foo(1,2,3)
sleep(2)
foo(4,5,6)
# foo called at Sun Apr 22 21:11:19 2018
# (1, 2, 3) {}
# 3
# foo called at Sun Apr 22 21:11:21 2018
# (4, 5, 6) {}
# 9

# 4.装饰器的return
from time import ctime,sleep

def timefun(func):
    def wrappedfunc():
        print("%s called at %s" % (func.__name__, ctime()))
        func()
    return wrappedfunc

@timefun
def foo():
    print('i am foo')

@timefun
def getInfo():
    # print('i am getInfo')
    return '--hahahahah--'

foo()
sleep(2)
print(getInfo())
# foo called at Sun Apr 22 21:18:34 2018
# i am foo
# getInfo called at Sun Apr 22 21:18:36 2018
# None

# 修改装饰器为 return func
from time import ctime,sleep

def timefun(func):
    def wrappedfunc():
        print("%s called at %s" % (func.__name__, ctime()))
        return func()
    return wrappedfunc

@timefun
def foo():
    print('i am foo')

@timefun
def getInfo():
    # print('i am getInfo')
    return '--hahahahah--'

foo()
sleep(2)
print(getInfo())
# foo called at Sun Apr 22 21:20:31 2018
# i am foo
# getInfo called at Sun Apr 22 21:20:33 2018
# --hahahahah--

# 装饰器带参数，在原有装饰器的基础上，设置外部变量
from time import ctime,sleep

def timefun_arg(pre = "hello"):
    def timefun(func):
        def wrappedfunc():
            print('%s called at %s %s ' % (func.__name__, ctime(), pre))
            return func()
        return wrappedfunc
    return timefun

@timefun_arg('today')
def foo():
    print('i am foo')

@timefun_arg('python')
def too():
    print('i am too')

foo()
sleep(1)
too()
# foo called at Sun Apr 22 21:29:46 2018 today
# i am foo
# too called at Sun Apr 22 21:29:47 2018 python
# i am too

# 类装饰器（拓展）
# 装饰器函数其实是这样一个接口约束，它必须接受一个callable对象作为参数
# ，然后返回一个callable对象。在Python中一般callable对象都是函数，
# 但也有例外。只要某个对象重写了__call__()方法，那么这个对象就是callable的
class dog(object):
    def __init__(self, func):
        print("---初始化---")
        print("func name is %s" % func.__name__)
        self.__func = func
        # 在__init__方法中的func变量指向了test函数体
        # 需要一个实例属性来保存这个函数体的引用
    def __call__(self):
        print("---装饰器的功能---")
        self.__func()

@dog
def test():
    print("i am jingmao")
    # test函数相当于指向了用dog创建出来的实例对象
    # 当在使用test()进行调用时，就相当于Dog对象()的__call__方法

test()
# ---初始化---
# func name is test
# ---装饰器的功能---
# i am jingmao

@dog
def test2():
    print("i am zangao")

test2()
```
## Python基础20. *args and **kwargs
```
# 当你不确定你的函数要传多少参数时，你可以用*args.它可以传递任意数量的参数
def print_everything(*args):
    for count, thing in enumerate(args):
        print('{0}.{1}'.format(count, thing))

print_everything("apple", 'banana', "cabbage")
# 0.apple
# 1.banana
# 2.cabbage

# **kwargs允许你使用没有事先定义的参数名:
def table_things(**kwargs):
    for name, value in kwargs.items():
        print('{0} = {1}'.format(name, value))

print(table_things(apple = "fruit", cabbage = "vegetable", banana = "fruit"))
apple = fruit
cabbage = vegetable
banana = fruit
None

# 也可以混着用。命名参数首先获得参数值，然后其他的参数都传递给*args和**kwargs,命名参数在列表的最前端
def table_things(titlestring, **kwargs)

# 当调用函数时你也可以用*和**语法.例如:
def print_three_things(a, b, c):
    print('a = {0}, b = {1}, c = {2}'.format(a, b, c))

mylist = ['aardvark', 'baboon', 'cat']
print_three_things(*mylist)
# * 可以传递列表（或者元组）的每一项并把它们解包，注意必须与它们在函数里的参数相吻合。
```

## Python基础21 面向切面编程AOP和装饰器
> 装饰器的作用就是为已经存在的对象添加额外的功能。
http://stackoverflow.com/questions/739654/how-can-i-make-a-chain-of-function-decorators-in-python


## Python基础22 鸭子类型
> “当看到一只鸟走起来像鸭子、游泳起来像鸭子、叫起来也像鸭子，那么这只鸟就可以被称为鸭子。”
我们并不关心对象是什么类型，到底是不是鸭子，只关心行为。

> list.extend()方法中,我们并不关心它的参数是不是list,只要它是可迭代的,
所以它的参数可以是list/tuple/dict/字符串/生成器等.

> 鸭子类型在动态语言中经常使用，非常灵活，使得python不想java那样专门去弄一大堆的设计模式。


## Python基础23 Python中重载
函数重载主要是为了解决两个问题。
- 1.可变参数类型
函数功能相同，但是参数类型不同，python 如何处理？
答案是根本不需要处理，因为 python 可以接受任何类型的参数，如果函数的功能相同，那么不同的参数类型在 python 中很可能是相同的代码，没有必要做成两个不同函数。
- 2.可变参数个数
函数功能相同，但参数个数不同，python 如何处理？
大家知道，答案就是缺省参数。对那些缺少的参数设定为缺省参数即可解决问题。因为你假设函数功能相同，那么那些缺少的参数终归是需要用的

## Python基础24 新式类和旧式类

## Python基础25.\_\_new__ 和 \_\_init__ 的区别
- 1.__new__是一个静态方法，而__init__是一个实例方法
- 2.\_\_new__ 方法会返回一个创建的实例，而__init__什么都不返回
- 3.只有在__new__返回一个cls的实例时，后面的__init__才能被调用
- 4.当创建一个新实例时调用__new__,当初始化一个新实例时用__init__

\_\_metaclass\_\_是创建类时起作用.所以我们可以分别使用__metaclass__, __new__和__init__来分别在类创建, 实例创建和实例初始化的时候做一些小手脚.

## Python基础26.单例模式
- 1.使用__new__方法：
```
class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls,"_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)
        return cls._instance

class MyClass(Singleton):
    a = 1

print(MyClass.__dict__)
# {'__module__': '__main__', 'a': 1, '__doc__': None}
```

- 2.共享属性
```
# 创建实例时把所有实例的__dict__指向同一个字典，这样它们具有同样的属性和方法：
class Borg(object):
    _state = {}
    def __new__(cls, *args, **kwargs):
        ob = super(Borg, cls).__new__(cls, *args, **kwargs)
        ob.__dict__ = cls._state
        return ob
class MyClass2(Borg):
    a = 3
print(MyClass2.__dict__)
# {'__module__': '__main__', 'a': 3, '__doc__': None}
```

- 3.装饰器版本（最通用）
```
def singleton(cls, *args, **kwargs):
    instances = {}
    def get_instance():
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance()

@singleton
class Myclass3:
    pass
    # a = 1
print(Myclass3.__dict__)
```

- 4.作为python的模块是天然的单例模式
```
# mysingleton.py
class My_singleton(object):
    def foo(self):
        pass

my_singleton = My_singleton()

# to use
form mysingleton import my_singleton

my_singleton.foo()
```

## Python基础27 Python中的作用域
python中一个变量的作用域总是由在代码中被赋值的地方决定的。
当python遇到一个变量，会进行以下搜索：
本地作用域——当前作用域被嵌入的本地作用域——全局/模块作用域——内置作用域

## Python基础28.GIL线程全局锁
Global interpreter lock ,是python为了保证线程安全而采用的独立线程运行的限制

## Python基础29.协程
协程是进程和线程的升级版，进程和线程都面临着内核态和用户态的切换问题，因此耗费许多切换时间，
而协程就是用户自己控制切换的时机，不需要陷入系统的内核态。

## Python基础30.闭包
闭包是一种组织代码结构，提高了代码的可重复使用性
当一个内嵌函数引用其外部作用域的变量，我们就得到一个闭包。
满足以下几点：
- 1.必须有一个内嵌函数
- 2.内嵌函数必须引用外部函数中的变量
- 3.外部函数的返回值必须是内嵌函数

## Python基础31 lambda函数 匿名函数。

## Python基础32 .函数式编程
```
# http://coolshell.cn/articles/10822.html
a = [1, 2, 3, 4, 5, 6, 7]
b = filter(lambda x: x > 5, a)
print(list(b))
c = map(lambda x: x * 2,a)
print(list(c))
from functools import reduce
print(reduce(lambda x,y: x * y,range(1,4)))
```

## Python基础33.Python中的拷贝
```
import copy
a = [1, 2, 3, 4, ['a', 'b']]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
print(a, b, c, d)
# [1, 2, 3, 4, ['a', 'b']] [1, 2, 3, 4, ['a', 'b']] [1, 2, 3, 4, ['a', 'b']] [1, 2, 3, 4, ['a', 'b']]

a.append(5)
print(a, b, c, d)
# [1, 2, 3, 4, ['a', 'b'], 5] [1, 2, 3, 4, ['a', 'b'], 5] [1, 2, 3, 4, ['a', 'b']] [1, 2, 3, 4, ['a', 'b']]
a[4].append('c')
print(a, b, c, d)
# [1, 2, 3, 4, ['a', 'b', 'c'], 5] [1, 2, 3, 4, ['a', 'b', 'c'], 5] [1, 2, 3, 4, ['a', 'b', 'c']] [1, 2, 3, 4, ['a', 'b']]
```

## Python基础34.python的垃圾回收机制
Python GC 中主要使用引用计数来跟踪和回收垃圾.
在引用计数的基础上,通过"标记-清除"解决容器对象可能产生的循环引用问题,
通过"分代回收"以空间换时间的方法来提高垃圾回收效率.
- 1.引用计数
PyObject是每个对象都必有的内容,其中ob_refcnt就是做为引用计数.
当一个对象有新的引用时,它的ob\_refcnt就会增加,当引用它的对象被删除,它的ob_refcont就会减少.引用计数为0时,该对象的生命就结束了
- 2.标记-清除机制
基本思路是先按需分配,等到没有空闲内存的时候,从寄存器和程序栈的引用出发,遍历以对象为节点.以引用为边构成的图,把所有可以访问的对象打上标记,然后清理一遍内存空间,不然把所有没标记的对象释放.
- 3.分代技术
分代回收就是:将系统中的所有内存块根据其存活时间的划分为不同的集合,每个集合就成为一个"代"。
垃圾收集频率随着"代"的存活时间的增大而减少,存活时间通常利用经过几次回收来度量
Python默认定义了三代对象集合，索引数越大，对象存活时间越长。
当某些内存块M经过了3次垃圾收集的清洗之后还存活时，我们就将内存块M划到一个集合A中去，而新分配的内存都划分到集合B中去。
当垃圾收集开始工作时，大多数情况都只对集合B进行垃圾回收，而对集合A进行垃圾回收要隔相当长一段时间后才进行，这就使得垃圾收集机制需要处理的内存少了，效率自然就提高了。
在这个过程中，集合B中的某些内存块由于存活时间长而会被转移到集合A中，当然，集合A中实际上也存在一些垃圾，这些垃圾的回收会因为这种分代的机制而被延迟。

## Python基础35.python的list(L.sort()和sorted())
```
# 1.list生成
nestedList = [list(range(x)) for x in range(1, 4)]
print(nestedList)
list0 = range(1, 4)
print(list(list0))
newList = [y for x in nestedList for y in x]
print(newList)

# 2. L.sort()与sorted()函数
sorted(iterable, cmp=None, key=None, reverse=False)
L.sort(cmp=None, key=None, reverse=False)

A = [3, 6, 1, 5, 4, 2]
A.sort()
print(A)

student = [['Tom', 'A', 20],['Jack', 'C', 18],['Andy','B', 11]]
student.sort(key=lambda student: student[2])
print(student)# 按照年龄进行排序
student.sort(cmp=(lambda x, y: x[2] - y[2]))

L = ['cat', 'binary', 'big', 'dog']
print(sorted(L, key=lambda x:(x[0], x[1], x[2])))
```

## Python基础36.python 的is 和 ==
is是对比地址, == 是对比值
```
# is 比较的是两个实例对象是不是完全相同，它们是不是同一个对象，占用的内存地址是否相同。
# 比较的id是否相同，这id类似于人的身份证标识
# == 比较的是两个对象的内容是否相等，即内存地址可以不一样，内容一样就可以了。
# 默认会调用对象的 __eq__()方法。
a = ['I', 'Love', 'Python']
b = a
# a的引用复制给b，在内存中其实是指向了用一个对象
print(b is a)
# True
print(id(a))
# 39515784
print(id(b))
# 39515784
# 直接赋值都是赋值的引用

b = a[:]
# b通过切片操作重新分配了对象，但是值和a相同
print(b is a)
# False
print(id(a))
# 39515784
print(id(b))
# 37588104
print(b == a)
# True
print(b[0] is a[0])
# True
# 因为切片拷贝是浅拷贝，列表中的元素并未重新创建

# 通常，我们关注的是值，而不是内存地址，因此 Python 代码中 == 出现的频率比 is 高。但是什么时候用 is 呢？

# is 与 == 相比有一个比较大的优势，就是计算速度快，因为它不能重载，不用进行特殊的函数调用，少了函数调用的开销而直接比较两个整数 id
# 而 a == b 则是等同于a.__eq__(b)。继承自 object 的 __eq__ 方法比较两个对象的id，结果与 is
# 一样。但是多数Python的对象会覆盖object的 __eq__方法，而定义内容的相关比较，所以比较的是对象属性的值。
# 在变量和单例值之间比较时，应该使用 is。目前，最常使用 is 的地方是判断对象是不是 None。
# 下面是推荐的写法：
# a is None

# Python会对比较小的整数对象进行缓存，下次用的时候直接从缓存中获取，所以is 和 == 的结果可能相同
a = 1
b = 1
print(a == b)
print(b is a)
# True
# True

a = 257257
b = 257257
print(a == b)
print(a is b)
# Python仅仅对比较小的整数对象进行缓存（范围为范围[-5, 256]）缓存起来，而并非是所有整数对象。需要注意的是，这仅仅是在命令行中执行.
# 而在Pycharm或者保存为文件执行，结果是不一样的，这是因为解释器做了一部分优化。
```
- 1、is 比较两个对象的 id 值是否相等，是否指向同一个内存地址；
- 2、== 比较的是两个对象的内容是否相等，值是否相等；
- 3、小整数对象[-5,256]在全局解释器范围内被放入缓存供重复使用；
- 4、is 运算符比 == 效率高，在变量和None进行比较时，应该使用 is。

## Python基础37.read,readline,readlines
read 读取整个文件
readline 读取下一行,使用生成器方法
readlines 读取整个文件到下一个迭代器以供我们遍历

## Python基础38 Python2和3的区别
```
# python2.7
s = "中国zg"
print(s)
print(type(s))
# <class 'str'> str实例包含原始的8位值
# 操作系统使用自己的默认编码方式"utf-8"，将中国zg进行了编码，并把编码后的01串给了程序。然后这个字符串被赋值给了s。
e = s.encode("utf-8")
# 是将字符串s用utf-8进行编码，并将编码后的字符串赋值给e,
print(e)
# <class 'bytes'>
print(type(e))
# python程序就会用它自己默认的编码当作s的编码，进而来识别s中的内容。
# 这个默认的编码是ASCII，所以，它会用ASCII来解释这个01串，识别出字符串的内容，再将这个字符串转为utf-8编码。
# ascii编码里面没有0xe4，所以报错，并不是说encode("utf-8")错了。

# python2.7中先解码，再编码。相当于先将s中的01序列变为unicode的实例，在编码成”utf-8”
e = s.decode("utf-8")
# print(e)  # 以ascii编码不能输出
print(type(e))  # unicode类型 而unicode的实例，则包含Unicode字符
# print(isinstance(e, unicode))
e = e.encode("utf-8")
print(e)
print(type(e))
# 再编码成"utf-8"
# <type 'str'>
```

```
# 1. 字符串的编码形式
# 字符串的编码最一开始是ascii，使用8位二进制表示，因为英文就是编码的全部。
# 后来其他国家的语言加入进来，ascii就不够用了，所以一种万国码就出现了，它的名字就叫unicode，
# unicode编码对所有语言使用两个字节，部分汉语使用三个字节。
# 但是这就导致一个问题，就是unicode不仅不兼容ascii编码，而且会造成空间的浪费，
# 于是uft-8编码应运而生了，utf-8编码对英文使用一个字节的编码，由于这样的特点，很快得到全面的使用。
# 2. 字节码bytes
# python3中bytes用b’xxx’表示，其中的x可以用字符，也可以用ascii表示
# python3中的二进制文件（如文本文件）统一采用字节码读写。
# 2.1. 字节码的使用举例
b = b'asd\64'
print(b)
print(type(b))  # <class 'bytes'>
# 2.2. 字节码的修改
# 要修改bytes中的某一个字节必须要将其转换为bytearray以后才可以。
c = bytearray(b)
print(c)  # bytearray(b'asd4')
print(type(c))
c[0] = 110
print(c)  # bytearray(b'nsd4')
# 2.3. 字节码bytes与字符之间的关系
# 将表示二进制的bytes进行适当编码就可以变为字符了，比如utf-8或是gbk等等编码格式都可以。
# （所以我个人的理解就是：有utf-8格式的bytes，也有gbk格式的bytes等等）。
# 2.4. 字节码bytes与unicode编码的相互转换
```

```
python3的编码
# 1. python2与python3字符串编码的区别
# python3默认使用的是str类型对字符串编码，默认使用bytes操作二进制数据流，两者不能混淆！！
# Python3有两种表示字符序列的类型：bytes和str。前者的实例包含原始的8位值，后者的实例包含Unicode字符
# Python2也有两种表示字符序列的类型，分别叫做str和Unicode，与Python3不同的是，str实例包含原始的8位值；而unicode的实例，则包含Unicode字符。
# 2. bytes与str类型的相互转换
# str(unicode)类型是基准！要将str类型转化为bytes类型，使用encode()内置函数；反过来，使用decode()函数。
oath = '我爱你'
print(type(oath))
# <class 'str'>
# python3中str类型不能直接decode
oath = oath.encode('utf-8')
print(type(oath))
# <class 'bytes'>
print(oath)
# b'\xe6\x88\x91\xe7\x88\xb1\xe4\xbd\xa0'
oath = oath.decode("utf-8")
print(oath)
# 为了方便开发者的使用，可以编写两个helper函数，第一个是无论输入的是str还是bytes类型都输出str。
def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        values = bytes_or_str.decode("utf-8")
    else:
        values = bytes_or_str
    return values
# 还需要编写接受str或bytes，并总是返回bytes的方法：
def to_bytes(bytes_or_str):
    if isinstance(bytes_or_str, str):
        values = bytes_or_str.encode("utf-8")
    else:
        values = bytes_or_str
    return bytes_or_str
# 3. 文件的编码
with open('sun.bin','wb') as f:
    f.write(b'sui')
# 在open文件的时候用“wb”方式打开，即二进制写的方式，所以下面的write函数对象用的是bytes类型的b’sui’
# 这个时候如果使用f.write(‘sui’)就会出错的。

# 4. 网页的编码
import urllib.request
response = urllib.request.urlopen('http://www.baidu.com')
html = response.read()
print(html)
print(type(html))
# 所以要正常显示的话，就要使用decode()方法了。
print(html.decode('utf-8'))
```

# python语言特性
## 1. python中容易出错的三个问题
- 1.可变数据类型作为函数定义中的默认参数
```
def search_for_links(page, add_to = []):
    new_links = page.search_for_links()
    add_to.extend(new_links)
    return add_to
#应该改为：
def search_for_links(page, add_to = None):
    if not add_to:
        add_to = []
     new_links = page.search_for_links()
     add_to.extend(new_links)
     return add_to
def fn(var1, var2 = []):
    var2.append(var1) #下一次执行的时候，var2不初始化
    print(var2)
fn(3)
fn(4)
fn(5)

def fn_new(var1, var2 = None):
    if not var2:# var is None
        var2 = []
    var2.append(var1)
    print(var2)
fn_new(3)
fn_new(4)
fn_new(5)
# 对于不可变数据类型，比如元组、字符串、整型，是不需要考虑这种情况的
# 例如：
def func(message = "my message"):
    return message
```
- 2.可变数据类型作为类变量
```
class URLCatcher(object):
    urls = []
    def add_url(self, url):
        self.urls.append(url)
a = URLCatcher()
a.add_url("http://www.google.com")
b = URLCatcher()
b.add_url("http://www.bbc.co.hk")
print(b.urls)
# ['http://www.google.com', 'http://www.bbc.co.hk']
print(a.urls)
# ['http://www.google.com', 'http://www.bbc.co.hk']
# 创建类定义时，URL 列表将被实例化。该类所有的实例使用相同的列表。
# 我们希望每个对象有一个单独的存储，应该修改代码为：
class URLCatcher(object):
    def __init__(self):
        self.urls = []
    def add_url(self, url):
        self.urls.append(url)
a = URLCatcher()
a.add_url("http://www.google.com")
b = URLCatcher()
b.add_url("http://www.bbc.co.hk")
print(b.urls)
# ['http://www.bbc.co.hk']
print(a.urls)
# ['http://www.google.com']
# 当创建对象时，URL 列表被实例化。当我们实例化两个单独的对象时，它们将分别使用两个单独的列表。
```
- 3.可变的分配错误
```
a = {'1': "one",
     '2': "two"
     }
b = a
b['3'] = "three"
print(a)
# {'1': 'one', '2': 'two', '3': 'three'}
print(b)
# {'1': 'one', '2': 'two', '3': 'three'}
c = (2, 3)
d = c
d = (4, 5)
print(c)
# (2, 3)

# 如果我们真的需要复制一个列表进行处理，我们可以这样做：b = a[:]
# 这将遍历并复制列表中的每个对象的引用，并且把它放在一个新的列表中。
# 但是要注意：如果列表中的每个对象都是可变的(可迭代的)，我们将再次获得它们的引用，而不是完整的副本。
# 字典以相同的方式工作，并且你可以通过以下方式创建一个昂贵副本：b = a.copy()
# 再次说明，这会创建一个新的字典，指向原来存在的相同的条目
a = {'1': "one",
     '2': "two"
     }
b = a.copy()
b['3'] = "three"
print(a)
# {'1': 'one', '2': 'two'}
print(b)
# {'1': 'one', '2': 'two', '3': 'three'}
```
##2.  python2与python3编码的区别
```
# 在Python2中，字符串字面量对应于8位的字符或面向字节编码的字节字面量。
# 这些字符串的一个重要限制是它们无法完全地支持国际字符集和Unicode编码。
# 为了解决这种限制，Python2对Unicode数据使用了单独的字符串类型。
# 要输入Unicode字符串字面量，要在第一个引号前加上前最'u'。

# Python2中还有一种称为字节字面量的字符串类型，它是指一个已经编码的字符串字面量，
# 在Python2中字节字面量和普通字符串没有差别，
# 因为在Python2中普通字符串实际上就是已经编码(非Unicode)的字节字符串。

# 在Python3中，不必加入这个前缀字符，否则是语法错误，这是因为所有的字符串默认已经是Unicode编码了.
# 如果使用-U选项运行解释器，Python2会模拟这种行为(即所有字符串字面量将被作为Unicode字符对待，u前缀可以省略)。
# 在Python3中，字节字面量变成了与普通字符串不同的类型。

s = '今天练习1'
# c = s.encode('utf-8')
print(s)
# print(s.encode('utf-8').decode('utf-8'))
# print(c)

def main():
    print('今天练习2')

print(__name__)
# _name__的值为"__main__"

if __name__ == '__main__':
    main()
    print('今天练习3')
```

## 3. 判断文件是否存在

```
# 一. 使用OS模块
# 1.判断文件是否存在
import os
os.path.exits('test_file.txt')
os.path.exits('no_exist_file.txt')
# 2.判断文件夹是否存在
import os
os.path.exists('test_dir')
os.path.exists('no_exist_dir')
# 3.只检查文件
import os
os.path.isfile("test-data")
# 4.判断文件是否可做读写操作
#os.access(, )
# 判断文件是否存在
os.F_OK()
# 检查文件是否可读
os.R_OK
# 检查文件是否可以写入
os.W_OK
# 检查文件是否可以执行
os.X_OK

#判断文件路径是否存在和各种访问模式的权限
import os
if os.access("/file/path/foo.text", os.F_OK)
    print("Given file path is exist")

# 二、使用Try语句
try:
    f = open("test.file")
    f.close()
except FileNotFoundError:
    print("File is not found")
except PersmissionError:
    print("You don't have permission to access this file")

# 三.使用pathlib模块
# 使用pathlib需要先使用文件路径来创建path对象。
# 检查路径是否存在
import pathlib
path = pathlib.Path("path/file")
print(path.exist())
# 检查路径是否是文件
path = pathlib.Path("path/file")
print(path.is_file())
```

## 4. 改写函数访问二维数组

```
class Array:
	def __init__(self, lst):
	    self.__coll = lst
	
	def __repr__(self):
	    return '{!r}'.format(self.__coll)
	
	def __getitem__(self, key):
	    slice1, slice2 = key
	    row1 = slice1.start
	    row2 = slice1.stop
	    col1 = slice2.start
	    col2 = slice2.stop
	    return [self.__coll[r][col1:col2] for r in range(row1, row2)]

a = Array([["a", "b", "c", "d"],
           ["e", "f", "g", "h"],
           ["i", "j", "k", "l"],
           ["m", "n", "o", "p"],
           ["q", "r", "s", "t"],
           ["u", "v", "w", "x"]
           ])

print(a[1:5, 1:3])
```
## 4. python 中的 \*args 和 \*\* kw

```
# *args 非关键字参数，用于元组
# **kw 关键字参数，用于字典（kwyword）
# 1.*args

def tupleArgs(arg1, arg2 = "b", *arg3):
	print('arg1 : %s' % arg1)
	print('arg2 : %s' % arg2)
	for eachArgNum in range(len(arg3)):
		print('the %d in arg3 : %s' % (eachArgNum, arg3[eachArgNum]))

if __name__ == '__main__':
	tupleArgs("A")
	# arg1 : A
	# arg2 : b
	tupleArgs("23",'c')
	# arg1 :  23
	# arg2 :  c
	tupleArgs('23','a','lol','bupt')
	# arg1 : 23
	# arg2 : a
	# the 0 in arg3 : lol
	# the 1 in arg3 : bupt

# 2.**kw
def dictKw(kw1, kw2 = "b", **kw3):
	print('kw1 : %s' % kw1)
	print('kw2 : %s' % kw2)
	for eachKw in kw3:
		print('the %s -----> %s' % (eachKw, kw3[eachKw]))

if __name__ == '__main__':
	dictKw("A")
	# kw1 : A
	# kw2 : b
	dictKw("23",'c')
	# kw1 :  23
	# kw2 :  c
	dictKw('23','a',c = 'lol', d = 'bupt')
	# kw1 : 23
	# kw2 : a
	# the c -----> lol
	# the d -----> bupt
	dictKw('dianzi', c = 'lol', d = 'bupt', kw = 'guangxian')
	# kw1 : dianzi
	# kw2 : b
	# the c -----> lol
	# the d -----> bupt
	# the kw -----> guangxian
```
## 5. 用python实现单例模式
```
# 方法一：使用__new__()
# 将类的实例与一个类变量 _instance 关联起来
# 如果_instance为None 则创建实例，
# 否则返回 cls._instance
class Singleton(object):
    # _instance = None
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance

obj1 = Singleton()
obj2 = Singleton()

obj1.attr1 = '我是独一无二的attr'
print(obj1.attr1, obj2.attr1)
print(obj1 == obj2)
print(id(obj1), id(obj2))

# 方法二：使用装饰器。
# 定义一个装饰器singleton, 它返回一个内部函数 getinstance，
# 该函数判断某个类是否在字典instances 中。
# 如果不存在，则将 cls 作为 key，cls(args. *kw) 作为value 存到instances 中
# 如果存在，则返回 instances[cls]

from functools import wraps
def singleton(cls):
	instances = {}
	@wraps(cls)
	def getinstance(*arg, **kw):
		if cls  not in instances:
			instances[cls] = cls(*arg, **kw)
		return instances[cls]
	return getinstance

@singleton
class Singleton(object):
	pass

obj1 = Singleton()
obj2 = Singleton()

obj1.attr1 = '我是独二无二的attr'
print(obj1.attr1, obj2.attr1)
print(obj1 == obj2)
print(id(obj1), id(obj2))
```

