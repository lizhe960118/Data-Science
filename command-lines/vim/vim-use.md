Table of Contents
=================

   * [vim-use](#vim-use)
      * [1.存活](#1存活)
         * [1.安装vim](#1安装vim)
         * [2.启动vim](#2启动vim)
      * [2.感觉良好](#2感觉良好)
         * [各种插入模式](#各种插入模式)
         * [简单移动光标](#简单移动光标)
         * [拷贝粘贴](#拷贝粘贴)
         * [替换](#替换)
         * [撤销与恢复](#撤销与恢复)
         * [文件操作](#文件操作)
      * [3.更好 更强 更快](#3更好-更强-更快)
         * [更好](#更好)
         * [更强](#更强)
         * [更快](#更快)
      * [4.使用VIM的超能力](#4使用vim的超能力)
         * [0   I-- [ESC]](#0---i---esc)
         * [多行注释](#多行注释)
         * [取消多行注释](#取消多行注释)
         * [为每一行行末添加something](#为每一行行末添加something)
         * [分屏](#分屏)
      * [打造IDE](#打造ide)
      * [reference](#reference)

# vim-use

## 1.存活
### 1.安装vim
### 2.启动vim
vim在Normal模式下

command(**Normal**) | 描述
--|--
i | 进入Insert模式
x | 删除当前光标所在的一个字符
:wq | 存盘（w）+退出（q）
dd | 删除当前行
p | 粘贴剪切板
:help<command>| 显示相关命令的使用
hjkl | 左下上右

笔记：  
- Ctrl-S --> <C-S>
- :q + <enter> --> :q<enter>

## 2.感觉良好
### 各种插入模式
command(**Normal**) | 描述
--|--
a | 在光标后插入
o | 在当前行后插入新行
O | 在当前行前插入新行

### 简单移动光标
command(**Normal**) | 描述
--|--
0 | 移动光标到当前行开头
$ | 到本行行尾
/something | 搜索字符串something（n可以跳到下一个）

### 拷贝粘贴
command(**Normal**) | 描述
--|--
p | 在当前位置之后
P | 在当前位置之前
yy | 拷贝当前行
2yy | 复制两行

### 替换
command(**Normal**) | 描述
--|--
:s/abc/123 | 将当前行中第一个abc替换为123
:s/abc/123/g | 将当前行中所有abc替换为123
:%s/abc/123 | 将所有行中第一个abc替换为123
:%s/abc/123/g | 将所有行中abc替换为123
:10,30s/abc/123/g | 将10-30行中abc替换为123
### 撤销与恢复
command(**Normal**) | 描述
--|--
u | undo
<C-r> | redo

### 文件操作
command(**Normal**) | 描述
--|--
:e <path/to/file>|打开一个文件
:w | 存盘
:saveas <path/to/file> | 另存为
:wq(:x, ZZ) | 保存退出
:q! | 退出不保存
:bn | 切换到下一个文件
:bp | 切换到上一个文件
 
## 3.更好 更强 更快
### 更好
command(**Normal**) | 描述
--|--
2dd | 删除2行
3p | 粘贴文本3次
100idesu [ESC] | 写下"desu"100次
. | 重复上一次命令
3\. |重复上一次命令3次
6,9 co 12 | 复制第6行和第9行之间内容到第12行后面
5,9 de | 删除多行

### 更强
command(**Normal**) | 描述
--|--
:23 | 到当前行下23行
23G | 到23行
gg | 到第一行
G | 到最后一行
\* | 移动下一个匹配单词
\# | 移动上一个匹配单词

### 更快
<start position><command><end position>

command(**Normal**) | 描述
--|--
0y$ |
0 | 当前行头
y | 从这里拷贝
$ | 当前行尾


## 4.使用VIM的超能力

### 0 <C-v> <C-d> I-- [ESC]

command | 描述
--|--
0 | 当前行头
<C-v> | 进入块操作
<C-d> | 向下移动
I-- | 插入'--'
[ESC] | 为每一行生效

### 多行注释
command | 描述
--|--
0 | 当前行头
<C-v> | 进入块操作
<C-d> | 向下移动
I-- | 插入'//'
[ESC] | 为每一行生效

### 取消多行注释
command | 描述
--|--
<C-v> | 进入块操作
l | 横向选中列的个数 '//'需要选中两列
j | 选中注释符号
d | 全部取消注释

### 为每一行行末添加something
command | 描述
--|--
0 | 当前行头
<C-v> | 进入块操作
<C-d> | 向下移动
$ | 到行最后
A | 输入模式
something |输入'something'
[ESC] | 为每一行生效

### 分屏
command(**Normal**) | 描述
--|--
:sp(:split) | 创建水平分屏
:vsp(:vsplit) | 创建垂直分屏

## 打造IDE
- 系统级配置文件： /etc/vim/vimrc
- 用户级配置文件： ~/.vim/vimrc
修改vimrc 或者 .vimrc

## reference
https://www.cnblogs.com/cloverclt/p/5553103.html