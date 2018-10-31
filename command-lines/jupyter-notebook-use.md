# jupyter-notebook-use

## 远程服务器 install python,pip

```
apt-get remove python-pip python3-pip
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
python2 get-pip.py
```
## 安装jupyter

```
sudo pip install jupyter
```

## 启动jupyter

```
jupyter notebook --no-browser --port=8889 --ip=127.0.0.1 --allow-root
```

## 远程jupyter与本地端口绑定

```
ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host
```

## 两种模式
- 编辑模式（Enter进入）
- 命令模式（ESC进入）

## 编辑模式
command | 描述
--|--
Tab | 代码补全或缩进
Shift-Tab | 提示
Ctrl-A | 全选
Ctrl-Z | 复原
Ctrl-S | 文件存盘
Esc | 进入命令模式
Ctrl-M | 进入命令模式
Shift-Enter | 运行本单元，选中下一单元
Ctrl-Enter | 运行本单元
Alt-Enter | 运行本单元，在下面插入一单元
Up | 光标上移或转入上一单元
Down | 光标下移或转入下一单元

## 命令模式
command | 描述
--|--
Y | 单元转入代码状态
M | 单元转入markdown状态
R | 单元转入raw状态
A | 在上方插入新单元
B | 在下方插入新单元
X | 剪切选中的单元
C | 复制选中的单元
shift-V | 粘贴到上方单元
V | 粘贴到下方单元
1 | 设定 1 级标题
2 | 设定 2 级标题
Up | 选中上方单元
Down | 选中下方单元
Z | 恢复删除的最后一个单元
D,D | 删除选中的单元
Shift-M | 合并选中的单元
Ctrl-S | 文件存盘
Esc : 关闭页面
Q : 关闭页面
H : 显示快捷键帮助
I,I : 中断Notebook内核
0,0 : 重启Notebook内核
Shift : 忽略
Shift-Space : 向上滚动
Space : 向下滚动

## reference
https://blog.csdn.net/lawme/article/details/51034543