# 模型特征
- server服务器。（输入tmux命令，开启一个服务器）
- session会话。（一个服务器，多个会话）
- window窗口。（一个会话，多个窗口）
- pane面板。（一个窗口，多个面板）

# 远程主机连接

```
# 一键启动远程主机的tmux
ssh -t username@server.com tmux
# 有且只有一个tmux会话，直接连接
ssh -t username@server.com tmux a
# 有多个会话，指定连接名为foo的会话 (t:target)
ssh -t username@server.com tmux a -t foo
# 去掉detach较小桌面，强制在较大桌面打开session
ssh -t username@server.com tmux a -d -t foo
```

# 配置文件
可以将tmux的**前缀键**重映射为**Ctrl+a**

# tmux（服务）

commend | 解释
--|--
tmux | 启动tmux会话
tmux new -s mysession | 创建名为mysession的新会话
tmux a(at/attach) | 重新连接当前仅有的一个会话
tmux a -t mysession | 重新连接名为mysession的会话
tmux ls | 显示所有会话
tmux kill -session -t mysession | 关闭名为mysession的会话

# session（会话）
> 定义前缀键 -> **Ctrl + a**

当前我们在一个会话窗口里面
- 按下 前缀键 
- 放开 前缀键
- 按下 命令键**S**

commend（**S**） | 解释
--|--
d(detach) | 从会话中跳出，会话在后台运行
s | 显示会话

# window（窗口）
commend（**S**） | 解释
--|--
c(create) | 创建新窗口
w | 显示窗口列表
& | 关闭当前窗口

# pane（面板）
commend（**S**） | 解释
--|--
% | 垂直分割面板
" | 水平分割面板
x | 关闭当前面板
q | 短暂显示面板编号
o | 循环移动当前焦点

# 复制模式
commend（**S**） | 解释
--|--
[ | 进入复制模式
] | 进行粘贴

commend | 解释
--|--
space | 开始选中文本
esc | clear selection
enter | copy

# referrence
https://blog.csdn.net/simple_the_best/article/details/51360778