Table of Contents
=================

   * [结合实验楼的linux实验进行学习](#结合实验楼的linux实验进行学习)
      * [实验3：用户及文件权限管理](#实验3用户及文件权限管理)
      * [实验4：linux目录结构及文件基本操作](#实验4linux目录结构及文件基本操作)
      * [实验5 环境变量与文件查找](#实验5-环境变量与文件查找)
         * [linux环境变量](#linux环境变量)
      * [实验6 文件打包与压缩](#实验6-文件打包与压缩)
      * [实验7 文件系统操作与磁盘管理](#实验7-文件系统操作与磁盘管理)
      * [实验8：linux下的帮助命令](#实验8linux下的帮助命令)
      * [实验9：Linux任务计划crontab](#实验9linux任务计划crontab)
      * [实验10：命令执行顺序控制与管道](#实验10命令执行顺序控制与管道)
         * [1)简单的顺序执行：使用“；”分隔](#1简单的顺序执行使用分隔)
         * [2)有选择的执行：](#2有选择的执行)
         * [3) | 管道：将前一个命令的输出作为下一个命令的输入](#3--管道将前一个命令的输出作为下一个命令的输入)
         * [4）cut:打印每一行的某一字段](#4cut打印每一行的某一字段)
         * [5）grep:在文本或者stdin中查找匹配字符串](#5grep在文本或者stdin中查找匹配字符串)
         * [6）wc:简单小巧的计数工具](#6wc简单小巧的计数工具)
         * [7）sort:排序](#7sort排序)
         * [8) uniq:去重（只能去重连续的行，不能全文去重）](#8-uniq去重只能去重连续的行不能全文去重)
      * [实验11：简单的文本处理](#实验11简单的文本处理)
         * [1）tr 删除一段文本信息中的某些文字](#1tr-删除一段文本信息中的某些文字)
         * [2）col 将Tab换为等数量的空格键](#2col-将tab换为等数量的空格键)
         * [3）join 将两个文件中包含相同内容的一行合并在一起](#3join-将两个文件中包含相同内容的一行合并在一起)
         * [4）paste 在不对比数据的情况下，简单地将多个文件合并到一起](#4-paste-在不对比数据的情况下简单地将多个文件合并到一起)
      * [实验12：数据流重定向](#实验12数据流重定向)
      * [实验13：正则表达式基础](#实验13正则表达式基础)
         * [1）Regular Expression](#1regular-expression)
         * [2）sed编辑器的使用](#2sed编辑器的使用)
         * [3）awk:文本处理语言](#3awk文本处理语言)
      * [实验14 linux下的软件安装](#实验14-linux下的软件安装)
      * [实验15 Linux进程概念：](#实验15-linux进程概念)
      * [实验16 linux进程管理](#实验16-linux进程管理)
      * [实验17 linux日志系统](#实验17-linux日志系统)

# 结合实验楼的linux实验进行学习

## 实验3：用户及文件权限管理
查看文件权限：

命令 | 描述
---|---
ls -l | 显示文件具体信息(文件类型和权限 链接数 所有者 所属用户组 文件大小 最后修改时间 文件名)
> (d rwx r-x r-x 
> d 目录 - 普通文件 l 软链接=快捷方式 
> rwx所有者权限 r-x所属用户组权限 r-x 其他用户权限)

命令 | 描述
---|---
ls  | -lh 查看文件大小
ls -A | 显示出隐藏文件
ls -dl <目录名> | 显示某一目录的完整属性
ls -AsSh | 小s显示文件大小，大S为按文件大小排序
pwd | 查看当前所在目录
sudo chown shiyanlou hanchuan | 变更文件所有者（将hanchuan的所有者改为shiyanlou）
chmod 600 hanchuan | 修改文件权限  （-rw-------  所有者读写 用户组不能操作 其他用户不能操作）
chmod go-rw hanchuan | g o u 表示用户组group，其他others，所有者user

useradd 和 adduser的区别  
useradd只创建用户 需要用passwd lilei来为用户设置密码  
adduser 创建用户，创建目录，创建密码，像是一个程序
类似于deluser

## 实验4：linux目录结构及文件基本操作

> 从逻辑上说：linux的磁盘是挂载在目录上的。  
> 每一个目录不仅能使用本地磁盘分区的文件系统，而且也能使用网络上的文件系统  
> FHS标准（Filesystem Hierarchy Standatd）  
> 文件系统层次结构标准：定义了两层规范：  
> 1：/ 下面的目录  
> 2：/user 和 /var 两个子目录来定义  

命令 | 描述
---|---
sudo apt-get install tree | 安装软件包

命令 | 描述
---|---
cd .. | 进入上一级目录
cd ~  | 进入用户的home目录
cd /home/shiyanlou | 绝对路径进入

文件操作 | 命令
---|---
新建空白文件 | touch test
新建目录 | mkdir mydir
建立多级目录 | mkdir -p father/son/grandson
复制文件到指定目录 | cp test father/son/grandson
复制目录 | cp -r father family
删除文件 | rm test
忽略提示删除只读文件 | rm -f test
删除目录 | rm -rf famliy
移动文件 | mv file Documents
重命名文件 | mv oldfile newfile 
批量重命名 | rename
批量创建文件 | touch file{1..5}.txt
批量删除文件 | rm file{1..5}.txt
查看文件 | cat file
显示行数 | cat -n file
分页查看 | more file less file 
查看文件前10行 | head file
后10行 | tail file
后一行 | tail -n 1 file
查看文件类型 | file file_name

## 实验5 环境变量与文件查找
### linux环境变量  
set>env>export  
env 显示与当前用户相关的环境变量，还可以让命令在指定环境中运行。  
环境变量的作用域比自定义变量的要大。  

三种变量：  
Shell进程私有用户自定义变量  
Shell本身内建的变量  
从自定义变量导出的环境变量  
/etc/bashrc 存放shell变量  
/etc/profile 存放环境变量  


环境变量相关操作描述| 命令
---|---
PATH中保存了Shell中执行命令的搜索路径 |echo $PATH 
gcc 生成可执行文件（./脚本 执行脚本) | gcc -o hello_world hello_world.c 
添加路径到环境变量（这里使用绝对路径）| PATH = $PATH:/home/shiyanlou/mybin 
使用 >> 添加内容 |echo "PATH=$PATH:/home/shiyanlou/mybin" >> .zshrc
使用 > 是重写内容 | echo "PATH=$PATH:/home/shiyanlou/mybin" > .zshrc
修改环境变量 | path = $PATH
从尾向前开始匹配，删除符合匹配字串的最短数据 | path=$(path%/home/shiyanlou/mybin} 
从前向后开始匹配，删除符合匹配字串的最短数据 | path=$(path#/home/shiyanlou/mybin} 
将第一个匹配的字串替换为新字串 | path=${path/old_string/new_string)
将所有匹配的字串替换为新字串 | path=${path//old_string/new_string)
删除环境变量 | unset temp
让环境变量立刻生效 | source .zshrc => source = . => . ./.zshrc 必须指定完整的绝对或相对路径名

搜索文件相关操作描述 | 命令
---|---
**简单快速**|whereis 
**快而全**|locate(通过“/var/lib/mlocate/locate.db"数据库查找，首先要updatedb.)
找到某一照片|locate /usr/share/\.*jpg
会自动递归子目录查找 |locate /etc/sh 
**小而精** | which
通常用which确定是否安装了某个软件，只从PATH环境变量指定的路径中搜索| which man
**精而细 ** | find
基本用法 | find path option action 
查找配置文件 |sudo find /etc/ -name interfaces
最近n天内修改过的文件 | find ~ -mtime n
找到比指定文件夹新的文件 | find ~ -newer /home/shiyanlou/Code 

## 实验6 文件打包与压缩
zip:
 命令|描述
---|---
zip file_name.zip file | 打包文件
zip -r -q -o file_name.zip |打包目录
-r | 递归打包包含子目录的全部内容
-q | 使用安静模式，不向屏幕打印信息
-o | 输出文件，后面紧跟着打包输出文件名
du -h -d file_name.zip | 不解压查看文件 
-h | human_readable
-d | max_depth 查看文件深度
zip -r -e -o file_name.zip content | 加密文件
unzip file_name.zip |解压到当前目录
unzip -q file_name.zip -d content | 解压到指定目录(可以不存在)
unzip -l file_name.zip | 不解压，只查看压缩包内容

tar:
命令|描述
---|---
tar -cf file_name.tar file | 打包文件 
tar -cf file_name.tar content | 打包文件 
tar -xf file_name.tar | 解压到当前目录
tar -xf file_name.tar -C content | 解压到指定目录（必须已存在）
tar -tf file_name.tar | 只查看不解压
tar -czf file_name.tar.gz content | 使用tar创建不同压缩格式的文件
tar -xzf file_name.tar.gz content | 使用tar创建不同压缩格式的文件


## 实验7 文件系统操作与磁盘管理

df => display file system disk 显示磁盘容量
命令|描述
---|---
df -h | 显示磁盘容量

/dev/sda2 主硬盘分区 2表示分区号 a表示第几块硬盘  
挂载 将目录与硬盘对应起来，硬盘可移动到某一目录  
将文件系统挂载到目录上，使得文件系统中的文件可以被访问 

du => display space usage 显示当前目录文件大小  
命令|描述
---|---
du -h -d 1 ~  | 1表示查看2级目录信息（从0开始）

## 实验8：linux下的帮助命令
命令|描述
---|---
type 命令 | 查看命令属性
内建命令 | shell程序的一部分，执行速度快。
外部命令|linux系统实用程序的一部分，使用时才将其调入内存。


常用的帮助命令：  
命令|描述
---|---
help | help 内建命令 外部命令 --help
man | man 命令名
info |	info 命令名 

## 实验9：Linux任务计划crontab

命令|描述
---|---
启动rsyslog, 以便通过日志信息了解任务是否被执行。|sudo apt-get install -y rsyslog<br>   sudo service rsyslog start
手动启动crontab|sudo cron -f &
*使用crontab*<br>添加任务| crontab -e
*任务格式*<br>时间<br>任务 | minute hour “day of month” month “day of week”<br>user-name command to be executed
查看任务| crontab -l
查看cron是否成功启动| ps -aux \|grep cron
查看日志中的信息反馈| sudo tail -f /var/log/syslog
删除任务 | crontab -r

## 实验10：命令执行顺序控制与管道
### 1)简单的顺序执行：使用“；”分隔  
```
sudo apt-get update;sudo apt-get install some-tool;some-tool
```
### 2)有选择的执行：  
&&:若前面的命名为真（返回0），则执行后面的命令  
||:若前面的命令为假（返回1），则执行后面的命令  
echo $? 查看命令是否为真  

### 3) | 管道：将前一个命令的输出作为下一个命令的输入  

### 4）cut:打印每一行的某一字段  
命令|描述
---|---
cut /etc/passwd -d ':' -f 1,6 | 打印每一行的某一字段 
-d ':' | 以‘:'为分隔符 分割字符串
-f 1,6 | 第1个字段和第6个字段
cut /etc/passwd -c -5 | 打印前五个字符（包含第五个）
cut /etc/passwd -c 5-| 打印前五个之后的字符（包含第5个）
cut /etc/passwd -c 5 | 打印第5个字符
cut /etc/passwd -c 2-5 | 打印2到5之间的字符（包含第五个）

### 5）grep:在文本或者stdin中查找匹配字符串  
命令|描述
---|---
grep -rnI "shiyanlou" ~ | grep [命令选项] 用于匹配的表达式 [文件]
-r | 在目录中递归到子目录
-n | 打印文本中的行号
-I | 忽略二进制文件
export \| grep " *yanlou$" | 输出export中的文本，使用grep匹配其中的以”yanlou"结尾的字符串

### 6）wc:简单小巧的计数工具
命令|描述
---|---
wc /etc/passwd |统计并输出文件中行、单词和字节数
-l | 行数
-w | 单词数
-c | 字节数
-m | 字符数

### 7）sort:排序
默认为字典排序。
命令|描述
---|---
-r|反转排序
-n|按照字符串表示的数字值来排序
-t|指定分隔符，-k,指定字段进行排序
cat /etc/passwd | sort -t ':' -k 3 | 联合使用

### 8) uniq:去重（只能去重连续的行，不能全文去重）
全文去重：先排序，再去重。
命令|描述
---|---
history \| cut -c 8- \| cut -d ' ' -f 1 \| sort \| uniq |  先排序，再去重
-dc | 输出重复过的行及其重复次数
-D | 输出所有重复的行

## 实验11：简单的文本处理
### 1）tr 删除一段文本信息中的某些文字
命令|描述
---|---
tr [option] set1 [set2] | 删除一段文本信息中的某些文字
-d | 删除和set1匹配的字符
echo 'hello shiyanlou' \| tr -d 'olh' | 联合使用
-s | 去除set2 中连续的set1
echo 'hello' \| tr -s 'l' | 联合使用
tr [a-z] [A-Z] set2 | 将set2中的小写全部转换为大写

### 2）col 将Tab换为等数量的空格键  
命令|描述
---|---
cat /etc/protocols \| col -x | 将tab转换为空格
col -h | (默认选项) 将空格转换为tab

### 3）join 将两个文件中包含相同内容的一行合并在一起
命令|描述
---|---
join | 将两个文件中包含相同内容的一行合并在一起
-t | 指定分隔符 
-1 | 指明第一个文件要用哪一个字段来对比，默认对比第一个字段
-2 | 指明第二个文件要用哪一个字段来对比
### 4）paste 在不对比数据的情况下，简单地将多个文件合并到一起
命令|描述
---|---
paste [option] file .. | 在不对比数据的情况下，简单地将多个文件合并到一起
-d | 指定合并的分隔符，默认为tab
-s | 不合并到一行，每个文件为一行

## 实验12：数据流重定向
命令|描述
---|---
\> | 将标准输出的数据导向一个文件
\>> | 将标准输出的数据重定向到一个文件

文件描述符：  
0 /dev/stdin 标准输入  
1 /dev/stdout 标准输出  
2 /dev/stderr 标准错误  

命令|描述
---|---
2>&1 | 将标准错误重定向到标准输出
tee | 将输出重定向到文件并打印到终端
echo 'hello shiyanlou' \| tee hello | 联合使用
exec 1>somefile | 使用exec替换当前进程的重定向，将标准输出重定向到一个文件
cd /dec/fd/;ls -Al;cd - | 查看当前shell进程中打开的文件描述符
exec 3>&- | 关闭文件描述符
/dev/null | 完全屏蔽命令的输出：<br>通常用于丢弃不需要的输出流，或者作为输入流的空文件。
xargs| 将参数列表转换成小块分段传递给其他命令，以免参数列表过长
cut -d ':' -f 1 < /etc/passwd \|sort\| xargs echo | 联合使用

## 实验13：正则表达式基础
### 1）Regular Expression 
命令|描述
---|---
gr(e\|a)p | ()表示优先级 \|表选择
数量限定 | 
\* | 不限
\+ | 至少一次
？| 至多一次
^ | 匹配字符串开头 
$ | 匹配字符串结尾
grep | 特殊符号说明
[[:alnum:]] |  0-9 a-z A-Z
[[:alpha:]] | A-Z a-z
[[:digit:]] | 0-9
	
注意转义字符的使用：\

### 2）sed编辑器的使用 
命令|描述
---|---
sed -i 's/sad/happy/g' test | 在全局范围内将test中的sad转化为happy
nl passwd | sed -n '2,5p' | 打印2-5行
nl passwd | sed -n '1~2p' | 打印奇数行
sed -n 's/shiyanlou/hehe/gp' passwd | 将输入文本中的’shiyanlou’全局替换为‘hehe'，并只打印替换的那一行

### 3）awk:文本处理语言  

## 实验14 linux下的软件安装
命令|描述
---|---
在线安装 | sudo apt-get install softname <br> source ~/.zshrc
重新安装| sudo apt-get --reinstall install softname
更新软件源 | sudo apt-get update
升级没有依赖问题的软件包 |sudo apt-get upgrade
升级并解决依赖问题 | sudo apt-get dist-upgrade
修复依赖关系的安装 | sudo apt-get -f install
卸载软件 | sudo apt-get remove softname
软件搜索 | sudo apt-cache search softname

从磁盘安装deb软件包
命令|描述
---|---
dpkg | Debian Package
-I | 显示deb包文件的信息
-i | 安装指定deb包
-s | 显示已安装软件的信息
-S | 搜索已安装的软件包
-r | 移除已安装的软件包
-R | 加目录名，安装改目录下的所有deb软件包
-L | 显示已安装软件包的目录信息

从二进制软件包安装  
将从网络上下载的二进制包解压后放到合适的目录，然后将包含可执行的主程序文件的目录添加进PATH环境变量即可。

从源代码编译安装  

## 实验15 Linux进程概念：
笔记|描述
---|---
进程的特性|动态性，并发性，独立性，异步性，结构性  
以服务对象分类|用户进程，系统进程  
以服务类型分类|	交互进程，批处理进程，守护进程

命令|描述
---|---
fork()| 为当前进程创建一个子进程<br>进程是进程组的成员：
PGID | (process group ID) 
jobs | 查看被停止放置在后台的工作
kill -signal %jobnumber | 重新操作工作
-1 | restart  
-2 | ctrl+c  
-9 | 强制终止该任务  
-15 | 正常的方式终止该任务

## 实验16 linux进程管理
top | 实时查看进程状态
---|---
PID| 进程id
USER| 该进程的所属用户
PR |该进程执行的优先级（priority 0-139）
NI |该进程的nice值（静态优先级 -20-19）
VIRT| 使用的虚拟内存总数
RES |使用的物理内存总数
SHR |共享内存的大小
S |进程的状态（S =sleep，R = running， Z = zombie）
%CPU | cpu的利用率
%MEM | 内存的利用率
TIME+ |活跃的总时间
COMMAND | 运行的名字
交互命令|  
q |退出程序
l |显示平均负载和启动时间
P |按照CPU使用百分比大小进行排序
M |根据驻留内存进行排序
k |输入PID 终止进程


ps | 静态查看当前进程的信息
---|---
ps aux | 查看当前进程
ps axjf | 树状打印
显示 |
PPID | 父进程的id
SID | session 的id
TPGID |前台进程组的id
TTY |终端id
STAT | 进程状态
R | 运行中 
S | 等待调用 
D | 不可终端睡眠
T | 暂停后者跟踪状态 
X | 即将被撤销 
Z | 僵尸进程
自定义参数显示|	ps -afxo user,ppid,pid,pgid,command
pstree | 查看活跃进程的树形结构
pstree -up |
-u | 同时列出process的PID
-p | 同时列出process所属账号名称
kill | 使用kill直接操作pid
kill -9 1608 |  
renice| 修改已经存在的进程的优先级
renice -5 pid | 

## 实验17 linux日志系统
日志存放在 /var/log中  

```
alternatives.log 系统额一些更新替代信息记录
apt/history.log 使用apt-get安装卸载软件的信息记录
```

日志收集工具：rsyslog  
日志文件管理工具：logrotate
