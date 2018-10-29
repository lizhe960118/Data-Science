Table of Contents
=================

         * [概念理解](#概念理解)
         * [branch分支](#branch分支)
         * [远程仓库](#远程仓库)
         * [历史记录](#历史记录)
         * [版本更新](#版本更新)
         * [状态查看](#状态查看)
         * [版本回退](#版本回退)

### 概念理解
工作区，缓冲区，HEAD(指向最后一次提交的结果）

命令|描述
---|---
git add . | 从工作区到缓冲区
git commmit -m "" | 从缓冲区提交到本地仓库HEAD中
git push origin master | 把当前主分支提交到远端仓库
git remote add origin <server> | 将本地仓库连接到某个远程服务器

### branch分支

命令 | 描述
---|---
git checkout -b br1 |创建并切换到br1
git checkout master | 切换回主分支
git branch -d br1 | 将分支br1删除
git branch -a | 查看所有分支
git branch -r -d origin/lzbr | 在本地删除分支lzbr
git push origin :lzbr | 在远程分支上删除lzbr
git merge br1 | 合并br1到当前分支


### 远程仓库
命令|描述
--|--
git push origin <branch> | 讲分支推送到远端仓库
git pull | 更新本地仓库与远程同步
git remote rm origin<repository> | 删除远程仓库

### 历史记录
命令|描述
--|--
git log --oneline | 了解本地仓库的历史记录


### 版本更新
命令|描述
--|--
git checkout --a.txt | 用HEAD中的a.txt替换本地改动a.txt
git fetch origin | 从服务器获取最新的历史版本
git reset --hard origin/master | 丢弃本地的所有改动，并将本地主分支指向最新版本

### 状态查看
命令|描述
--|--
git status | 查看当前状态

### 版本回退
命令|描述
--|--
git checkout HEAD^2 | 在HEAD有多个父提交的时候，指向第2个父提交
git checkout HEAD~2 | 是指向父提交的父提交

> git checkout HEAD^ 和 git reset --hard HEAD^的区别：  
> checkout只会移动HEAD指针，reset会将master重置到父提交，由于HEAD指向master分支，所以HEAD的引用值发生了改变。  

> reset两个作用：  
> 1.回退文件 git reset HEAD filename 将文件从暂存区回退到工作区  
> 2.回退版本 