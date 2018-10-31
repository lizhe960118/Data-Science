# conda设置python虚拟环境

## 查看哪些python版本可以用：

```
conda search '^python$'
```

## 创建虚拟环境

```
conda create --name py2-env python=2.7
```

## 激活虚拟环境

```
source activate py2-env
```

## 查看相关环境

```
conda info --envs
```

## 删除虚拟环境

```
conda remove --name py2-env --all
```

## 指定目录操作

### 安装虚拟环境到指定目录

```
conda create --prefix=anaconda/python36/py36 python=3.6
```

### 激活环境

```
activate anaconda/python36/py36 
```

### 删除环境

```
conda remove --prefix=anaconda/python36/py36 --all
```