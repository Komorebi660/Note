# Ubuntu常用命令

- [Ubuntu常用命令](#ubuntu常用命令)
  - [查询文件占用空间](#查询文件占用空间)
  - [删除文件夹下所有指定文件名的文件](#删除文件夹下所有指定文件名的文件)
  - [读取文件](#读取文件)
  - [文件合并](#文件合并)
  - [for循环命令](#for循环命令)
  - [查询系统版本以及硬件信息](#查询系统版本以及硬件信息)
  - [进程管理](#进程管理)
  - [文件校验](#文件校验)
  - [压缩](#压缩)
  - [Vim](#vim)
  - [Git](#git)
  - [其他](#其他)
  - [关于`Windows`命令行](#关于windows命令行)

## 查询文件占用空间

```bash
#查询文件夹大小, max-depth指定查询深度
du -h --max-depth=1

#查询文件大小
du -h --max-depth=0 *

#查询磁盘空间
df -hl

#查询每个用户所占空间
sudo du -sh /home/*
```

## 删除文件夹下所有指定文件名的文件

```bash
PATH=文件夹路径
FILE=文件名

find ${PATH} -name ${FILE} -exec rm -f {} \;
```

## 读取文件

```bash
# 获取文件行数
wc -l 1.txt

# 获取文件前10行
head -n 10 1.txt

# 获取文件尾10行
tail -n 10 1.txt
```

## 文件合并

```
# 将1.txt和2.txt合并到3.txt
cat 1.txt 2.txt > 3.txt

# 将2.txt追加到1.txt中
cat 2.txt >> 1.txt
```

## for循环命令

```
# 阻塞式
for i in $(seq 0 5)
do
  ...  #每次执行完这一条命令才会进入下一个循环
done

# 非阻塞式
for i in $(seq 0 5)
do
{
  ...  #这条命令会被放至后台执行, 因此会立即执行下一循环的命令
}&
done
wait  #等待直到循环中的全部后台命令执行完毕
```

## 查询系统版本以及硬件信息

```bash
#查询系统版本
cat /proc/version

#查询CPU信息
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c

#查询内存信息
sudo lshw -C memory
```

## 进程管理

```bash
#查询进程号
ps -ef | grep process_name

#杀死进程
kill -s 9 pid
```

## 文件校验

```bash
# MD5
md5sum xxx

# SHA256
sha256sum xxx
```

## 压缩

```bash
# 仅打包
tar -cvf xxx.tar xxx      # 打包
tar -xvf xxx.tar          # 解包

# 打包并压缩
tar -zcvf xxx.tar.gz xxx  # 打包
tar -zxvf xxx.tar.gz      # 解包

# .gz
gzip -d xxx.gz            # 解压
gzip xxx                  # 压缩

# .zip
unzip xxx.zip             # 解压
zip -r xxx.zip xxx        # 压缩
```

## Vim

按`i`进入输入模式, 按`esc`进入命令模式, 按`:`进入底行模式。

命令模式:

```
gg 文首
G 文尾

{ 段首
} 段尾

| 行首
$ 行尾

b 词首
e 词尾

- 前一行首
+ 后一行首

b 前一单词
w 后一单词

u 撤销

/ 搜索, n下一个匹配, N上一个匹配
```

## Git

```bash
# init setting
git config --global user.name "username"
git config --global user.email email-address

# generate ssh key
ssh-keygen -t rsa -C email-address
ssh -T git@github.com

# init a repo
git init
git add *
git commit -m "add ..."
git remote add origin git@github.com:Komorebi660/[仓库名].git
git push -u origin master

# create new branch
git checkout -b xxx
git push origin xxx

# merge branch
git fetch origin main
git merge origin/main

# reduce .git size
git gc --prune=now
```

## 其他

在脚本中添加``` cd `dirname $0` ```可以直接定位到脚本所在目录，这样就不用关心在什么位置执行脚本了。

## 关于`Windows`命令行

`Windows`中有一些文件夹包含空格, 要想访问带有空格路径的可执行文件, 需要对带有空格的部分添加双引号:

```
PS: C:/Program Files/CMake/bin/cmake.exe --version
The term 'C:\Program' is not recognized as the name of a cmdlet, function, script file, or operable program.

PS: "C:/Program Files/CMake/bin/cmake.exe" --version
At line:1 char:42
+ "C:\Program Files\CMake\bin\cmake.exe" --version
+                                          ~~~~~~~
Unexpected token 'version' in expression or statement.

PS: C:/"Program Files"/CMake/bin/cmake.exe --version
cmake version 3.25.1

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```
