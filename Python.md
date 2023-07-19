# Usage of Python

- [Usage of Python](#usage-of-python)
  - [data load \& store](#data-load--store)
    - [`.txt`](#txt)
    - [`.tsv`](#tsv)
    - [`.pt`](#pt)
    - [`.bin`](#bin)
  - [yeild](#yeild)
  - [Basic Data Structure](#basic-data-structure)
    - [dict](#dict)
    - [set](#set)
    - [list](#list)
  - [numpy](#numpy)
  - [pytorch](#pytorch)
  - [Matplotlib](#matplotlib)
  - [Faker](#faker)
  - [FAISS](#faiss)

## data load & store

### `.txt`

```python
with open("xxx.txt", "w", encoding="utf8") as f:
    f.write("xxx\n123\n")

with open("xxx.txt", "r", encoding="utf8") as f:
    f.read() # xxx\n123\n

with open("xxx.txt", "r", encoding="utf8") as f:
    f.read(1) # x

with open("xxx.txt", "r", encoding="utf8") as f:
    f.readline() # xxx

with open("xxx.txt", "r", encoding="utf8") as f:
    f.readlines() # ['xxx\n', '123\n']
```

### `.tsv`

```python
import csv

with open("xxx.tsv", "w", encoding="utf8") as f:
    f.write("xxx\txxx\txxx\n")

with open("xxx.tsv", "r", encoding="utf8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        ...
```

### `.pt`

```python
import pickle

with open("xxx.pt", 'wb') as f:
    pickle.dump((data1, data2), f)

with open("xxx.pt", 'rb') as f:
    data1, data2 = pickle.load(f)
```

**Warning: `pickle.load()`不信任的文件可能会带来安全隐患。** `pickle`是专门为python设计的序列化工具，它可以将python中的对象转换为字节流，并能复原为python对象。但是，python为`class`添加了一个特别的`__reduce__()` method用来告诉`pickle`如何复原数据，我们可以利用这一method执行不安全的代码。一个例子如下:

```python
import pickle
import subprocess

class Dangerous:
    def __reduce__(self):
        return (
            subprocess.Popen, 
            (('/bin/bash', "-c", "ls"),),
        )
d = Dangerous()

with open("data.pt", 'wb') as f:
    pickle.dump(d, f)

with open("data.pt", 'rb') as f:
    data = pickle.load(f)
```

执行上述代码，在`load pickle`文件时，会执行`Dangerous`的`__reduce__`method用于恢复数据，在上例中就是打开了`bash`并执行`ls`命令。可以发现，如果随意加载`pickle`文件，可能会带来安全隐患。

### `.bin`

```python
import struct

with open("xxx.bin", 'wb') as f:
    data = [1.0, 2.0, 3.0]
    length = len(data)
    # struct.pack(format, data)
    f.write(struct.pack("i", length))
    f.write(struct.pack("%sf" * length, *data))

with open("xxx.bin", 'rb') as f:
    # struct.unpack(format, buffer)
    i = struct.unpack("i", f.read(4))[0]
    _list = struct.unpack("%sf" % i, f.read(4 * i))[0]
```

## yeild

返回一个`generator`, 可以用于节省内存。

```python
def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b 
        a, b = b, a + b 
        n = n + 1
 
for n in fab(5): 
    print n
```

在`for`循环执行时, 每次循环都会执行`fab`函数内部的代码, 执行到`yield b`时, `fab`函数就返回一个迭代值, 下次迭代时, 代码从`yield b`的下一条语句继续执行。也可以手动调用`next()`方法来获取下一个元素:

```python
f = fab(5)
f.next() # 1
f.next() # 1
f.next() # 2
f.next() # 3
f.next() # 5
```

`yield from`可用于调用另一个`generator`:

```python
s = 'ABC'
t = tuple(range(3))

def f1(*iterables):
    for it in iterables:
        for i in it:
            yield i

def f2(*iterables):
    for it in iterables:
        yield from it

#f1与f2等价
list(f1(s, t)) # ['A', 'B', 'C', 0, 1, 2]
list(f2(s, t)) # ['A', 'B', 'C', 0, 1, 2]
```

## Basic Data Structure

### dict

```python
d = {"a": [1, 2], "b": [3], "c": [4, 5, 6]}

# insert
if key not in d.keys():
    d[key] = [value]
else:
    d[key].append(value)

# delete
for key in list(d.keys()):
    if d[key] == value:
        del d[key]

# traverse
for key in d.keys():
    print(key, d[key])

# str to dict
import json
str1 = "{'a': 1, 'b': 2}"
dict1 = json.loads(str1)
```

### set

```python
a = set([1, 2, 3])
b = set([2, 3, 4])

# 求交集(计算recall)
a.intersection(b) # {2, 3}

# 求并集
a.union(b) # {1, 2, 3, 4}

# 求差集
a.difference(b) # {1}
b.difference(a) # {4}
```

### list

```python
# transform
a = ['1', '2', '3']
b = list(map(int, a)) # [1, 2, 3]

# sort
a = [[1, 3], [2, 2], [3, 1]]
a.sort(key=lambda x: x[1])               # [[3, 1], [2, 2], [1, 3]]
a.sort(key=lambda x: x[1], reverse=True) # [[1, 3], [2, 2], [3, 1]]
```

## numpy

```python
#设置随机种子
np.random.seed(0)
# 依概率p从data中随机采样size个数据
r = np.random.choice(data, p=p, size=10)
# 随机生成0~1之间的浮点数矩阵
r = np.random.random((10, 10))
# 随机生成[0, 9]整数矩阵
r = np.random.randint(0, 10, (10, 10))


# find index of a given value in the array
a = np.array([1, 3, 3, 4, 5])
index = np.where(a == 3)[0]     # [1, 2]
b = a[index]                    # [3, 3]


# L2 norm
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#按行计算
np.linalg.norm(x, axis=1)       # [3.74165739, 8.77496439, 13.92838828]
#按列计算
np.linalg.norm(x, axis=0)       # [9.53939201, 11.22497216, 12.12435565]

# 按行累乘
np.cumprod(x, axis=1)           # [[1, 2, 6], [4, 20, 120], [7, 56, 504]]
# 按列累乘
np.cumprod(x, axis=0)           # [[1, 2, 3], [4, 10, 18], [28, 80, 162]]


# arg sort
a = np.array([4, 2, 1, 5, 3])
np.argsort(a)           
# [2 1 4 0 3]
# first element is "1" which is in a[2]


# delete column/row
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.delete(a, 1, axis=0) # [1, 2, 3], [7, 8, 9]
np.delete(a, 1, axis=1) # [1, 3], [4, 6], [7, 9]


#矩阵求逆
np.linalg.inv(a)    # a is two-dim


#求多元变量的均值和协方差矩阵
a = np.random.rand(100, 10) #100组数据，每组10个特征
mean = np.mean(a, axis=0)   #10
cov = np.cov(a.T)           #10*10
#多元高斯采样
np.random.multivariate_normal(mean, cov)


#对角矩阵转化
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.diagonal(a)      # [1, 5, 9]
c = np.diag(b)          # [[1, 0, 0], [0, 5, 0], [0, 0, 9]]


#填充
a = np.array([0, 0, 0])
b = np.pad(a, (2, 3), 'constant', constant_values=(1, -1))  # [1, 1, 0, 0, 0, -1, -1, -1]


#求topk (dataset按照id从小到大有序排列: 0, 1, ...)
def get_topk(query, dataset, k):
    distance = np.linalg.norm(dataset - query, axis=1)
    topk_index = np.argpartition(distance, k)[:k]
    topk_distance = distance[topk_index]
    return topk_index[np.argsort(topk_distance)]

def get_topk(query, dataset, k):
    distance = np.dot(dataset, query)
    topk_index = np.argpartition(distance, -k)[-k:]
    topk_distance = -distance[topk_index]
    return topk_index[np.argsort(topk_distance)]
```

## pytorch

```python
import torch as th

torch.clamp(input, min, max) # 将input中的元素限制在[min, max]之间


# 求一个矩阵行与行之间的l2 norm距离, 返回flat之后的三角矩阵
import torch.nn.functional as F
x = th.tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]])
F.pdist(x, p=2) # tensor([0., 1., 1.])
```

## Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

#设置全局字体
plt.rc('font', family='Times New Roman')

#设置标题
plt.title('Test', fontsize=20, color='black')
# 设置坐标轴标签
plt.xlabel('axis_x', fontsize=15, color='black')
plt.ylabel('axis_y', fontsize=15, color='black')
# 设置刻度范围
plt.xlim(-10.0, 10.0)
plt.ylim(0.0, 10000.0)
#设置刻度scale
plt.yscale('log')
# 设置刻度标签
plt.xticks(np.arange(11), ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十'], fontsize=10, color='gray')
plt.yticks(fontsize=10, color='gray')

#画曲线
plt.plot(x, y, color='red', linewidth=1.0, linestyle='--', marker='.', label='label1')
"""
color: 颜色
linewidth: 线宽
linestyle: 线型
marker: 标记样式
label: 图例
"""

#画散点
plt.scatter(x, y, color='red', marker='o', label='label2', s=10, alpha=0.6)
"""
color: 颜色
marker: 标记样式
label: 图例
s: 标记大小
alpha: 透明度
"""

#画直方图
plt.hist(data, bins=100, facecolor="#99CCFF", edgecolor="black")
"""
bins: 多少根柱子
facecolor: 填充颜色
edgecolor: 边缘颜色
"""

#画柱状图
num_of_algo = 3     #参与绘图的算法数目
num_of_data = 5     #数据数目
bar_width = 0.30    #柱宽

#设置每个柱子的x坐标
index = np.arange(0, num_of_data, 1)

# 画柱状图 (data是 num_of_algo*num_of_data 的矩阵)
for i in range(num_of_algo):
    plt.bar(index + i * bar_width, data[i], bar_width, label=label[i], facecolor=facecolor[i], edgecolor=edgecolor[i], hatch=hatch[i])
    """
    index + i * bar_width: 柱子的x坐标
    data[i]: 柱子的高度
    bar_width: 柱宽
    label: 图例
    facecolor: 柱子填充颜色
    edgecolor: 柱子边框颜色
    hatch: 柱子填充样式
    """
    for a, b in zip(index + i * bar_width, data[i]):
        # a: 文字的x坐标，b: 文字的y坐标
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5, rotation=90)

# 设置x轴刻度在中间
plt.xticks(index + (num_of_algo-1)*bar_width / 2,  index)



plt.legend(bbox_to_anchor=(0.9, 1.2), fontsize=30, ncol=2, markerscale=2, frameon=True)
"""
bbox_to_anchor: 图例位置
fontsize: 字体大小
ncol: 列数
markerscale: 标记大小
frameon: 是否显示边框
"""

#紧致布局
plt.tight_layout()
#保存为矢量图
plt.savefig("multicol_traverse.svg", format="svg")
plt.show()
```

## Faker

install:

```bash
pip install faker
```

usage: https://www.jianshu.com/p/6bd6869631d9


```python
from faker import Faker

fake = Faker()
location_list = [fake.country() for _ in range(200)]
```

## FAISS

install:

```bash
pip install faiss-gpu [or faiss-cpu]
```

usage:

```python
import faiss

def build_index(data):
    data_num, data_dim = data.shape

    index = faiss.IndexFlatL2(data_dim)     # L2 norm
    #index = faiss.IndexFlatIP(data_dim)    # inner product
    
    # use GPU
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # build index
    index.add(data)
    assert index.ntotal == data_num

    # save index
    index_cpu = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index_cpu, 'index_xxx')

def search(query):
    query_num, query_dim = data.shape

    index = faiss.read_index('index_xxx')
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    D, I = index.search(query, 100) # search top-100

    for i in range(query_num):
        results = I[i]
        # results[0:99] are top-100 ids
```