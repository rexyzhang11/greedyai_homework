### Numpy 
~~~
Numpy就是一个多维的数组对象，它的应用场景多是用在科学计算方面，因为它提供了很多数学方面的计算函数
~~~



```python
import numpy as np 
import pandas as pd
data = [1,2,3,4]
n = np.array(data)*10
m = np.array(data*10)
l = np.array([1,1,2,2]).reshape(-1,1) #一列4x1
print(l)
print(n)
print(m)
print(n*l)
```

    [[1]
     [1]
     [2]
     [2]]
    [10 20 30 40]
    [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1
     2 3 4]
    [[10 20 30 40]
     [10 20 30 40]
     [20 40 60 80]
     [20 40 60 80]]
    


```python
n.shape
```




    (4,)




```python
n.dtype #获取数据类型
```




    dtype('int32')




```python
arr = [[1,2,3,4],[3,4,5,6]]
arr2 = np.array(arr)
print(arr2)
arr2.shape
```

    [[1 2 3 4]
     [3 4 5 6]]
    




    (2, 4)




```python
arr2.ndim #维度
```




    2




```python
np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
np.ones(10)
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])




```python
np.empty((2,3,4))
```




    array([[[6.23042070e-307, 1.86918699e-306, 1.69121096e-306,
             2.13620807e-306],
            [7.56587585e-307, 7.56593017e-307, 1.24610383e-306,
             1.24610723e-306],
            [1.37962320e-306, 1.29060871e-306, 2.22522597e-306,
             1.33511969e-306]],
    
           [[1.78022342e-306, 1.05700345e-307, 1.11261027e-306,
             1.11261502e-306],
            [1.42410839e-306, 7.56597770e-307, 6.23059726e-307,
             1.78022342e-306],
            [6.23058028e-307, 9.34609790e-307, 1.37961709e-306,
             3.72364926e-317]]])



### 矢量化运算


```python
arr1 = np.array([1,2,3,4])
arr2 = np.array([1,2,3,4])
arr1 + arr2
```




    array([2, 4, 6, 8])




```python
arr1 = np.array([[1,2,3,4],[1,2,3,4]])
arr2 = np.array([[1,2,3,4],[1,2,3,4]])
arr1 + arr2
```




    array([[2, 4, 6, 8],
           [2, 4, 6, 8]])




```python
#矩阵的相乘运算不能这么算，因为矩阵相乘的规则并不是对应为相乘，所以下面的这个矩阵运算是错误的（其实是点乘）
arr12 =arr1 *arr2 #点乘
```


```python
#矩阵的访问
arr = np.arange(10)
print(arr[1])
print(arr[2:])
print(arr12[0,1])
```

    1
    [2 3 4 5 6 7 8 9]
    4
    

### Fancy indexing
- 这个是NumPy中一个术语，利用整数数组来进行索引


```python
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
arr
```




    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [2., 2., 2., 2.],
           [3., 3., 3., 3.],
           [4., 4., 4., 4.],
           [5., 5., 5., 5.],
           [6., 6., 6., 6.],
           [7., 7., 7., 7.]])




```python
arr[[2,5]] #取出第2行和第5行
```




    array([[2., 2., 2., 2.],
           [5., 5., 5., 5.]])




```python
arr[2,3] #取出第2行第3列的元素
```




    2.0




```python
arr = np.arange(32).reshape(8,4) #8行4列
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])




```python
arr[[1,5,7,2]] #取出第1,5,7,2行数据
```




    array([[ 4,  5,  6,  7],
           [20, 21, 22, 23],
           [28, 29, 30, 31],
           [ 8,  9, 10, 11]])




```python
arr[[1,5,7,2],[0,3,1,2]] #[1,5,7,2]为行， [0,3,1,2]为列，索引元素
```




    array([ 4, 23, 29, 10])




```python
arr[[1,5,7,2]][:,[0,3,1,2]] #先花式索引[1,5,7,2]行，然后再花式索引[0,3,1,2]列
```




    array([[ 4,  7,  5,  6],
           [20, 23, 21, 22],
           [28, 31, 29, 30],
           [ 8, 11,  9, 10]])




```python
arr[np.ix_([1,5,7,2],[0,3,1,2])] #效果同上
```




    array([[ 4,  7,  5,  6],
           [20, 23, 21, 22],
           [28, 31, 29, 30],
           [ 8, 11,  9, 10]])




```python
arr = np.random.randn(4,4)
arr
```




    array([[-1.18190805,  1.08236799,  0.49291509, -1.26068093],
           [-0.12123853, -0.98671358,  0.81417061, -1.81904692],
           [ 0.81012181, -0.66693597, -0.64762893,  0.80248794],
           [-0.99495788, -0.00629768,  0.55008675,  0.39162807]])




```python
arr.mean()
```




    -0.171351889053472




```python
arr.sum()
```




    -2.741630224855552




```python
arr = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])
arr
```




    array([[1, 1, 1, 1],
           [2, 2, 2, 2],
           [3, 3, 3, 3],
           [4, 4, 4, 4]])




```python
arr.sum(0) #求第每列的和，轴号Axis = 0 相当于列
```




    array([10, 10, 10, 10])




```python
arr.sum(1) #求第每行的和，轴号Axis = 1 相当于行
```




    array([ 4,  8, 12, 16])




```python
arr.mean(1)
```




    array([1., 2., 3., 4.])




```python
x = np.array([[1,2,3],[4,5,6]])
y = np.array([[1,2],[4,5],[7,8]])
x.dot(y) #矩阵相乘
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-d798632bde33> in <module>
    ----> 1 x = np.array([[1,2,3],[4,5,6]])
          2 y = np.array([[1,2],[4,5],[7,8]])
          3 x.dot(y) #矩阵相乘
    

    NameError: name 'np' is not defined


### Pandas 
~~~
Pandas是基于Numpy构建的，让以Numpy为中心的应用变得更加地简单
专注于数据处理，
这个库可以帮助数据分析、数据发掘、算法等工程师岗位的人员轻松快速的解决预处理的问题
比如，数据类型的转换、缺失值的处理、描述性统计分析、数据汇总等功能
~~~



```python
import pandas as pd
import numpy as np
#构造数列 series
obj = pd.Series([1,2,3,4])
obj
```




    0    1
    1    2
    2    3
    3    4
    dtype: int64




```python
obj[0]
```




    1




```python
obj[[0,2]]
```




    0    1
    2    3
    dtype: int64




```python
#自定义索引
obj2 = pd.Series({"姓名":"张三","地址":"北京","年龄":20})
obj2
```




    姓名    张三
    地址    北京
    年龄    20
    dtype: object




```python
obj2[["姓名","年龄"]]
```




    姓名    张三
    年龄    20
    dtype: object




```python
#另一种写法
obj3 = pd.Series(["李四","上海",25],index=["姓名","地址","年龄"])
obj3
```




    姓名    李四
    地址    上海
    年龄    25
    dtype: object




```python
number1 = pd.Series([4,8,16,32,64])
np.log2(number1)
```




    0    2.0
    1    3.0
    2    4.0
    3    5.0
    4    6.0
    dtype: float64



### DataFrame 数据框


```python
df1 = pd.DataFrame([["张三",29,"男"],["李四",20,"女"],["王五",21,"男"]])
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>张三</td>
      <td>29</td>
      <td>男</td>
    </tr>
    <tr>
      <td>1</td>
      <td>李四</td>
      <td>20</td>
      <td>女</td>
    </tr>
    <tr>
      <td>2</td>
      <td>王五</td>
      <td>21</td>
      <td>男</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1[0]
```




    0    张三
    1    李四
    2    王五
    Name: 0, dtype: object




```python
df1[0][1]
```




    '李四'




```python
data = {
        "60":["狗子","嘎子","二妞"],
        "70":["卫国","建国","爱国"]
}
frame_data = pd.DataFrame(data)
frame_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>60</th>
      <th>70</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>狗子</td>
      <td>卫国</td>
    </tr>
    <tr>
      <td>1</td>
      <td>嘎子</td>
      <td>建国</td>
    </tr>
    <tr>
      <td>2</td>
      <td>二妞</td>
      <td>爱国</td>
    </tr>
  </tbody>
</table>
</div>




```python
#windows
!type data1.csv                 
#  !cat data1.csv                linux 
```

    a,b,c,d,e
    1,2,3,4,5
    6,7,8,9,10
    


```python
!type data1.txt

```

    a	b	c	d	e
    1	2	3	4	5
    6	7	8	9	10
    


```python
pd.read_csv("data1.txt",sep="\t") #读取txt文件
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv("data1.txt",sep="\t",header=None) #不要header
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>d</td>
      <td>e</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv("data1.txt",sep="\t",index_col="b") #指定b列为索引
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
    <tr>
      <th>b</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <td>7</td>
      <td>6</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
!pip3 install xlrd -i https://pypi.tuna.tsinghua.edu.cn/simple
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting xlrd
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b0/16/63576a1a001752e34bf8ea62e367997530dc553b689356b9879339cf45a4/xlrd-1.2.0-py2.py3-none-any.whl (103kB)
    Installing collected packages: xlrd
    Successfully installed xlrd-1.2.0
    


```python
pd.read_excel("data.xlsx") #读取excel第一页（默认的）
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>place</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>22</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>13</td>
      <td>23</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>14</td>
      <td>24</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>15</td>
      <td>25</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>16</td>
      <td>26</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>17</td>
      <td>27</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>18</td>
      <td>28</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>19</td>
      <td>29</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>20</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_excel("data.xlsx",sheet_name="工作表2") #读取excel第二页
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>place</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>101</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>102</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>13</td>
      <td>103</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>14</td>
      <td>104</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>15</td>
      <td>105</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>16</td>
      <td>106</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>17</td>
      <td>107</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>18</td>
      <td>108</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>19</td>
      <td>109</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>20</td>
      <td>110</td>
    </tr>
  </tbody>
</table>
</div>




```python
csv_data = pd.read_csv("executive.csv")
csv_data.shape #5650行4列
```




    (5650, 4)




```python
csv_data.dtypes #查看数据类型
```




    name:ID    object
    sex        object
    age         int64
    :LABEL     object
    dtype: object




```python
csv_data.sex #访问一列
```




    0       男
    1       男
    2       男
    3       男
    4       男
           ..
    5645    男
    5646    女
    5647    男
    5648    女
    5649    男
    Name: sex, Length: 5650, dtype: object




```python
csv_data.sex[0] #访问一个
```




    '男'




```python
csv_data.describe() #只统计age这一列，因为只有这一个是数字类型
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>5650.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>51.669735</td>
    </tr>
    <tr>
      <td>std</td>
      <td>7.691150</td>
    </tr>
    <tr>
      <td>min</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>52.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>56.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>96.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dates = pd.date_range('20191001',periods=6)
dates
```




    DatetimeIndex(['2019-10-01', '2019-10-02', '2019-10-03', '2019-10-04',
                   '2019-10-05', '2019-10-06'],
                  dtype='datetime64[ns]', freq='D')



~~~
linux查看文件: cat, tail, head
~~~


```python
obj = pd.Series([4.5,9.8,-2.3],index=["a","b","c"])
obj
```




    a    4.5
    b    9.8
    c   -2.3
    dtype: float64




```python
obj.a
```




    4.5




```python
obj_1 = obj.reindex(['a','b','c','q','w','e','r'])
obj_1
```




    a    4.5
    b    9.8
    c   -2.3
    q    NaN
    w    NaN
    e    NaN
    r    NaN
    dtype: float64




```python
obj_2 = obj.reindex(['a','b','c','q','w','e','r'], fill_value=0)
obj_2
```




    a    4.5
    b    9.8
    c   -2.3
    q    0.0
    w    0.0
    e    0.0
    r    0.0
    dtype: float64


