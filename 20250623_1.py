import numpy as np # alias

a = np.arange(8)
a.shape
a.reshape(2,4)


np.arange(8).reshape(2,4)

c = np.linspace(0,1,5)

c
np.arange(1,9,2)
np.linspace(0,1,6)
np.ones((3,3),int)


a = np.arange(1,6)
a
b = a.copy()
b


np.arange(1,9,2)
np.linspace(0,1,6)
np.ones((3,3),int)


a = np.arange(5)
np.hstack([a * 10 , a * 20])
np.vstack([a * 10, a * 20])

a = [ 1,2,3 ]
b = [ 4,5,6 ]
np.vstack([a,b]).shape(4,6)








a = np.array([[1,2,3],[4,5,6]])
a
output = np.vsplit(a,2)
[a,b] = output 
a
(a,b) = output 
a
b

a = a.flatten()
a
a.reshape(3,)
a.reshape(3)
a.tolist()




a = np.arange(0,6)
a

b = np.arange(0,51,10).reshape(6,1)
b

arr = a + b
arr
arr[[0,1,2,3,4,5], [5,4,3,2,1,0]]


a = np.arange(1, 4)
b = np.arange(4, 7)
c = np.arange(7, 10)
d = np.vstack([a, b, c])
d

a = np.arange(4)
b = np.zeros(4, dtype=int)
b
x = np.array([[80,67,91],[78,95,80],[93,95,99],[89,60,76]])

x.sum(axis=0)[0]
x.sum(axis=0)[1]
x.sum(axis=0)[2]


a = np.arange(1,6)
b = np.arange(6,11)
c = np.arange(11,16)
d = np.arange(16,21)
e = np.arange(21,26)
e[0] = 12

result = np.vstack([a,b,c,d,e])

a = np.arange(1,26).reshape(5,5)


a = np.zeros((10,10))
a
a[0,9] = 5
a[9,0] = 37
a



a = np.arange(10)
np.where(a>3,0,1)
np.where(a<5,a,a*10)
np.where(a<5,a,0)


import random
random.random()
random.randint()
random.choices()
random.sample()


np.random.rand(10)
np.random.rand(2,3)
np.random.random(10)
np.random.random((10))
np.random.random((10,10))
np.random.randint(3)


np.random.sample((2,3))*2 +1

np.random.choice(np.arange(1,7),(2,3))
np.random.choice(np.arange(1,7),(3,),replace=True)
np.random.choice(np.arange(1,7),(3,),replace=False)
np.random.choice(np.arange(1,46),(6,),replace=False)
np.random.choice(np.arange(1,46),(6),replace=False)


np.random.choice(np.arange(50,100),(3,10),replace=False)
np.random.randint(50,101,(3,10))



np.random.rand(2,3)
np.random.sample((2,3))
np.random.rand(80)*5+10
# 평균이 80이고 표준편차는 5인 과목별접수 10개를 10X3으로 만들어봐라
np.random.normal(80,5,(10,3))
np.random.randn(10,3)*5+80


# 배열을 판다스로 만들수 있음
import pandas as pd
a = np.random.randint(50,101,(3,10)).T

df = pd.DataFrame(a)
df.columns = ['영어','수학','과학']
# df.columns = ['영어', '수학', '과학', '국어', '사회', '미술', '체육', '음악', '정보', '기술']
df
df.values
df.to_numpy()
