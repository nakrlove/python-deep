a =  [1,2,3]

b = a[:]

b
id(b)
id(a)

b = copy(a)
id(b)

import numpy as np # alias
a = np.array([5,4,2,3])


a = np.arange(1,4).reshape(2,2,)
b = np.arange(10,14).reshape(2,2,)
a + b

c = np.vstack((a,b))
d = np.hstack((a,b))
np.concatenate((a, b))


import numpy as np
a = np.arange(18).reshape(2,3,3)
b = np.ones(a, dtype=int)
np.ones_like(a)
np.zeros_like(a)
# 5보다 큰수는 모두 1로, 작은 수는 0으로 바꾸어 봅시다
result = np.where(a > 5, 1, 0)
result
# 5보다 큰수는 모두 그대로 두고 작은 수는 0으로 바꾸어 봅시다.
result = np.where(a < 5, 0,a )
result

# 행의 개수가 6개인 2차원 배열로 바꾸어 봅시다.
a.reshape(6,-1)





a = np.arange(0,11).reshape(1,-1)
a

b = a+10
c = a+20
d = a+30
e = a+40
f = a+50



a = np.arange(0,11).reshape(1,-1)
a
b = np.arange(0,81,20).reshape(5,-1)
arr = a + b
arr

a = np.arange(11) + np.arange(0, 81, 20).reshape(5, -1)
a
# array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
#        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
#        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
#        [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
#        [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]])
# 위치 검색 [ 0,  1,  9, 10, 22, 28, 43, 47, 64, 66, 85]
a[[0,0,0,0,1,1,2,2,3,3,4], [0,1,9,10,2,8,3,7,4,6,5]]

# hstack , vstack은 무엇을 하는것?


a = np.array([1,2,3])
b = 10
a+b # array([11, 12, 13])
np.sum(a+b)


# Pandas의 DataFrame과 Series의 차이점은 무었일까요?
# 1차원 labeled array	2차원 labeled table (열의 모음)
# np.random.randn() , np.random.normal()의 차이는 무었일까요?
# - np.random.randn() 표준 정규분포 (평균=0, 표준편차=1, 고정)
# - np.random.normal() 일반 정규분포 (평균, 표준편차 직접 설정 가능)

np.random.normal(80,1000,(2,3)) 
