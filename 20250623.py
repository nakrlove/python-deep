import re

with open('Q1.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
    # pat = re.compile(r'[가-힣]+|[ㄱ-ㅎㅏ-ㅣ]+')

    # # findall 로 한글 단어 리스트 얻기
    # hangul_words = pat.findall(text)

    # # 단어 사이 띄어쓰기 포함하여 다시 합치기
    # result_text = ' '.join(hangul_words)

    # print(result_text)
    
    
pat = re.compile(r"[<].+?[>]")
pat = re.compile(r"<.+?>")
    # pat = re.compile(r'')
text = pat.sub("",text)
text
    
pat = re.compile("r[?!.,~ㅋㅠ]")
pat = re.compile("r[^가-힣0-9 ]")
new_text1 = pat.sub("",text)    
new_text1



import numpy as np # alias
import pandas as pd

np.ones((2,3))
np.zeros((4,5))


a = np.ones((4,5) ,dtype="int64") * 5
a
a.shape
np.zeros_like(a)

a = np.arange(21).reshape(3,7)
a
np.ones_like(a)
a = np.array([5,4,2,3])
b = a
b
b[0] = 50
b
a


import numpy as np


a = np.arange(0,4).reshape(2,2,)
b = np.arange(10,14).reshape(2,2,)

c = np.vstack((a,b))
d = np.hstack((a,b))
np.concatenate((a,b))
np.concatenate((a,b),axis=0)
np.concatenate((a,b),axis=1)

a = np.arange(8)

a.reshape(2,4)
a.reshape(2,-1)
a.reshape(4,-1)
a.reshape(-1,4)

b.reshape(1,-1)
b.reshape(1,-1).shape

b.flatten()
b.flatten().shape


b
a = np.arange(6).reshape(2,3)
a

np.transpose(a)
a.transpose()
a.T


a = np.arange(24).reshape(6,4)
a
a.T


a = np.array([1,2,3])
b = np.array([4,5,6])

c = np.vstack((a,b))
c

# np.concatenate((a,b) ,axis=1)
result = np.concatenate([a, b], axis=1)
result
a2 = a.reshape(3, 1)  # (3, 1)
b2 = b.reshape(3, 1)  # (3, 1)
a2
result = np.concatenate([a2, b2], axis=1)
result



a = np.arange(1,5)
b = np.array([1,2,5,6])
np.random.shuffle(a)

a == b
a != b
a[a==b]
b[a==b]
a[a!=b]

a
a[ a > 2 ]

a = np.arange(1,101).reshape(10,10)
a
d = (a % 3 == 0)*3
d

a[a % 3 == 0]
a[a % 3 == 0].reshape(3,-1)

a = np.array([3,2,4,1,5])
sorted(a)
np.sort(a)

a.sort()
a


a = np.array([[3,5,9],[2,4,6],[2,1,8]])
a
np.sort(a)
np.sort(a,axis=0)
a.sort(axis=0)
a
a.sort(axis=1)
a
a[ : , ::-1]


a = np.arange(5)

a = np.arange(0,6).reshape(1,-1)
a
b = a +10
b
a.reshape(1,-1).shape
a
a = np.arange(0,56)
a


a = np.arange(0,6).reshape(1,-1)
b = a+10
c = a+20
d = a+30
e = a+40
f = a+50
[]
result = np.vstack([a, b, c, d, e, f])
result


print(result[:, 2])


for i in range(0,51,10):
    print(i)
    
    
a = np.arange(0,6).reshape(1,-1)
b = np.arange(0,51,10).reshape(6,-1)
arr = a + b

arr

arr [ 4: , 4:]
arr [2::2,::2]

arr
b = [arr[0,1],arr[1,2],arr[2,3]]
b
a = np.array([[1,2],[3,4]])
a

(a[0][0],a[1][1])
a[[0,1],[1,1]]

np.arange(48).reshape(4,4,3)



text ="<html><head><title>Title</title></head><body>abc123!</body></html>"
re.compile("\w(?=<)")