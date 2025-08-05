import numpy as np # alias
import pandas as pd
# np: 100배 빠르다고 함 

type([1,2,3])

a = pd.Series([1,2,3])
type(a)
a = np.array([1,2,3])
type(a)



[ li * 2 for li in [1,2,3]]

li = [1,2,3]

m = map(li * 2, li)
m

list(m)[0]
print(list(m)[0])



for i in li:
    print(i)
    
    
a = np.array([0,1,2,3])
list(a.data)


m= np.array = [0,1,2,3]
print(m)

np.array([[1,2,3],[3,4,5]])


a = np.array([[1,2,3],[3,4,5]])

a = np.array([[1, 2, 3], [3, 4, 5]])
a


arr = np.arange(36).reshape(3,4,3)
arr

a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
a+b

a = np.arange(1,10).reshape(3,3)
a
b = np.arange(10,19).reshape(3,3)
b
a*b
np.matmul(a,b)
np.dot(a,b)
a.dot(b)
array[[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]]]
a = np.array[[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]]]
arr

a = np.arange(0,48).reshape(4,4,3)
a


import re
text ="<html><head><title>Title</title></head><body>abc123!</body></html>"
pat = re.compile(r"(?<=<body>).+(?=</body>)")
print(pat.search(text).group())

text="오늘 날짜는 2025년 06월 19일 입니다. 전화번호는 010-1234-5678 입니다."
pat = re.compile(r"\d+")
result = pat.findall(text)
result


pat = re.compile(r"\b(\d{3}-\d{4}-\d{4})\b")
print(pat.search(text).group())
# findall로 리스트 추출
result = pat.findall(text)
print(result)

pat = re.compile(r"(?<=년 )\d{2}월 \d{2}(?=일)")
print(pat.search(text).group())

pat = re.compile(r"\d{4}년 \d{2}월 \d{2}일")
print(pat.search(text).group())


pat2 = re.compile(r"(?>년 ).*?(?=일)")
print(pat2.search(text).group())

pat3 = re.compile(r"(?<=날짜는 ).*?일")
print(pat3.search(text).group())


pat = re.compile(r"(?<=날짜는 ).*?(?=일)")

# findall 사용 → 리스트로 바로 저장
result = pat.findall(text)

print(result)


text="오늘 날짜는 $2025 $06 $19 입니다. 전화번호는 010-1234-5678 입니다."

pat = re.compile(r"(?<=날짜는 ).*?(?= 입니다)")
result = pat.findall(text)
result


pat = re.compile(r"(?<=[$])\d+")
pat = re.compile(r"\d+")
pat.findall(text)



