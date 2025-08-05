
def msg(prt):
    print("######################")
    print(prt)
    print("######################")
    
def total(num):
    total = 0 
    for i in range(1, num+1):
        if i % 3 == 0 or i % 5 == 0: 
            total += i
    
    return total    

def add(a: int, b: int) -> int: 
    return a+b



# ld = lambda x: x * x
# def lamdFun(lamd : ld , int a ) -> lambda
#     return lamd

if __name__ == "__main__":
   print(f"{__name__} , main OK")
elif __name__ != "__main__":
   print(f"{__name__} , main이 ------ 아닙니다.")
   
   
   
   

   a = "Hello"
   b = "100.0"
   type(a)
   type(b)

   a = "Hello python! | need python!"
   a[-16:-8]

   key = "축구:농구:배구:야구"


   bool(-1)
   bool('거짓')
   bool([])


   number = 0
   for i in range(1,5):
       number = number + i

   print(number)    

result = lambda x: x * 3

num = int(input("숫자를 입력하세요: "))
print("결과:", result(num))



def divide(a, b):
    quotient = a // b    # 몫
    remainder = a % b    # 나머지
    print("몫:", quotient)
    print("나머지:", remainder)

# 사용자 입력 받기
num1 = int(input("첫 번째 수를 입력하세요: "))
num2 = int(input("두 번째 수를 입력하세요: "))

divide(num1, num2)



numbers = [1,2,3,4,5]
a = []
for n in numbers:
    if n%2 == 1:
        a.append(n*2)

print(a)


class FirstClass:
    def __init__(self,number):
        self.number = number
    
    def printN(self):
        return self.number

a = FirstClass(3)
print(a.printN())    

import numpy as np # alias
import pandas as pd
np.arange(24).reshape(2,3,4).ndim


a = np.ones((2,3),int)
print(a)

a = np.identity(2,int)
b = np.full((2,2),3)
print(a)
print(b)

a = np.array([1,2,3])
b = np.array([4,5,6])
np.vstack([a,b])
np.hstack([a,b])


import numpy as np

arr = np.arange(36).reshape(3, 4, 3)
print(arr)



numbers = [1, 2, 3]
nums = pd.Series(numbers)
print(nums)






    수학 영어  음악 체육 
서준 90   98   85  100 
우현 80   89   95  90
인아 70   95  100  90
