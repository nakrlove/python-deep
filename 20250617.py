li=[-5,3,8,2,-7,9,-7,1]
result = list(map(lambda x: 1 if x > 0 else 0, li))
print(result) 




class Vehicle:
    
    def __init__(self,name):
        self.name = name
    
        
    def run(self):
        return f"{self.name}가 달립니다."
    
    
    
class Car(Vehicle):

    def __init__(self,name,manufacturer):
        
        super().__init__(name)
        # self.name = name
        self.manufacturer = manufacturer
        
    def stop(self):
        return f"{self.name} STOP"
    
    def run(self,speed):
        self.speed = speed
        return f"시속 {speed}km로 달립니다."
        
    
# v = Vehicle("자전거")
c = Car("Sonata","Kia")
c.manufacturer
c.name



import math
import math as m
from math import pi, ceil
# math.pi
# math.ceil(1,23)

pi
ceil(1.23)



# 0 ~ 1사이의 값을 한 개 내 놓는다
import random
round(random.random() * 10)
random.random()
random.uniform(0, 1)
random.randrange(5,10,2)
random.choice([1,2,5,6,7])
random.choices([1,2,5,6,7], k = 3)
random.choices(list(range(1,64)), k = 6)
# [a, b] 사이의 정수가 아웃풋
random.randint(5,10)

li = [1,2,3,4,5]
random.shuffle(li)
li



random.sample(li, k = 3)


random.seed(1)
for i in range(5):
    print(random.random())



import numpy as np
np.random



import sys
print(sys.version) 

import os
os.name
os.getcwd()
os.listdir()


os.mkdir("test1")
os.chdir("test1")
os.chdir("..") # 상위경로 이동
os.rmdir("test1") #디렉토리 삭제

os.system("dir")


# datetime 라이브러리
import datetime
now = datetime.datetime.now()
time = datetime.datetime(2025,6,17)
type(time)

delta = now - time
delta + now
type(delta)


chrismtmas = datetime.datetime(2025,12,25)
today = datetime.datetime.now()
delta = chrismtmas - today
delta.days
delta.seconds

print(f"{delta.days}일 시간 {math.floor(delta.seconds/60/60)}시간 {delta.seconds}초 남았습니다.")

# time 모듈
import time
time.time() # 1970.1.1.0.0.0

for i in range(5):
    print("Hello")
    time.sleep(2)

# urllib

# import urllib.request
import urllib.request as req

# response = urllib.request.urlopen("https://google.com")
response = req.urlopen("https://google.com")
response.read()


from urllib import request
from urllib.request import urlopen






li = [1, 2, 3]
li1 = ['a', 'b', 'c']

temp = []


for i in range(3):  
    temp.append((li[i] , li1[i]))

list(zip(li, li1))




import webbrowser

webbrowser.open_new('http://python.org')
time.sleep(2)
webbrowser.open("naver.com")


