#%%
class Person:
    
    def __init__(self,name,age):
        self.name = name
        self.age  = age
        
    # def introduce(self)
    #     return self.name,self.age
    
    def introduce(self):
        print(f"저는 {self.name}이고 {self.age}살 입니다.")
        
# class Student(Person):
    
        
p1 = Person("콜라",20)
p1.introduce()



#%%
class Calculator:
    
    def __init__(self,li):
        self.li = li
        self.result = 0
        
    def __init__(self,*member_list):
        self.total =  [i for i in member_list] 
     
        
        
    def msg(self):
        print(self.total)
        
    def total(self):
        # self.result = sum(self.li)
        # print(self.result)
        for i in self.li:
            self.result += i
            
        print(self.result) 


cal1 =Calculator([80,75,95,100,90])
#cal1.total()   
cal1.msg()
        
        
#%%
class Bread:
    
    def __init__(self,price,customer,review):
        self.price = price
        self.customer = customer
        self.review = review
        
    def meterial(self):
        print(f"가격은 {self.price}원이고 {self.customer}개가 판매되었습니다")


b = Bread(5000,10,"별이 5개")
b.meterial()        



#%%
class Person:
    
    def __init__(self,name,age):
        self.name = name
        self.age  = age
    
    def who(self):
        return f"이름은 {self.name}이고 나이는 {self.age}세 입니다."
  
    def __str__(self):
        return f"name = {self.name} , age = {self.age}"

p1 = Person("홍길동", 23)
p1.who()     
print(p1)  



#%%
class Book:
    
    def __init__(self,book_name,writer_name,price,publisher):
        self.book_name = book_name
        self.writer_name = writer_name
        self.price = price
        self.publisher = publisher
        

    def sale(self,sale_count):
        return f"판매 부수는 {sale_count}권, 총 매출은 {sale_count*self.price}원입니다."
        

book = Book("놀부전", "고전", 4500, 'publisher')
book.sale(8)



#%%

name1 ="Choi" # global variable

class Family:
    
    lastname = "kim"   #class variable
    def __init__(self,name):
        self.name = name # local variable, method varibale member varible
      
    def say(self):
        self.age = 30
        

     
a = Family('Park')
name1
a.lastname 
Family.lastname # 클래스변수는 객체를 만들지 않고 접근한다.
a.name   



b = Family('Lee')
b.lastname     
Family.lastname # 클래스변수는 객체를 만들지 않고 접근한다.
b.name   

id(Family.lastname)   
id(a.lastname)
id(b.lastname)

Family.lastname = "CC"
id(Family.lastname)   
id(a.lastname)
id(b.lastname)



id(a.name)
id(b.name)



#%%
# name 전역변수


name = "아이유"  #전역변수

class Person:
    
    name = "이지은"  # 클래스변수
    
    def __init__(self,instance_name): 
        self.name = instance_name  #인스턴스 변수 
        
        
p = Person("박보검")
name
Person.name
p.name


#%%
class Person:
    
    
    def __init__(self,name,age): 
        self.name = name  #인스턴스 변수 
        self.age = age
        
class Korean(Person):
    
    def __init__(self,name,age,lang): 
        super().__init__(name,age)
        self.lang = lang
        
            
class American(Person):
    
    def __init__(self,name,age,lang = None): 
        super().__init__(name,age)
        self.lang = lang if lang is not None else "영어" 
        

p = Person("길동",20)
p.name
p.age


p1 = Korean("길동",20,"한국어")
p1.name
p1.age
p1.lang

p2 = American("Smith",22,"English")
p2.name
p2.age
p2.lang
        
        
        
        
        
#문제1. 
# Person이 부모클래스인 student 자식클래스 만들기
# Person 클래스
class Person:
   
    def __init__(self, name, age, height, weight):
        
        self.name= name
        self.age = age
        self.height = height
        self.weight = weight
       
    def say(self, sentence):
        self.sentence = sentence
        print(f'{self.name}이/(가) "{self.sentence}"이라고 말합니다.')
       
    def eat(self, food="밥"):
        self.food = food
        print(f"{self.name}이/(가) {self.food}을/를 먹습니다.")        
        
# Student 클래스(부모클래스 : Person)        
class Student(Person):
   
    def __init__(self, name, age, height, weight, grade="F"):
        super().__init__(name, age, height, weight)
        self.grade = grade


student1 = Student("영희", 20, 160, 50, "B")
student2 = Student("철수", 20, 170, 68, "A")

student2.say("안녕")
student1.eat() 





#static :정적인, 항상,늘     static IP  : 고정IP
#dynamic:동적인, 그 때, 그대 dynamic IP : 유동IPs










#########
import math


r = 10
pi = math.pi
print(pi*r*r)

math.ceil(10.2)



import numpy as np
np.ceil(123.456)
np.floor(123.456)
