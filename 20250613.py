result = 0
# def add(x , y):
#     global result
#     result = x + y
#     return result



# def subtract(x,y):
#     result = x - y
#     return result

# def multiply(x,y):
#     result = x * y
#     return result


# add(1,2)
# add(10,20)


class Pos:
    
    result
    
    def add(self, x, y):
        self.result = x + y
        return self.result

p = Pos() #객체 object 인스턴스 instance
p.add(1,2)
p.result


p1 = Pos()
p1.add(10,20)
p1.result


p2 = Pos()
p2.add(11,12)
p2.result


# constructor 
# 클래스: 생성자, 속성,메소드
class Calculator:
    
    # result
    color = "white"
    price = 10000
    
    # def __init__(self):
        
    def __init__(self,color,price):
       # print( "만들었습니다." )
       self.color = color
       self.price = price
       self.result = 0
       self.manufacturer = "현대"
    
    def add(self,a,b):
        self.result = a + b
        return self.result
    
    
    def subtract(self,x,y):
        self.result = x - y
        return self.result

    def divide(self,x,y):
        self.result = x / y
        return self.result

    def multiply(self,x,y):
        self.result = x * y
        return self.result
    
    def chang(self,msg):
        self.manufacturer = msg
        

#%%
class Child(Calculator):
    
    def power(self,x,y):
        self.result = x ** y
        return self.result
    
    def divide(self,x,y):
        try:
            self.result = x / y
            return self.result
        
        except ZeroDivisionError as e:    
            print("0으로 나눌수 없습니다")
      
    def divide1(self,x,y):
        if x == 0 or y == 0:
            return "0으로 나눌수 없습니다"
        self.result = x / y
        return  self.result
    
    
    
    
#%%    
    
cal = Calculator("black",22)
# cal = Calculator()
cal.color
cal.manufacturer
cal.chang("DDDDD")
cal.price
cal.add(10,20)
cal.subtract(10,20)
cal.multiply(10,20)
cal.divide(10,20)
cal.result



#%%
c  = Child('black',1000)
c.add(10, 20)
# c.divide(10, 0)
c.divide1(10, 0)