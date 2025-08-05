PI = 3.141592
# def number_input():
    # output = float(input("숫자 입력"))
    # return output

def get_circle(radius):
    return 2*PI*radius

def get_area(radius):
    return PI*radius*radius

# r = number_input()


print(__name__)
if __name__ == "__main__": 
    print(get_circle(10)) 
    print(get_area(10))

