print()
print()
print(1+1)
#print()
#print()
#print()
print("Hello")
def test():
    print("우리나라")
    print("대한민국") 



import subprocess
import datetime


day = datetime.date(2000,12,3)
print(day.weekday())

with open('test.txt', 'wb') as f:
    out = subprocess.run(['cmd', '/c', 'dir'], capture_output=False)
    out.stdout
    #f.write()

