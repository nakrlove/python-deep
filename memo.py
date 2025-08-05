import sys
# print(sys.argv[0])
# print(sys.argv[1])
# print(sys.argv[2])


if sys.argv[1] == '-w':
    with open("memo.txt","w") as f:
        f.write(sys.argv[2])


if sys.argv[1] == '-a':
    with open("memo.txt","a") as f:
        f.write(sys.argv[2])
        