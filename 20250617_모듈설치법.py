
# 1)
# pip freeze > requirements.txt

# 2)
# pip install -r requirements_fixed.txt




# //파일이 잘못 만들어 졌을때 gpt가 답변한것
# 다중 커서로 정렬된 공백을 한 번에 ==로 바꾸기

# <잘못된 형식>
#  aiohappyeyeballs              2.6.1
#  requests                      2.32.0
#  numpy                         1.26.4

# <올바른 형식>
# aiohappyeyeballs==2.6.1
# requests==2.32.0
# numpy==1.26.4

with open('requirements.txt', 'r') as f:
    lines = f.readlines()

with open('requirements_fixed.txt', 'w') as f:
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            f.write(f"{parts[0]}=={parts[1]}\n")