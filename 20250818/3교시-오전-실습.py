
#  name = "강휘준,길명철,김대영,김동건,김정호,김지건,손규성,양현성,우상익,이대식,이선우,이세희,이용환,이재준,이재준,장유정,정경주,정경환,조시현,채창도"
#20명을 5명씩 4개조로 나누어 주세요


import random


name = "강휘준,길명철,김대영,김동건,김정호,김지건,손규성,양현성,우상익,이대식,이선우,이세희,이용환,이재준,이재준,장유정,정경주,정경환,조시현,채창도"

# 문자열 → 리스트 변환
names = name.split(",")

# 랜덤 섞기
random.shuffle(names)

# 5명씩 4개 조로 나누기
groups = [names[i:i+5] for i in range(0, len(names), 5)]

# 결과 출력
for idx, group in enumerate(groups, start=1):
    print(f"{idx}조: {group}")
