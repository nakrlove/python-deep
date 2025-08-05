# 1에서 65까지 숫자를 랜덤으로 30개를(6,5)모양의 배열로 만들어주세요

import numpy as np
a = np.random.randint(1,66, (6,5))
a

a = np.random.randint(1,66,30,).reshape(6,5)
a
# 위값을 중복없이 뽑아내라
np.random.choice(np.arange(1,66), (6,10), replace=False)
np.random.choice([1,2,3], (6,5), replace=True,p=[0.5,0.4,0.1])

# 데이터프레임 만들기
#     수학  영어  국어
# 철수  90  85  75
# 영희  85  95  85
# 유리  75  65  95
import pandas as pd
df = pd.DataFrame([[90,85,75],[85,95,85],[75,65,95]] , index = ['철수','영희','유리'] , columns =['수학','영어','국어'])

df = pd.DataFrame({"수학":[90,85,75],"영어":[85,95,85],"국어":[75,65,95]},index= ['철수','영희','유리'])
df

# 컬럼 이름 바꾸기
#  영어 -> 과학
#     수학  과학  국어
# 철수  90  85  75
# 영희  85  95  85
# 유리  75  65  95
df = pd.DataFrame([[90,85,75],[85,95,85],[75,65,95]] , index = ['철수','영희','유리'] , columns =['수학','영어','국어'])
# df.columns = ["수학", "국어","과학"]
df = df.rename(columns={"영어":"과학"})
df

df.rename({"영어":"과학"},axis=1)

# index 바꾸기
df.index = np.random.choice( ['철수','영희'],3,p=[0.7,0.3])
df
