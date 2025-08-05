import numpy as np
import pandas as pd

# 칼럼 추가하기( = 파생변수 만들기 )
# 열 = 칼럼 column = 속성 attribute = 변수 variable = 피처 feature
# 행 = 로우 row    = 레크드 record  = 관측치 observation = 데이터포인트 datapoint
df = pd.DataFrame([[90,85,75],[85,95,85],[75,65,95]] , index = ['철수','영희','유리'] , columns =['수학','영어','국어'])


df_origin = df.copy()
df = pd.DataFrame({"수학":[90,85,75],"영어":[85,95,85],"국어":[75,65,90]},index= ['철수','영희','유리'])
df
df['과학'] = [60,50,80]
df['과학'] = [60,50] #에러
df['과학'] = 60
df

df['과힉'] = 60 # 있으면 덮어쓰고 없으면 만들어 버린다.
df

 
#     수학  영어  국어  과학  과힉
# 철수  90  85  75  60  60
# 영희  85  95  65  50  60
# 유리  75  85  95  80  60

#     수학  영어  국어
# 철수  90  85  75
# 영희  85  95  65
# 유리  75  85  95
df = df.iloc[:3,:3]
df

# 칼럼 추가하기(2)
# 총점
#     수학  영어  국어   총점
# 철수  90  85  75  250
# 영희  85  95  65  245
# 유리  75  85  95  255
df["총점"] = df["수학"]+df["영어"]+df["국어"]
df['총점1'] = df.sum(axis=1)
df


# 영어,수학의 평균값을 갖는 칼럼을 추가
df["평균"] = df[['수학', '영어']].mean(axis=1)
df["평균1"] =  (df["수학"]+df["영어"])/2
df
df = df.iloc[:,-2]
df


# 행 추가하기 df.loc[len(df)]
df = df_origin.copy()
df
#     수학  영어  국어
# 철수  90  85  75
# 영희  85  95  85
# 유리  75  65  95





df.iloc[2,:] = 10
df
#     수학  영어  국어
# 철수  90  85  75
# 영희  85  95  85
# 유리  10  10  10
df.loc[3,:] = 10
df
df.loc[30] = [10,20,30]
df.loc[len(df)] = [10,20,30]



#         수학    영어    국어
# 철수    90.0  85.0  75.0
# 영희    85.0  95.0  85.0
# 유리    75.0  65.0  95.0
# 과목평균  83.3  81.7  85.0
df.loc['과목평균'] = df.mean(axis=0).round(1)


# 행, 열 삭제하기
df = df[['수학', '영어']]
df = df.drop('영어',axis=1) #열 삭제
df.drop('영어',axis=1,inplace=True) # inplace=True 열삭제 후 df에 자동으로 덮어쓰기
df.drop(columns='영어')

df = df.drop('영희',axis=0) #행 삭제 

# 사용자 정의함수 적용하기 df.apply(함수)
df.mean(axis=0)
df.mean(axis=1)
df.half(axis=1) # 에러
df.apply(half)
df.apply(sum)
df.apply(sum, axis=1)
df.apply(lambda x : np.round(x/2)+10)
df.applymap(half) #DataFrame.applymap has been deprecated. Use DataFrame.map
df.map(half)

def half(x):
    return np.round(x/2) + 10
half(10)


# 영어점수가 90점 이상인 학생정보
df.where(df['영어'] > 90)
df.where(df > 80, other=1)



# np.where 사용법 
# array([[1, 1, 0],
#        [1, 1, 1],
#        [0, 0, 1]])
np.where(df > 80, 1, 0)


#     수학  영어  국어
# 철수   1   1   0
# 영희   1   1   1
# 유리   0   0   1
df[:] = np.where(df > 80, 1, 0)
df

# 총첨 칼럼을 만들어 봅시다.
#     수학  영어  국어   총점
# 철수  90  85  75  250
# 영희  85  95  85  265
# 유리  75  65  95  235
df['총점'] = df.sum(axis=1)
df

# 250 이상이면 'PASS' 그렇지 않으면 'FAIL' 이라는 '성공여부' 칼럼을 만들어주세요
#     수학  영어  국어   총점
# 철수  90  85  75  250
# 영희  85  95  85  265
# 유리  75  65  95  235

df['성공여부'] =np.where(df['총점']>250,'PASS','FAIL')



#정렬하기 sort_values()
#     수학  영어  국어
# 철수  90  85  75
# 영희  85  95  85
# 유리  75  65  95

#   수학  영어  국어
# 영희  85  95  85
# 철수  90  85  75
# 유리  75  65  95
df.sort_values('영어',ascending=False)

# 철수 85를 95로 바꾸기
df.iloc[0,1] = 95
df
df.sort_values('영어',ascending=False)
df.sort_values(['영어','수학'])


# apply : 한 단계 낮은 것에 적용된다
df.apply(sum)
df['수학'].apply(lambda x : x*2)



def twice(x):
    return x*2

apply(twice) # 함수도 객체다 모든것은 객체다

def print_me(x):
    print(x)
    
df.apply(print_me)

print_me(df['수학'])
print_me(df['영어'])
print_me(df['과학'])
df.apply(sum,axis=1)
    
# groupby
# mpg.csv
df = pd.read_csv("https://gist.githubusercontent.com/omarish/5687264/raw/7e5c814ce6ef33e25d5259c1fe79463c190800d9/mpg.csv")
# df = pd.read_csv("https://gist.githubusercontent.com/omarish/5687264/raw/7e5c814ce6ef33e25d5259c1fe79463c190800d9/mpg.csv")
df.info()

# 연비의 최소값과 최대값은?
df['mpg'].max()
df['mpg'].min()
df['mpg'].mean()


# cylinders의 종류는?
df['cylinders'].unique()

# cylinders의 종류별 개수?
df['cylinders'].value_counts()


# import seaborn as sns
# sns.load_dataset("mpg")

a  = df['name'].str.split(n=1,expand=True)
df[['first'],['two']] = a
df['first'] = a[0]
df['two'] = a[1]
df['first']


# 실린더별로 배기량의 평균
df.groupby('cylinders')['displacement'].mean()
# 제조사별 연비의 최대값은 나타내면
df.groupby('name')['displacement'].max().sort_values(ascending=False)
df.groupby('name')['mpg'].max().sort_values(ascending=False)
