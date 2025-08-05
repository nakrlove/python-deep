import pandas as pd
df =pd.read_csv('https://github.com/EasysPublishing/do_it_pandas/tree/main/data/gapminder.tsv',sep='\t')
# df =pd.read_csv('gapminder.tsv',sep='\t')
df


df =pd.read_csv('https://raw.githubusercontent.com/EasysPublishing/do_it_pandas/main/data/gapminder.tsv',sep='\t')
df


df.iloc[[0,1,5],[0,2,3]]
df.loc[[0,1,5],['country','year','lifeExp']]

df.iloc[0:10, : ]
df.iloc[:10]
df.head(10)
df[:10]
#  뒤에서 10개
df[ -10:]


df.iloc[ -10: , 2:]
df.loc[ 5:10,['country','year','lifeExp']]
df.loc[ [0,99,999],['country','year','lifeExp']]


df


df.groupby('year')['lifeExp'].mean()
df.groupby(['year','continent'])['lifeExp'].mean()
df['country']
len(df['country'].unique())
len(set(df['country']))

df['country'].nunique()

# 빈도수 세기
df['country'].value_counts()
df['continent'].value_counts()
s = pd.Series(['banana',42])
pd.DataFrame(['banana',42])


pd.Series(['banana',42])
pd.Series(['banana',42],index=[10,20])
s = pd.Series(['banana',42],index=['영희','철수'])
s.index # index만 보기


# 데이터프레임 만들기
df = pd.DataFrame([[1,2,3,4],[5,6,7,8]])
df.index
df.columns

# 인덱스 이름 부여하기
#      0  1  2  3
# 철수  1  2  3  4
# 영희  5  6  7  8
df.index = ['철수','영희']

# 컬럼이름 부여하기
#     국어  영어  수학  과학
# 철수   1   2   3   4
# 영희   5   6   7   8
df.columns =['국어','영어','수학','과학']
df
df.T
df.loc
df.shape
df.size
df.values
df.to_numpy()


# 칼럼 이름을 일부분만 바꾸기
df.index
df.columns[0] 
df.columns[1] = "English" #에러발생함
df.columns = ['국어','English','수학','과학']
df
df = df.rename(columns={"영어":"english"})
df
df.rename({"english":"영어"},axis=1)


df = pd.DataFrame([[1,2,3,4],[5,6,7,8]] , index = ['철수1','영희1'] , columns =['국어','영어','수학','과학'])
df

# 데이터프레임 만들기 2 : 딕셔너리로 만들기
#    a  b
# 0  1  4
# 1  2  5
# 2  3  6
pd.DataFrame({"a":[1,2,3],"b":[4,5,6]})

#    a  b
# A  1  4
# B  2  5
# C  3  6
pd.DataFrame({"a":[1,2,3],"b":[4,5,6]} , index=['A',"B","C"])



df =pd.read_csv('https://raw.githubusercontent.com/EasysPublishing/do_it_pandas/main/data/gapminder.tsv',sep='\t')
df

ages[ages > ages.mean]

df.iloc[[1,3,4,5],:]


import numpy as np # alias

d = np.random.randint(100)
arr = d.tolist()


li = np.random.randint(0,df.size, 100)  # 0~99 랜덤 정수 100개
df.index.stop
print(lst)
df.iloc[ li.tolist() , :]



df = df.sample(100)
df = df[['country','lifeExp']]
df
df['lifeExp'].mean()
df[df['lifeExp'] > df['lifeExp'].mean()]

df.query(" lifeExp > lifeExp.mean()")
