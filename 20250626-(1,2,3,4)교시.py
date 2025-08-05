import pandas as pd



# 문제 
# 1)아래와 같이 데이터프레임을 만들어 보세요
#    이름  국어  영어
# 0  철수  90  85
# 1  영희  80  95
# 2  민수  70  75
df = pd.DataFrame([['철수',90,85],['영희',80,95],['민수',70,75]] ,  columns =['이름','국어','영어'])
df


# 2)국어,영어의 평균을 구하여 '평균' 칼럼을 추가해 주세요
df["평균"] = (df["국어"]+df["영어"])/2

df["평균"] = df[['국어', '영어']].mean(axis=1)
df

# df.sum(axis=1)

# 2)국어가 90점 이상이면 'A', 80점 이상이면 'B', 나머지는  'C'로 등급을 매기는 새로운 칼럼
#'국어등급'을 추가해 주세요
# 방법 1
import numpy as np
df['국어등급'] = np.where(df['국어'] >= 90 , 'A',np.where(df['국어'] >= 80 , 'B','C'))
df


# 방법 2
def jumsu(x):
    return "A" if x >= 90 else "B" if x >= 80 else "C"


df['국어'].apply(jumsu)           
df['국어'].map(jumsu)        
df[['국어','영어']].map(jumsu)  

# 방법 3
# pd.cut 
pd.cut(df['국어'],bins=[80,90])
df['등급'] = pd.cut(df['국어'],bins=[0,80,90,100],  labels=['A','B','C'] ,right=False)



import seaborn as sns
mpg = sns.load_dataset("mpg")
mpg.info()

mpg.groupby('origin').mean() #문자값 칼럼때문에 에러 발생함 
mpg.groupby('origin').iloc[:,:6] # 에러가 발생함 

mpg.iloc[:, :6].groupby('origin')

mpg.groupby('origin')[['mpg','cylinders','displacement']].mean()
mpg.groupby('origin')[['mpg','cylinders','displacement']].sum()
mpg.groupby('origin').sum()
mpg.groupby('origin').sum()[['mpg','cylinders','displacement']]

mpg[['mpg','cylinders','displacement']].groupby('origin').sum() #  origin컬럼이 없어서 에러가 남


mpg.groupby('origin')[['mpg','cylinders','displacement']].apply(np.sum,axis=0)
mpg.groupby('origin')[['mpg','cylinders','displacement']].agg(np.sum,axis=0)



# 판다스 입문 책 256,257 page 참조
def test(x):
    result = x.sum()/len(x)
    return result

mpg.groupby('origin')[['mpg','cylinders','displacement']].agg(test)
mpg.groupby('origin')[['mpg','cylinders','displacement']].agg([np.mean,sum])
mpg.groupby('origin')[['mpg','cylinders','displacement']].agg(['mean','sum'])
mpg.groupby('origin')[['mpg','cylinders','displacement']].agg({"mpg":"mean","cylinders":"sum","displacement":test})
mpg.groupby('origin')[['mpg','cylinders','displacement']].agg({"mpg":"mean","cylinders":"sum"})


mpg.groupby(['origin','cylinders'])


for name,data in mpg[['origin','cylinders','displacement']].groupby(['origin','cylinders']):
    print(name)
    print(data)
    
mpg.groupby(['origin','cylinders'])[['displacement','mpg']].agg('mean')   


np.random.seed(1234)
np.random.randn(2,3)
np.random.randn(2,3)


np.random.seed(1234)
np.random.normal(80,5,(2,3)) 

(np.random.normal(80,5,(2,3)) - 80) / 5 

# 직접 표준화 해보기
def standardize(x):
    return (x - x.mean() ) / x.std()



mpg[['mpg','displacement']].apply(standardize)
mpg[['mpg','displacement']].agg(standardize)
mpg[['mpg','displacement']].transform(standardize)




# 컬럼 제거 연습
url = "https://raw.githubusercontent.com/sidsriv/Introduction-to-Data-Science-in-python/master/mpg.csv"
mpg = pd.read_csv(url)
mpg.iloc[:,1:]
mpg[:2]
# 제조사별로 연비의 최소값

mpg = mpg.goroupby() 

import seaborn as sns
sns.load_dataset("mpg")

# 결측지
# iris,titanic,mpg,gapminder,tips,diamonds
titanic = sns.load_dataset("titanic")
titanic.info()
titanic

# 결측지(빈칸) 찾기
#  - isnull()
titanic.isnull().sum()
titanic['sex'].value_counts()
titanic.value_counts()
titanic.iloc[0,0]  = np.nan
titanic[:2]
import numpy as np
np.sum(titanic.isnull())


# 결측지 처리하기
# - 결측치 행 삭제하기
# - 결측치 채우기
#    1.적당한 값으로 채우기 
#    2.바로 위 셀의 값으로 채우기
#    3.바로 아래 셀의 값으로 채우기
titanic.isnull().sum()
titanic[titanic['embarked'].isnull()]

# 일부 행만 삭제하기
#      survived  pclass     sex   age  ...  deck  embark_town  alive  alone
# 0         NaN       3    male  22.0  ...   NaN  Southampton     no  False
# 1         1.0       1  female  38.0  ...     C    Cherbourg    yes  False
# 2         1.0       3  female  26.0  ...   NaN  Southampton    yes   True
# 3         1.0       1  female  35.0  ...     C  Southampton    yes  False
# 4         0.0       3    male  35.0  ...   NaN  Southampton     no   True
# ..        ...     ...     ...   ...  ...   ...          ...    ...    ...
# 886       0.0       2    male  27.0  ...   NaN  Southampton     no   True
# 887       1.0       1  female  19.0  ...     B  Southampton    yes   True
# 888       0.0       3  female   NaN  ...   NaN  Southampton     no  False
# 889       1.0       1    male  26.0  ...     C    Cherbourg    yes   True
# 890       0.0       3    male  32.0  ...   NaN   Queenstown     no   True

# [889 rows x 15 columns]
titanic.dropna(subset=['embarked'])


# 하나라도 na가 있으면 그행을 모두 지운다
#      survived  pclass     sex   age  ...  deck  embark_town  alive  alone
# 1         1.0       1  female  38.0  ...     C    Cherbourg    yes  False
# 3         1.0       1  female  35.0  ...     C  Southampton    yes  False
# 6         0.0       1    male  54.0  ...     E  Southampton     no   True
# 10        1.0       3  female   4.0  ...     G  Southampton    yes  False
# 11        1.0       1  female  58.0  ...     C  Southampton    yes   True
# ..        ...     ...     ...   ...  ...   ...          ...    ...    ...
# 871       1.0       1  female  47.0  ...     D  Southampton    yes  False
# 872       0.0       1    male  33.0  ...     B  Southampton     no   True
# 879       1.0       1  female  56.0  ...     C    Cherbourg    yes  False
# 887       1.0       1  female  19.0  ...     B  Southampton    yes   True
# 889       1.0       1    male  26.0  ...     C    Cherbourg    yes   True

# [182 rows x 15 columns]
titanic.dropna()

# 
# - fillna()
titanic.info()
titanic[:10]
titanic['age'].fillna(20)
titanic['age'].bfill()[:10]
titanic['age'].fillna(method='bfill')[:10]
titanic['age'].ffill()[:10]
titanic['age'].fillna(method='ffill')[:10]
titanic['age'].fillna(200)[:10]
titanic['age']

age_mean= titanic['age'].mean()
titanic['age'].fillna(age_mean)[:10]





