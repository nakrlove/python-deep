# ##########################################################
# 오전 질문사항들 정리함 
# ##########################################################

# matplolib에서 지금까지 만났던 그래프들의 서술하시오
# (hist,count,scatter,box,rug,violin,bar,density)

# long형의 데이터프레임을 wide형으로 변환하기 위한 메서드는 무엇일까요? pivot
# wide형의 데이터프레임을 long형으로 변환하기 위한 메서드는 무엇일까요? melt

# 데이터 시각화를 할때 생각해 둘 사항은 무엇일까요?
# - 그림의 종류
# - 변수의 개수 (=칼럼의 수 = 차원의 수)
# - 변수의 데이터 타입 (숫자형,문자형,범주형,날짜형) Category,Factory .... datetime
#  

# 여러 오브젝트들을 이미지화해서 캡쳐, 01010 00101,101101 등의 바이너리 데이터로 구성하여 저장하고 
# 언제든지 그 형태 그대로 다시 불러오기 가능합니다. 이러한 원리로 제작된 라이브러리 이름은?  pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
titanic = sns.load_dataset("titanic")
titanic[:2]


plt.rcParams['font.family'] = 'Malgun Gothic' # 한글표시

sns.barplot(data=titanic,x='class',y='fare',hue='sex')

# error : 오차
# error bar : 오차막대
sns.barplot(data=titanic,x='sex',y='fare',hue='class' ,errorbar=None)
plt.title('성별,등급별 평균 요금')
plt.show()


titanic[['class','fare','sex']]


# 남여의 숫자 hue는 색상을 다르게 표현
sns.countplot(data=titanic,x='sex',hue='sex')
sns.countplot(data=titanic,x='class',hue='sex')
sns.countplot(data=titanic,x='fare',hue='sex')
plt.show()

# url = "https://raw.githubusercontent.com/sidsriv/Introduction-to-Data-Science-in-python/master/mpg.csv"
# mpg = pd.read_csv(url)

mpg = sns.load_dataset("mpg")

mpg.info()
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   mpg           398 non-null    float64
#  1   cylinders     398 non-null    int64   <--원래 타입
#  2   displacement  398 non-null    float64
#  3   horsepower    392 non-null    float64
#  4   weight        398 non-null    int64
#  5   acceleration  398 non-null    float64
#  6   model_year    398 non-null    int64
#  7   origin        398 non-null    object
#  8   name          398 non-null    object


# cylinders 타입을 카테고리형으로 바꾸기
mpg['cylinders'] =  mpg['cylinders'].astype("category")

mpg.info()
 #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   mpg           398 non-null    float64
#  1   cylinders     398 non-null    category  <-- 타입이 변경됨
#  2   displacement  398 non-null    float64
#  3   horsepower    392 non-null    float64
#  4   weight        398 non-null    int64
#  5   acceleration  398 non-null    float64
#  6   model_year    398 non-null    int64
#  7   origin        398 non-null    object
#  8   name          398 non-null    object


# 'origin' 'cylinders' 별로 mpg의 최대값은?
mpg.groupby(['origin','cylinders'])['mpg'].max() 
df = mpg.groupby(['origin','cylinders'])['mpg'].agg(['max','min']) #['max','min'] 시리즈가 아닌 데이터프레임을 구함
df = mpg.groupby(['origin','cylinders'])['mpg'].agg(['max']) #['max'] 시리즈가 아닌 데이터프레임을 구함
df = mpg.groupby(['origin','cylinders'],as_index=False)['mpg'].agg(['max']) #index 다르게 처리
df

sns.barplot(data=df,x='origin',y='max',hue='cylinders')
sns.barplot(data=df,x='cylinders',y='max',hue='origin')
plt.show()



# 검색하기 cylinders가 8인 데이터들만 뽑기
# 1) True,False 검색
mpg[mpg['cylinders'] == 8]
# 2) 시리즈.isin() 메소드 사용
mpg[mpg['cylinders'].isin([8])]
# 3) df.query('')를 이용
mpg.query('cylinders == 8')

# cylinders가 8이고 origin은 미국인 데이터들만 뽑기
# 1) True,False 검색
mpg['cylinders'] == 8 & mpg['origin'] == 'usa' #에러
mpg[(mpg['cylinders'] == 8) & (mpg['origin'] == 'usa')]

condition1 = mpg['cylinders'] == 8
condition2 = mpg['origin'] == 'usa'
mpg[condition1 & condition2]

# 2) cylinders가 8이거나 3인 데이터들만 뽑기
condition1 = mpg['cylinders'] == 8 
condition2 = mpg['cylinders'] == 3
mpg[(condition1 | condition2)]['cylinders'].unique()
mpg

# 2) 시리즈.isin() 메소드 사용
mpg['cylinders'].isin([8,3])
mpg[mpg['cylinders'].isin([8,3])]
mpg[mpg['cylinders'].isin([8]) | mpg['origin'].isin(['usa'])]
# ######################################################
# 3교시 #################################################
# ######################################################
# 3) df.query('')를 이용
mpg.query(' cylinders == 8 or cylinders == 3 ')
mpg.query(' cylinders == 8 | cylinders == 3 ')


mpg.query(' cylinders == 8 & origin == "usa" ')
mpg.query(' cylinders == 8 | origin == "usa" ')

mpg.query(' cylinders == 8 and origin == "usa" ')
mpg.query(' cylinders == 8 or origin == "usa" ')
mpg.query(' mpg < 18 and origin == "usa" ')

condition1 = mpg['mpg'] < 18
condition2 = mpg['origin'] == 'usa'
mpg[ condition1 & condition2]

# 문자열인 칼럼을 검색하고 싶을때
mpg = sns.load_dataset("mpg")
mpg['name']
mpg['name'] = 'chevrolet chevelle malibu'
mpg[mpg['name'] == 'chevrolet chevelle malibu']

mpg['name'].str.startswith('chevrolet')
mpg[mpg['name'].str.startswith('chevrolet')]
mpg[mpg['name'].str.endswith('air')]
mpg[mpg['name'].str.contains('malibu')]

# gapminder로 연습해 봅시다.
df = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/refs/heads/master/gapminder.csv")
df
# 1. Aisa만 고르기
df[df['continent'] == 'Asia']
df[df['continent'].str.startswith('Asia')]

# 2. Aisa 또는 Africa  고르기
df[df['continent'].str.startswith('Asia') | df['continent'].str.startswith('Africa')]
df.query(' continent == "Asia" or  continent == "Africa"')


# 3  Aisa이고 year가 2000년 이후 데이터만 고르기
df.query('continent == "Asia" and  year > 2000')

# 4  Aisa이고 lifeExp가 80이상인 나라의 이름은?
df.query('continent == "Asia" and  lifeExp > 80')['country']

df.query('continent == "Asia" and  lifeExp > 80')['country'].to_list()
set(df.query('continent == "Asia" and  lifeExp > 80')['country'].to_list()) #중복제거

# 4  Aisa이고 lifeExp가 80이상인 값에 대하여 국가별 lifeExp평균은?
df.query('continent == "Asia" and  lifeExp >= 80').groupby('country')[['lifeExp']].agg('mean')
temp = df.query('continent == "Asia" and  lifeExp >= 80').groupby('country')[['lifeExp']].agg('mean').reset_index()

temp[0,0] = 'Hong Kong'
# bar 그래프를 ..
sns.barplot(data=temp,x='country',y='lifeExp',hue='country')
plt.ylim(80,83)
plt.show()


temp = df.query('continent == "Asia" and  lifeExp >= 80').groupby('country')[['lifeExp']].agg('mean')
temp.plot(kind='bar',y='lifeExp') #index가 X축으로 자동으로 잡힌다.
plt.show()


#  4교시 ##########################################
# 한국 데이터만 골라봅시다.
df.query("country == 'Korea, Dem. Rep.'")
df.query("country == 'Korea'")
df[df['country'].str.contains('Korea')]

df.loc[df['country'] =='Korea, Dem. Rep.','country'] = 'North Korea'
df['country'] =df['country'].replace('Korea, Rep.','South Korea')

df[df['country'].str.contains('Korea')]
df[df['country'].str.contains('South Korea')].plot(kind='line',x='year',y='lifeExp')
plt.show()

temp = df[df['country'].str.contains('South Korea')]
sns.lineplot(data=temp,x='year',y='lifeExp')
plt.show()



temp1 = df[df['country'].str.contains('Korea')] #남한과 북한을 모두 선그리기
sns.lineplot(data=temp1,x='year',y='lifeExp',hue='country')
plt.show()
