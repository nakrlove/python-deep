# 문자열이나 숫자 데이터를 굳이 카테고리형으로 바꿔야 할까요?
# 데이터프레임에서 원하는 컬럼을 검색할때 쓰는방법 3가지는 무엇일까요?
#  1) True,False 검색 Boolean Indexing
#  2) 시리즈.isin() 메소드 사용
#  3) df.query('')를 이용


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
gapminder = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/refs/heads/master/gapminder.csv")
gapminder

# 1) True,False 검색
# mpg[mpg['cylinders'] == 8]
# 2) 시리즈.isin() 메소드 사용
gapminder[gapminder['year'].isin([2002,2007])]
gapminder['year'].isin([2002])
# 3) df.query('')를 이용
gapminder.query('year == 2002 or year == 2007')
gapminder.query('year.isin([2002,2007])')

# 년도가 2002년이고 lifeExp가 60이상인 것만 뽑으면?
gapminder.query('year == 2002 and lifeExp >= 60')
gapminder.info()

# continent를 category형으로 바꾸어 주세요
gapminder['continent'] =  gapminder['continent'].astype("category")
gapminder.info()

gapminder['country'] =  gapminder['country'].astype("category")
gapminder.info()

gapminder['country']

# 다시 문자열(object = str)형으로 바꾸자
gapminder['country'] =  gapminder['country'].astype("str")
gapminder['country'] =  gapminder['country'].astype("object")

# 대륙별, 연도별, 평균gdp 변화
gapminder['continent']

gapminder.groupby(['continent','year'],as_index=False)['gdpPercap'].agg(['mean'])
gapminder.groupby(['continent','year'])['gdpPercap'].agg(['mean'])


df = gapminder


sns.lineplot(data=gapminder,x='year',y='gdpPercap',errorbar=None)
sns.lineplot(data=gapminder,x='year',y='gdpPercap',hue='continent')

sns.barplot(data=gapminder,x='cylinders',y='max',hue='origin')
plt.show()

# year칼럼을 datatime으로 바꾸기
df['year'].astype('datetime') #이렇게 하면 안됨
df['year'].astype('datetime64[ns]') #이렇게 해야 된다
df['year'] = pd.to_datetime(df['year']) #공식적인 방법
pd.to_datetime(df['year'],format="%Y")

gapminder.info()
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   country    1704 non-null   object
#  1   continent  1704 non-null   object
#  2   year       1704 non-null   int64     <-- type
#  3   lifeExp    1704 non-null   float64
#  4   pop        1704 non-null   int64
#  5   gdpPercap  1704 non-null   float64
# dtypes: float64(2), int64(2), object(2)

df.info()
 #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   country    1704 non-null   object
#  1   continent  1704 non-null   object
#  2   year       1704 non-null   datetime64[ns]  <-- type
#  3   lifeExp    1704 non-null   float64
#  4   pop        1704 non-null   int64
#  5   gdpPercap  1704 non-null   float64
# dtypes: datetime64[ns](1), float64(2), int64(1), object(2)


df[(df['year'] >= "1952-01-01") & (df['year'] <= '1962-01-01')]
df[(df['year'] >= "1952") & (df['year'] <= '1962')]

df[df['year'] >= "1962"]
df[df['year'] >= "1962-03"]
df[df['year'] >= "1962-03-01"]

# df.reset_index()
# df.set_index('칼럼')
# df.sort_index()
# df.reindex()

df =df.set_index('year')
df.sort_index(inplace=True)
df.loc["1963":"1963"]
df.info()







url = "https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/economics.csv"
economics = pd.read_csv(url)
economics.info()


df = economics
# 1. date 칼럼을 datetime으로 바꾸어 주세요
# 2. sns로 x = data , y = unemploy로 선그래프를 그려주세요.
# 3. date칼럼을 index로 넣어주세요
# 4. 1999년 1월부터 2000년 12월까지 데이터만 뽑아 주세요
# 5. 선그래프를 그려주세요 

# 1
df['date'] = df['date'].astype('datetime64[ns]') #공식적인 방법
df['date'] = pd.to_datetime(df['date']) #공식적인 방법
df.info()

# 2
sns.lineplot(data=df,x='date',y='unemploy')
sns.lineplot(data=df,x='date',y='unemploy',errorbar=None)
plt.show()

# 3 ?
df =df.set_index('date')

# 4
df = df.loc["1999-01":"2000-12"]

# 5
df.plot(y='unemploy')



sns.lineplot(data=df,x=df.index,y='unemploy')
plt.show()

df



# 두 데이터프레임 합치기 merge,join,concat
# merge 필수 : 칼럼이름으로 merge하기
df1 = pd.DataFrame({'key': ['a', 'a', 'b', 'b', 'c', 'c','d', 'd'],
                    'data1': range(8)})

df2 = pd.DataFrame({'key': ['a', 'b', 'd', 'e'],
                    'data2': range(4)})

df3 = pd.DataFrame({'name': ['a', 'b', 'd', 'e'],
                    'data2': range(4)})

df1
df2

pd.merge(df1,df2) #공통으로 있는 행
pd.merge(df1,df2, on='key') #명시적으로 공통 칼럼이름을 준다.
pd.merge(df1,df3, left_on='key',right_on='name') #명시적으로 공통 칼럼이름을 준다. 
pd.merge(df1,df2 , how='inner') 
pd.merge(df1,df2 , how='outer') 

df1.merge(df2) #메소드를 이용
pd


# index로 merge하기 
left = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                     index=['a', 'c', 'e'],
                     columns=['Ohio', 'Nevada'])
right = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                      index=['b', 'c', 'd', 'e'],
                      columns=['Missouri', 'Alabama'])

pd.merge(left,right,how='outer')
pd.merge(left,right,left_index=True,right_index=True)
pd.merge(left,right,left_index=True,right_index=True ,how='outer')

# index로 merge하기 -> pd.concat()을 주로 사용

# pd.concat()으로 두 데이터프레임을 하나로 만들기 
# index를 기준으로 생각할 때
df1 = pd.DataFrame({'a':['a0','a1','a2'],
                   'b':['b0','b1','b2'],
                   'c':['c0','c1','c2']},
                  index = [0,1,2])

df2 = pd.DataFrame({'b':['b2','b3','b4'],
                   'c':['c2','c3','c4'],
                   'd':['d2','d3','d4']},
                   index = [1,2,3])

pd.concat([df1,df2])
pd.concat([df1,df2],join='inner')
pd.concat([df1,df2],axis=1)
pd.concat([df1,df2],join='inner',axis=0)
pd.concat([df1,df2],join='inner',axis=1)
pd.concat([df1,df2],ignore_index=True)



left = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                     index=['a', 'c', 'e'],
                     columns=['Ohio', 'Nevada'])
right = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                      index=['b', 'c', 'd', 'e'],
                      columns=['Missouri', 'Alabama'])

pd.concat([left,right] , axis=1 , sort=True)
pd.concat([left,right],join='inner',axis=1)
pd.merge(left,right,left_index=True,right_index=True)
pd.merge(left,right,left_index=True,right_index=True ,how='outer')