# 결측치가 중요한 이유는?
# - 결측치를 제대로 처리 안하면 분석결과가 왜곡 될 수 있다.
# 결측치를 처리하는 방법들은 ?
# 1 결측치를 행을 삭헤함
# 2. 결측치 위의 값을 가져와 채움 df['columns'].ffill()
# 3. 결측치 위의 값을 가져와 채움 df['columns'].ffill()

# 어떤 셀(data)을 결측치로 만드는 방법은?
# 


# 결측치의 개수를 파악하는 방법은?
# isnull()

# 


import pandas as pd
url = "https://raw.githubusercontent.com/sidsriv/Introduction-to-Data-Science-in-python/master/mpg.csv"
mpg = pd.read_csv(url)
mpg



# mpg 데이터셋을 불러옵니다.
# mpg.csv -> url로 가져오기
# 라이버러리에서 제공해주는 데이터셋
# 'origin'과 'cylinders'별로 'horsepower'의 평균을 구해봅시다.
import seaborn as sns
mpg = sns.load_dataset("mpg")
mpg.info()


mpg.groupby(['origin','cylinders'])[['horsepower']].mean()
mpg.groupby(['origin','cylinders'])['horsepower'].mean()
type(mpg['origin'])
mpg[['origin']]
mpg[['origin','mpg']]



import pandas as pd
pd.DataFrame([1,2,3])
pd.Series([1,2,3])

# 객체를 파일로 저장하기
my_data = mpg.iloc[:3,:3]
my_data.index = ['영희','철수','민수']
my_data.to_csv("test.csv")
my_data.to_excel("test.xlsx")
my_data.to_pickle("test.pickle")

# my_data.to
my_data

# 파일읽기
data = pd.read_csv("./test.csv",index_col=0 , nrows=2)
data
data = pd.read_excel("./test.xlsx", engine="openpyxl")
data
a = pd.read_clipboard()
a
import os
os.startfile("test.xlsx")

import pickle
with open("test.pickle" , "wb") as f:
      pickle.dump(my_data,f)

with open("test.pickle" , "rb") as fr:
      pickle.load(fr)
        
     
# tidy


# wide <--> long
url = "https://raw.githubusercontent.com/kirenz/datasets/refs/heads/master/gapminder.csv"
df = pd.read_csv(url)
df
gapminder = pd.read_csv(url)
df = df[['country','year','lifeExp']]


pd.pivot(df,index='country',columns='year',values='lifeExp')
pd.pivot(df,index='year',columns='country',values='lifeExp')


gapminder.info()
gapminder.pivot(index='year',columns='country',values='lifeExp')
gapminder.pivot(index='year',columns='country',values='pop')
gapminder.pivot(index='year',columns='country',values='gdpPercap')

gapminder.pivot(index='continent',columns='country',values='lifeExp')
# pivot() -> pivot_table()
gapminder.pivot_table(index='continent',columns='year',values='lifeExp')
gapminder.pivot_table(index='continent',columns='year',values='lifeExp',aggfunc='min')

def test(x):
      return x.max()

gapminder.pivot_table(index='continent',columns='year',values='lifeExp',aggfunc=test)



df.pivot()


# groupby로 나타내기
# 대륙별 연도별 기대수명의 평균값
gapminder.groupby(['continent','year'])['lifeExp'].mean()  # Serial형식 
gap = gapminder.groupby(['continent','year'])[['lifeExp']].mean() # DataFrame 형식
gap.shape
gap.unstack()

gap = gapminder.groupby(['continent','year'],as_index=False)[['lifeExp']].mean() # DataFrame 형식
gap

# 위 gap을 wide 형식으로 pivot하기
gap.pivot(index="continent",columns="year", values="lifeExp")


# pivot함수 도움말로 테스트
df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two','two'],
                    'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                    'baz': [1, 2, 3, 4, 5, 6],
                    'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
df.pivot(index='foo', columns='bar', values='baz')



# wide 형을 long 형으로 바꾸기 melt()
df = gapminder.pivot(index="country",columns="year", values="lifeExp")
df
df.melt()

df = df.reset_index()
df1 = pd.melt(df,id_vars='country',value_vars=[1952,1957])
df1
df1 = pd.melt(df,id_vars='country',value_vars=[1952,1957,1962], value_name='lifeExp')
df1 = pd.melt(df,id_vars='country',value_vars=[1952,1957,1962], var_name="year", value_name='lifeExp')

df1 = pd.melt(df, id_vars='country', value_vars=[1952,1957,1962], value_name='lifeExp_value')

df1 = df1.reset_index()
df1 = df1.sort_values('country')
df1.set_index("country")



tips = sns.load_dataset('tips')
tips.info()
tips[['total_bill','tip']].sum()
# tips['tip'] # 문자형 object형으로 바꾸기
tips['tip'] = tips['tip'].astype('object')
tips[['total_bill','tip']].mean()

tips['tip'] = tips['tip'].astype('float')
