

import pandas as pd

import seaborn as sns
mpg = sns.load_dataset("mpg")
mpg.info()
mpg
mpg.groupby('origin').mean() #문자값 칼럼때문에 에러 발생함 
mpg.groupby('origin').iloc[2,2] # 에러가 발생함 
mpg.iloc[:, :6]
r = mpg.iloc[:, :6].groupby('weight')
r



url = "https://raw.githubusercontent.com/sidsriv/Introduction-to-Data-Science-in-python/master/mpg.csv"
mpg = pd.read_csv(url)