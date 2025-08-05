import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
tips = sns.load_dataset("tips")

sns.scatterplot(data=tips,x='total_bill',y='tip')
sns.regplot(data=tips,x='total_bill',y='tip')
sns.jointplot(data=tips,x='total_bill',y='tip')
sns.jointplot(data=tips,x='total_bill',y='tip',kind='hex')
sns.kdeplot(data=tips,x='total_bill',y='tip',kind='hex')
sns.kdeplot(data=tips,x='total_bill')
sns.kdeplot(data=tips,x='sex') #안된다

sns.kdeplot(data=tips,x='total_bill',y='tip',fill=True)
plt.show()

sns.countplot(data=tips,x='sex',hue='smoker')
plt.show()
sns.barplot(data=tips,x='sex',y='total_bill')
sns.barplot(data=tips,x='sex',y='total_bill',hue='smoker')
sns.barplot(data=tips,x='sex',y='total_bill',hue='smoker',errorbar=None)
sns.boxplot(data=tips,x='day',y='total_bill',hue='day')

sns.violinplot(data=tips,x='sex',y='total_bill')
sns.violinplot(data=tips,x='sex',y='total_bill',hue='sex')
sns.violinplot(data=tips,x='day',y='total_bill',hue='day')

sns.pairplot(data=tips)
sns.pairplot(data=tips,hue='time')
sns.pairplot(data=tips.query(' time == "Dinner"'))
sns.pairplot(data=tips.query(' time == "Dinner"'),hue='time')
plt.show()

sns.scatterplot(data=tips,x='total_bill',y='tip')
sns.scatterplot(data=tips,x='total_bill',y='tip',hue='smoker')

sns.scatterplot(data=tips,x='total_bill',y='tip',hue='smoker',size="size")

###############################################################################
# 6교시
###############################################################################

# pd.read_excel("./서울지역대학교위치.xlsx", engine="openpyxl")
df = pd.read_excel("서울지역대학교위치.xlsx")
df[:2]


# 지도 lib
# 지도 참조 사이트
# https://plotly.com/examples/
# https://d3js.org/ 

import folium
folium.Map(location=[37.55,126.98])
plt.show()
