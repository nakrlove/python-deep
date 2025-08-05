# 시각화
# -matplotlib.pyplot
# -seaborn
# -pandas의 시각화
import seaborn as sns
import pandas as pd
import numpy as np

anscombe = sns.load_dataset("anscombe")
print(anscombe)

ans = sns.load_dataset("anscombe")
ans
dataset_1 = ans[ans['dataset'] == 'I']
dataset_2 = ans[ans['dataset'] == 'II']
dataset_3 = ans[ans['dataset'] == 'III']
dataset_4 = ans[ans['dataset'] == 'IV']


import matplotlib.pyplot as plt
plt.plot(dataset_1['x'],dataset_1['y'],'ro')
plt.plot(dataset_2['x'],dataset_2['y'],'o')
plt.plot(dataset_3['x'],dataset_3['y'],'o')
plt.plot([0,1],[10,20],'ro')


# 방법 1
fig = plt.figure()
axes1 = fig.add_subplot(2,2,1)
axes2 = fig.add_subplot(2,2,2)
axes3 = fig.add_subplot(2,2,3)
axes4 = fig.add_subplot(2,2,4)


axes1.plot(dataset_1['x'],dataset_1['y'],'o')
axes2.plot(dataset_2['x'],dataset_2['y'],'o')
axes3.plot(dataset_3['x'],dataset_3['y'],'o')
axes4.plot(dataset_4['x'],dataset_4['y'],'o')
fig.suptitle("Title")
axes1.set_title("No1")
axes2.set_title("No2")
axes3.set_title("No3")
axes4.set_title("No4")



# 방법 2
fig, axes = plt.subplots(2,2,figsize=(8,2))


axes[0][0].plot(dataset_1['x'],dataset_1['y'],'o')
axes[0][0].plot(dataset_2['x'],dataset_2['y'],'o')
axes[0][0].plot(dataset_3['x'],dataset_3['y'],'o')
axes[0][0].plot(dataset_4['x'],dataset_4['y'],'o')
plt.tight_layout()


plt.show()




# 히스토그램 그리기
# 방법1 fig,ax
fig, ax = plt.subplots()

ax.hist(tips['total_bill'], bins=8, linewidth=0.5, edgecolor="white")

plt.style.available
plt.style.use('Solarize_Light2')
plt.style.use('tableau-colorblind10')


plt.show()


# 방법2 plt.hist
plt.hist(tips['total_bill'], bins=8, linewidth=0.5, edgecolor="white")
plt.show()




fig, ax = plt.subplots()
ax.scatter(tips['day'], tips['total_bill'])

plt.show()


tips = sns.load_dataset("tips")
tips[:4]

# 방법3 tips.plot(kind='hist',
tips.plot(kind='hist', y='total_bill')
plt.show()


# 방법4 seaborn
sns.histplot(data=tips,x='total_bill')
sns.histplot(data=tips,x='total_bill',color="#000000",edgecolor="red") # bar색상주기
plt.savefig("test.png", dpi=800)
plt.title("총요금")
plt.show()

# plot
fig,ax = plt.subplots()
ax.scatter(tips['day'],tips['total_bill'])


# scatterplot 그리기
fig,ax = plt.subplots()
ax.scatter(tips['tip'],tips['total_bill'])

plt.scatter(tips['tip'],tips['total_bill'])
plt.xlabel('tip')
plt.ylabel('total_bill')

tips.plot(kind='scatter',x='tip',y='total_bill')
plt.title("팁금액")
plt.xlabel('x축 ')
plt.ylabel('y축 ')
# plt.rcParams['font.family'] ='Malgun Gothic' #한글깨짐 처리
plt.rcParams['font.family'] = 'Malgun Gothic'
sns.scatterplot(data=tips,x='tip',y='total_bill')
sns.scatterplot(data=tips,x='tip',y='total_bill',hue='sex')  # sex 범례
sns.scatterplot(data=tips,x='tip',y='sex',hue='total_bill')  # total_bill 범례
sns.scatterplot(data=tips,x='smoker',y='sex',hue='total_bill')# total_bill 범례
sns.scatterplot(data=tips,x='sex',y='tip',hue='smoker')      # smoker 범례


temp = tips[tips['smoker']=='Yes']
sns.scatterplot(data=temp,x='tip',y='total_bill',hue='sex',s=200)      # smoker 범례

sns.violinplot(data=tips,x='sex',y='tip')      
sns.boxplot(data=tips,x='sex',y='tip')     
sns.boxplot(data=tips,x='total_bill',y='tip') 
sns.scatterplot(data=tips,x='total_bill',y='tip') 


sns.boxplot(data=tips,x='day',y='tip',hue="smoker") 
sns.boxplot(data=tips,x='day',y='tip',hue="day") 
tips

# 1.몇 차원
# 2 각 차원의 데이터타입
# 3.그림종류

plt.show()


fig,ax = plt.subplots(2,2)
sns.scatterplot(data=tips,x='sex',y='tip',ax=ax[0][0])
sns.boxplot(data=tips,x='day',y='tip',hue='sex',ax=ax[0][1])
plt.show()


sns.kdeplot(data=tips,x='tip') #밀도 그래프
sns.rugplot(data=tips,x='tip') #밀도 그래프

sns.displot(data=tips,x='tip',kde=True)

sns.kdeplot(data=tips,x='tip')
plt.show()

# 막대 그래프 - 변수 1개
df = tips['smoker'].value_counts()
df.plot(kind='bar')
df.plot(kind='barh')

sns.countplot(data=tips,x='smoker')
sns.countplot(data=tips,y='smoker')
plt.show()


# 막대 그래프 변수 2개
df = tips[['smoker','total_bill']].groupby('smoker',as_index=False)[['total_bill']].mean()
sns.barplot(data=tips,x='smoker',y='total_bill')
plt.show()