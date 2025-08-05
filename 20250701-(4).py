import pandas as pd
import numpy as np
import seaborn as sns
# %%
import matplotlib.pyplot as plt
import pyreadstat

# 데이터 불러오기
df = pd.read_spss('Koweps_hpwc14_2019_beta2.sav')

# 복사본 만들기 
welfare = df.copy()
welfare = welfare.rename(columns = {'h14_g3'     : 'sex',            #  성별
                                    'h14_g4'     : 'birth',          #  태어난 연도
                                    'h14_g10'    : 'marriage_type',  #  혼인 상태
                                    'h14_g11'    : 'religion',       #  종교 
                                    'p1402_8aq1' : 'income',         #  월급 
                                    'h14_eco9'   : 'code_job',       #  직업 코드
                                    'h14_reg7'   : 'code_region'})   #  지역 코드

# temp = welfare[['h14_g3','h14_g4' ,'h14_g10' ]]
# temp

welfare = welfare[['sex','birth','marriage_type','religion','income','code_job','code_region']]
welfare.rename({"sex":'gender'},axis=1,inplace=True)





# 데이터셋 살펴보기
# - 행과 열의 개수
# - 각 열의 의미
# - 각 열의 데이터 타입
# - 각 열의 기술통계 값
# - 결측치가 있는지,  몇개인지 처리방법
# - 이상치유무 처리방법


welfare['gender'].dtypes  # 변수 타입 출력

welfare.info()
welfare[:3]
welfare['gender'].value_counts()
welfare['gender'] = np.where(welfare['gender'] == 9, np.nan, welfare['gender'])
df['gender'] = df['gender'].replace('1.0','male')


welfare['gender'] = np.where(welfare['gender'] < 1.1,'male','female')

welfare['gender'] = welfare['gender'].map({1.0: 'male', 2.0: 'female'})
welfare['gender'] = welfare['gender'].replace({1.0: 'male', 2.0: 'female'})
pd.cut(welfare['gender'],bins=[0,1.1,2.1],labels=['male','female'])

welfare.isna().sum()
welfare['gender'].value_counts()



df = pd.DataFrame(welfare['gender'].value_counts())
df
df.plot(kind='bar',y='count')


sns.countplot(data=welfare,x='gender',hue='gender')

sns.barplot(data=df, x = 'gender', y='count',hue='gender')
plt.show()

welfare.describe()
welfare['income'].describe()
sns.boxplot(data=welfare,y='income')

sns.histplot(data=welfare,x='income',bins=20)
plt.show()

sum(welfare['income'] == 9999)

sex_income = welfare.dropna(subset = 'income').groupby('gender', as_index = False).agg(mean_income = ('income', 'mean'))
sex_income

# 성별로 소득의 평균을 구해 봅시다.
income_mean = welfare.groupby('gender',as_index=False)[['income']].agg('mean')

# barplot을 그려주세요
sns.barplot(data=income_mean,x='gender',y='income',hue='gender')
plt.show()

# 소득이 0 인 사람의 수는
sum(welfare['income']==0)

# 나이별(세대별) 평균소득
# 1. age 칼럼을 만들어 주자
# 2. 20세 이하는 제외
# 3. age 칼럼을 정수로 바꾼다.
# 4. (1년단위)나이별  월급의 평균
# 5.  age < 20 : 'youth' , age < 40 : young , age < 60: middle , age > 60 : old

2019 - welfare['birth']

# 1
welfare['age'] = 2019 - welfare['birth'] 
imcom_age = welfare.groupby(['age','gender'],as_index=False)[['income']].mean()
imcom_age.info()
imcom_age

# 2
welfare = welfare.query('age > 20')
# 3
imcom_age['age'].astype('int') #타입변경 
welfare['age'] = welfare['age'].astype('int') #타입변경 


# 4

sns.lineplot(data=imcom_age,x='age',y='income',hue='gender')
plt.show()


# 5
bins = [0, 20, 40, 60, 200, welfare['age'].max() + 1]  # 최대 나이 + 1을 upper bound로
labels = ['youth', 'young', 'middle', 'old']

# cut 함수로 범주형 변수 생성
welfare['generation'] = pd.cut(welfare['age'], bins=bins, labels=labels, right=False)

welfare
# 세대별로 소득의 평균값을 구해보자
welfare_gen = welfare.groupby(['generation','age'] ,as_index=False)[['income']].agg('mean')
sns.barplot(data=welfare_gen,x='generation',y='income')
plt.show()

welfare_gen
# %%
# 세대별,성별로 barplot을 그려주세요
sns.barplot(data=welfare_gen,x='generation')


# 소득 상위 10명만 뽑기 : 소득으로 정렬한다. -> 위에서 10개만 선택한다.

temp = welfare.sort_values('income',ascending=False)[:10]
temp[['birth','income']]


# %%
