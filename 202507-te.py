import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# 파일 읽어오기(201001~202005) : 파일명에서 년도 추출해서 index로 넣기
# 파일 읽어오기(202005~202103) : 파일sheet에서 년도 추출해서 index 넣기
# 파일 읽어서 dataframe 만들기

# folder_path = "C:/Users/WD/Data"

folder_path = "./sample/eData/"
xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
xls_files = glob.glob(os.path.join(folder_path, "*.xls"))

file_list = xlsx_files + xls_files

data_1 = []
data_2 = []
data_3 = []

for file in file_list:
    file_name = os.path.basename(file)
    
    if "immigration" in file_name:
        temp = pd.read_excel(file, header=1)
        
        year = file_name.split("_")[-1].replace(".xlsx","")
        temp["year"] = year

        data_1.append(temp)
#        temp.set_index("year", inplace=True)
    
    elif "세부통계" in file_name:
        temp = pd.read_excel(file, sheet_name=1)
        temp.rename(columns={"년월":"year"}, inplace=True)

        data_2.append(temp)        
#        temp.set_index("year", inplace=True)
        
    elif "목적별" in file_name:
        temp = pd.read_excel(file)   #!pip install xlrd==2.0.1
        temp = temp.iloc[1:]
        
        data_3.append(temp)
        
df_1 = pd.concat(data_1, ignore_index=True)  # 전체적인 추이를 볼 그래프 
df_2 = pd.concat(data_2, ignore_index=True)  # 2020년도 이후를 볼 그래프
df_3 = pd.concat(data_3, ignore_index=True)  # 2020년도 이후를 볼 그래프

df_1.info()
df_2.info()
df_3.info()


# nan값, 필요없는 값 지우기
df_1.isna().sum()
df_2.isna().sum()
df_3.isna().sum()

df_1 = df_1.query("국적 != ['계','전년동기','성장률(%)','구성비(%)']")
df_3 = df_3.query("국적 != '전 체'")


# df_1.dropna(inplace=True)

#df_1 column이름 바꾸기
df_1.rename(columns={"계":"총 인원"}, inplace =True)

#df_3 melt하기 ; # pd.melt(DataFrame, id_vars=기준 column(고정할 column), value_vars=변수(세로로 늘릴 column))
value_vars = [col for col in df_3.columns if "20" in str(col)]
df_3 = pd.melt(df_3, id_vars=["국적","목적"], value_vars=value_vars, var_name="year", value_name="인원")
df_3["인원"] = df_3["인원"].astype(int)




# year을 datetime형으로 바꾸기
df_1["year"] = pd.to_datetime(df_1["year"], format="%Y%m")
df_2["year"] = df_2["year"].astype(str)
df_2["year"] = pd.to_datetime(df_2["year"], format="%Y%m")
df_3["year"] = pd.to_datetime(df_3["year"], format="%Y년%m월")








# 2010년 자료 살펴보기
# 미주, 아시아주, 구주 데이터 보기

temp_data = df_1.query("국적 == ['미주','아시아주','구주']")
temp_data_2 = temp_data.groupby(["year", "국적"],as_index=False)[["계"]].sum()
sns.lineplot(temp_data_2, x="year", y="계", hue="국적")

# 아시아 국가 추이 보기
asia_data = df_1.query("국적 == ['일본','중국','대만','필리핀']")
temp_asia_data = asia_data.groupby(["year", "국적"])[["계"]].sum()
sns.lineplot(temp_asia_data, x="year", y="계", hue="국적")

# 2020년도는 왜 급락??  : df_1과 df_2가 차이가 커서 의미없음
asia_data_20 = df_1.query("year >= 2020.01 & 국적 == ['일본','중국','대만','필리핀']")
temp_asia_data = asia_data_20.groupby(["year","국적"])[["계"]].sum()
sns.lineplot(temp_asia_data, x="year", y="계", hue="국적")