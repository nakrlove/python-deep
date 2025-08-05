import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# 파일 읽어오기(201001~202005) : 파일명에서 년도 추출해서 index로 넣기
# 파일 읽어오기(202005~202103) : 파일sheet에서 년도 추출해서 index 넣기
# 파일 읽어서 dataframe 만들기

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 글씨 출력

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
    
    elif "세부통계" in file_name:
        temp = pd.read_excel(file, sheet_name=1)
        temp.rename(columns={"년월":"year"}, inplace=True)

        data_2.append(temp)        
        
    elif "목적별" in file_name:
        temp = pd.read_excel(file)  #!pip install xlrd
        temp = temp.iloc[1:]
        
        data_3.append(temp)
        
df_1 = pd.concat(data_1, ignore_index=True)
df_2 = pd.concat(data_2, ignore_index=True)
df_3 = pd.concat(data_3, ignore_index=True)

df_1.info()
df_2.info()
df_3.info()
# %%

# 데이터프레임 깔끔하게 만들기

# nan값, 필요없는 값 지우기
df_1.isna().sum()
df_2.isna().sum()
df_3.isna().sum()

#df_1 column이름 바꾸기
df_1 = df_1.rename(columns={"계":"총 인원"})

#불필요한 띄어쓰기 삭제
df_2["국적"] = df_2["국적"].str.replace(" ", "")
df_2["지역"] = df_2["지역"].str.replace(" ", "")
df_3["국적"] = df_3["국적"].str.replace(" ", "")
df_3 = df_3.query("목적 != ['소 계']")
df_3["목적"] = df_3["목적"].replace({"유학연수":"유학/연수"})

#불필요한 data 삭제
df_1 = df_1.query("국적 != ['계','전년동기','성장률(%)','구성비(%)']")
df_3 = df_3.query("국적 != '전체'")

# df_1 melt 하기
df_1 = pd.melt(df_1, id_vars=["year","국적"], value_vars=["관광","상용","공용","유학/연수","기타"], var_name="목적", value_name="인원")

#df_3 melt하기 ; # pd.melt(DataFrame, id_vars=기준 column(고정할 column), value_vars=변수(세로로 늘릴 column))
value_vars = [column for column in df_3.columns if "20" in str(column)]
df_3 = pd.melt(df_3, id_vars=["국적","목적"], value_vars=value_vars, var_name="year", value_name="인원")
df_3["인원"] = df_3["인원"].astype(int)

# year을 datetime형으로 바꾸기
df_1["year"] = pd.to_datetime(df_1["year"], format="%Y%m")
df_2["year"] = df_2["year"].astype(str)
df_2["year"] = pd.to_datetime(df_2["year"], format="%Y%m")
df_3["year"] = pd.to_datetime(df_3["year"], format="%Y년%m월")

# %%

#df_1,2,3 합치기 : df_1과 df_2 data 중에서 2020.05가 중복
df_1.query("국적 =='일본' & 목적 =='관광' & year.dt.year == 2020 & year.dt.month == 5")["인원"].sum()
df_2.query("국적 =='일본' & 목적 =='관광' & year.dt.year == 2020 & year.dt.month == 5")["인원"].sum()

#df_1에서 2020.05년도 data를 삭제
df_1 = df_1.query("year.dt.year != 2020 | year.dt.month != 5")

#df_2에서 공통 data만 뽑기
temp_df_2 = df_2[["year","국적","목적","인원"]]

#concat하기
df_123 = pd.concat([df_1, temp_df_2, df_3])


# 문제..
# df_2에는 지역(아시아주, 미주, 구주 등등), df_1, df_3에는 지역이 국적에 포함되어 있다
df_1["국적"].unique()
df_3["국적"].unique()
df_2["국적"].unique()
df_2["지역"].unique()
# %%

# 엑셀파일로 만들기
df_123.to_excel("TEST2.xlsx")
# %%




# -----------test
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

# %%

# df_1과 df_2 데이터프레임 합치기 
# temp_df_1 = df_1.set_index("year")
# temp_df_2 = df_2.set_index("year")
temp_df_1 = df_1.reset_index()
temp_df_2 = df_2.reset_index()

asia_data_20_2 = df_2.query("year >= 2020.01 & 국적 == ['일  본','중  국','대  만','필리핀']")
temp1 = asia_data_20.groupby(["year","국적"],as_index=False)["계"].sum()
temp2 = asia_data_20_2.groupby(["year","국적"] ,as_index=False)["인원"].sum()

temp1.rename(columns={"계":"인원"}, inplace=True)

a = pd.concat([temp1, temp2], axis=0)

sns.lineplot(a, x="year", y="인원", hue="국적")

temp2 = temp2.set_index("year")
temp1.info()
con_temp = pd.concat([temp1, temp2], axis=0)

sns.lineplot(con_temp, x="year", y="인원", hue="국적")

temp_data = con_temp.query("year >= 2020.01 & 국적 == ['일본','중국','대만','필리핀']")
con_temp_data = temp_data.groupby(["year","국적"])[["계"]].sum()


# %%



temp = df_123.query("국적 == ['아시아주','미주','구주']")

test = temp.groupby(["year","국적","목적"], as_index=False)["인원"].sum()
sns.lineplot(test, x="year", y="인원", hue="국적", ci=None)


test = temp.groupby(["year","국적","목적"], as_index=False)["인원"].sum()
sns.lineplot(test, x="year", y="인원", hue="목적", ci=None)

test = df_123.query("국적=='중국'").groupby(["year","목적"] ,as_index=False)["인원"].sum()
sns.lineplot(test, x="year", y="인원", hue="목적")
