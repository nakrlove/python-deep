import pandas as pd
import glob

# 폴더 안에 있는 모든 CSV 파일 찾기
csv_files = glob.glob("C:\\Users\\Admin\\study01\\local_test\\서울시_공동주택_아파트_정보11.csv")

for csv_file in csv_files:
    # 확장자만 xlsx로 바꾸기
    xlsx_file = csv_file.replace(".csv", ".xlsx")
    
    # 변환
    df = pd.read_csv(csv_file, encoding="utf-8")
    df.to_excel(xlsx_file, index=False, engine="openpyxl")
    
    print(f"✅ {csv_file} → {xlsx_file} 변환 완료")