import requests
import pandas as pd


#https://datadoctorblog.com/2025/03/17/Py-Crawling-API-gov-APT-trade/
def fetch_real_estate_data(region_code, start_date, end_date):
    url = f"http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RealEstateTradingService/getRTMSDataSvcAptTrade?"
    # url = f"https://datadoctorblog.com/2025/03/17/Py-Crawling-API-gov-APT-trade/"
    params = {
        "serviceKey": "%2FJKpxO3%2FyniFyL4oapSPtGy3gVG%2FKIYP2HPGLbRU5%2Fwa3418ooN5kIP%2Fd8jIUu4rorzv4iJe6KIddKqRt2iomA%3D%3D",  # 발급받은 API 키
        "LAWD_CD": region_code,        # 지역 코드
        "DEAL_YMD": start_date,        # 거래 년월 (YYYYMM)
        "startPage": 1,                # 시작 페이지
        "numOfRows": 1000              # 한 페이지당 데이터 수
    }
    response = requests.get(url, params=params)
    data = response.json()
    # 
    # http://openapi.seoul.go.kr:8088/737a527771726f7938355477424263/xml/tbLnOpendataRtmsV/1/5/
    # 데이터 가공
    items = data['response']['body']['items']['item']
    df = pd.DataFrame(items)
    return df

# 예시: 서울 강남구의 2023년 1월 거래 데이터 수집
df = fetch_real_estate_data("11680", "202301", "202301")
print(df.head())