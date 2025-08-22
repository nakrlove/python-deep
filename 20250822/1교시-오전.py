from selenium import webdriver
from selenium.webdriver.chrome.service import Service 
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import pandas as pd
"""
1.제목 ,소 제목 ,링크를 크롤링
  주소:패턴
  접근: urllib , requests
  soup 객체 = BeautifulSoup로 parsing
  soup.find_all(class_ = "")
  soup.a['href']
  pd.DataFrame
  
  2.User-Agent정보
  3.이미지 저장하기
  urlretrieve(url,"파일이름")
  4.접근이 안되는 경우 (naver 이미지. 클리스 이름이 없다)
  Selenium 사용 
  - 실제 브라우저 .(naver이미지. 클래스 이름이 없다)
  - 로그인(한빛미디어)
"""


######################################
# 자동로그인 처리
######################################
url = "https://www.hanbit.co.kr/login?redirect=https%3A%2F%2Fwww.hanbit.co.kr%2Findex.html"

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
driver.get(url)

ID = 'royalstatesi'
PW = 'sori2009!'
e = driver.find_element(By.ID,"id")
e.clear()
e.send_keys(ID)

e = driver.find_element(By.ID,"password")
e.clear()
e.send_keys(PW)

# submit요청
e.send_keys(Keys.ENTER)



url = "https://www.hanbit.co.kr/myhanbit/membership.html"
# driver.get(url)
grade =driver.find_element(By.CLASS_NAME,'txt1')
grade.text

# grade =driver.find_element(By.CSS_SELECTOR,'#container > div.myhanbit_wrap > div.my_info > div.my_rating > div.txt_area > p')
grade =driver.find_element(By.CSS_SELECTOR,'#container > div.myhanbit_wrap > div.my_info > div.my_rating')
grade.text

grade =driver.find_element(By.XPATH,'//*[@id="container"]/div[2]/div[1]/div[1]/div[2]/p')
grade.text



#표가져 오기
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote_plus

url = "https://finance.naver.com/marketindex/"
html = urlopen(url).read()

soup = BeautifulSoup(html,'html.parser')
soup.find(class_ = 'value').text
exchange = soup.find(class_ = 'head_info point_dn')
print(exchange.text)

"""
soup.find_all() -> [,,]
soup.find() -> 하나
soup.select -> [,,]
soup.select_one -> 하나

참조
https://blog.naver.com/kiddwannabe/221177292446
"""

soup = BeautifulSoup(html,'html.parser')
# soup.select('head_info point_dn').text
soup.select('.head_info.point_dn')[0].text
soup.select_one('.head_info.point_dn').text

#멍때림
soup.select_one(".h_lst")

table = soup.select_one(".tbl_exchange")
rows = table.select('tr')
len(rows)
data = []
for row in rows:
    cols = [col.get_text(strip=True) for col in row.select('td')]
    if cols:  # 빈 행 제외
        data.append(cols)
data

pd.DataFrame(data,columns=["A","B"])


url = "https://finance.naver.com/marketindex/exchangeList.naver"
html = urlopen(url).read()

soup = BeautifulSoup(html,'html.parser')

table = soup.select_one(".tbl_exchange")
rows = table.select('tr')
len(rows)
data = []
for row in rows:
    cols = [col.get_text(strip=True) for col in row.select('td')]
    if cols:  # 빈 행 제외
        data.append(cols)
data

pd.DataFrame(data,columns=['통화명','매매기준율','현찰','송금','미화환산율','사실 때 ','파실 때'])


dd = pd.read_clipboard()
df = pd.read_html(url,encoding='euc-kr')[0]
df[:3]