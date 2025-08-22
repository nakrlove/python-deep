from selenium import webdriver
from selenium.webdriver.chrome.service import Service 
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


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