
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote_plus

import requests

def search(search_word):

    link = []
    articles = []
    for i in range(1,3):

        url = f"https://www.pressian.com/pages/search?sort=1&search={quote_plus(search_word)}&page="+str(i)
        
        html = urlopen(url).read()
        soup = BeautifulSoup(html,'html.parser')
        
        
        titles = soup.find_all( class_= 'body' )

        for title in titles:
            sub_url = "https://www.pressian.com"+title.a["href"]
            sub_html = urlopen(sub_url).read()
            soup = BeautifulSoup(sub_html,'html.parser')
            body = soup.find( class_= 'article_body' )
            articles.append(body.text)

    return  pd.DataFrame(articles)


trump = search('트럼프')
trump.iloc[0,0]



from urllib.request import urlretrieve

url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=bts'

html = urlopen(url).read()  # read() 가 있음에 주의. urlopen()은 그야말로 열기만 함.
soup = BeautifulSoup(html, 'html.parser')


with open("sample.txt","w") as f:
    f.write(str(soup,encoding='utf-8'))


images = soup.find_all(class_ = 'thumb') 

for i,image in enumerate(images):
    print(image.img['src'])
    # img = urlopen(image.img['src']).read()
    # with open(f"{i}.png","wb") as f:
    #    f.write(img)
    urlretrieve(image.img['src'],f"{i}.png")  #이미지를 다운로드해서 저장함
    

import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# 크롬드라이버 경로 지정
# chrome_service = Service("C:\\Users\\Admin\\temp\\chromedriver.exe")

service = Service(ChromeDriverManager().install())
# 웹드라이버 실행
driver = webdriver.Chrome(service=service)
# 페이지 열기 (예시)
# driver.get("https://www.google.com")
# driver.get("https://www.naver.com")
# driver.get("https://www.daum.net")

driver.get(url)
time.sleep(1)
#종료
driver.quit()
print(driver.title)


html = driver.page_source

soup = BeautifulSoup(html,'html.parser')
# images = soup.find_all(class_ = 'thumb') 
images = soup.find_all(class_ = '_fe_image_tab_content_thumbnail_image') 
images[0]['src']



# count = 0
# for image in images:
#     data = urlopen(image['src']).read()
#     with open(f"./bts/{count}.jpg", 'wb') as h:        
#         h.write(data)
#         count += 1
        


from selenium import webdriver
from selenium.webdriver.chrome.service import Service 
from webdriver_manager.chrome import ChromeDriverManager

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=apple'
driver.get(url)
time.sleep(1)
html = driver.page_source
driver.quit()

soups = BeautifulSoup(html, "html.parser")
images = soups.find_all(class_ = '_fe_image_tab_content_thumbnail_image')

count = 1
for image in images:
    urlretrieve(image['src'], f"{count}.jpg")
    count += 1