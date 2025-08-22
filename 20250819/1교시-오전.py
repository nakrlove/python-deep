from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote_plus

import requests




def findHtml(page,ftage,findsFlag = 1):
    html = urlopen(page).read()
    soup = BeautifulSoup(html,'html.parser')
    
    if findsFlag == 1 :
     tag = soup.find_all( class_= ftage )
    else  :
     tag = soup.find( class_= ftage )
     
    return tag





main_title = []
for i in range(1,3):

    url = f"https://www.pressian.com/pages/search?sort=1&search={quote_plus('트럼프')}&page=1"
    titles = findHtml(url,'title')
    titles = titles[11:]

    for title in titles: 
        page = "https://www.pressian.com"+title.a["href"]
        sub_title = findHtml(page,'sub_title')
        
        body = findHtml(page,'article_body',0)
        print(body.text)
    print(main_title)
    




# url = f"https://www.pressian.com/pages/search?sort=1&search={quote_plus('트럼프')}&page=1"
url = f"https://search.daum.net/search?w=news&nil_search=btn&DA=PGD&enc=utf8&cluster=y&cluster_page=1&q=%ED%8A%B8%EB%9F%BC%ED%94%84&p=1"

headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"}
# html = urlopen(url).read()
html = requests.get(url ,headers=headers).text
soup = BeautifulSoup(html,'html.parser')
titles = soup.find_all( class_= 'tit-g clamp-g')

len(titles)
titles[9].text
titles[9].string
titles[9].a['href']
titles[9].a['onclick']
titles[9].a.attrs['href']
titles[9].a.attrs['onclick']


li1 = []
link = []
for title in titles:
   li1.append(title.text)
for title in titles:   
   link.append(title.a['href'])
   
df = pd.DataFrame([li1 ,link])
df = df.T
df.columns = ['제목','링크']
df


###2교시 이미지 검색후 이미지 저장하기
# url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=image&ssc=tab.image.all&query=%EC%A0%95%EC%B1%84%EC%97%B0&oquery=apple&tqi=j6nhuwqo15VssNM6luZssssstCl-224697&ackey=ymq6qnal"
# headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"}
# html = requests.get(url, headers=headers).text
# soup = BeautifulSoup(html, 'html.parser')
# images = soup.find_all(class_ ='thumb')
# images


from urllib.request import urlretrieve

url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=bts'

html = urlopen(url).read()  # read() 가 있음에 주의. urlopen()은 그야말로 열기만 함.
soup = BeautifulSoup(html, 'html.parser')
images = soup.find_all(class_ = 'thumb') 

for i,image in enumerate(images):
    print(image.img['src'])
    # img = urlopen(image.img['src']).read()
    # with open(f"{i}.png","wb") as f:
    #    f.write(img)
    urlretrieve(image.img['src'],f"{i}.png")  #이미지를 다운로드해서 저장함
