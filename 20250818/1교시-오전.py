from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd


li1 = []
li2 = []
links = []
for i in range(1,2):
    url = "https://www.pressian.com/pages/search?sort=1&search=%ED%8A%B8%EB%9F%BC%ED%94%84&page="+ str(i)
    print(url)
    resp = urlopen(url).read()


    soup = BeautifulSoup(resp,'html.parser')
    titles = soup.find_all( class_ ='title')
    titles = titles[11:]
    
    for title in titles:
        li1.append(title.text)

    bodys = soup.find_all( class_ ='body')
    # bodys = bodys[11:]
    for body in bodys:
        li2.append(body.text)
  
    
    for title in titles: 
        links.append("https://www.pressian.com"+title.a["href"])

    print(f"{i}페이지 완료")

df = pd.DataFrame({'제목':li1,'링크':links,'내용':li2})
df.to_csv("trump.csv", encoding='utf-8-sig', index=False)


df