from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote_plus

searchs_word = quote_plus(input("검색어를 입력하세요").strip())

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

    url = f"https://www.pressian.com/pages/search?sort=1&search={searchs_word}&page="+ str(i)
    titles = findHtml(url,'title')
    titles = titles[11:]

    for title in titles: 
        page = "https://www.pressian.com"+title.a["href"]
        sub_title = findHtml(page,'sub_title')
        
        body = findHtml(page,'article_body',0)
        print(body.text)
    print(main_title)



