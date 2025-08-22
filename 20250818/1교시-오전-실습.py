from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd

# links = []
main_title = []
for i in range(1,3):

    url = "https://www.pressian.com/pages/search?sort=1&search=%ED%8A%B8%EB%9F%BC%ED%94%84&page="+ str(i)
    html = urlopen(url).read()


    soup = BeautifulSoup(html,'html.parser')
    titles = soup.find_all( class_ ='title')
    titles = titles[11:]

    for title in titles: 
        page = "https://www.pressian.com"+title.a["href"]
        # links.append("https://www.pressian.com"+title.a["href"])
        
        print(page)
        html = urlopen(page).read()
        soup = BeautifulSoup(html,'html.parser')
        title = soup.find_all( class_ ='title')
        main_title.append(title[11].text)
        
    print(main_title)