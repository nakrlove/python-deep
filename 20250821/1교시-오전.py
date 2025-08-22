
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote_plus


from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service 
from webdriver_manager.chrome import ChromeDriverManager
from urllib.request import urlretrieve
import time


class Crawl:
    
    def __init__(self,search_word):
        self.search_word = search_word
    
    def crawl(self):
        import os
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service)
        #url = f'https://search.naver.com/search.naver?where=image&sm=tab_jum&query={quote_plus("뉴진스")}'
        url = f'https://search.naver.com/search.naver?where=image&sm=tab_jum&query={quote_plus(self.search_word)}'
        driver.get(url)
        time.sleep(1)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        images = soup.find_all(class_ = '_fe_image_tab_content_thumbnail_image')


        os.mkdir(f"{self.search_word}")
        for i,image in enumerate(images):
            urlretrieve(image['src'],f"./{self.search_word}/{i}.png")  #이미지를 다운로드해서 저장함

        driver.quit()



# word = input('검색어를 입력해주세요!')
cl = Crawl('한옥')
cl.crawl()

