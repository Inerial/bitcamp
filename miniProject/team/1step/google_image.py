from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus
import os, shutil
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# while True:
for plusUrl in ['한지민','신세경','전지현','문채원','정은지','윤아','미나','지수','아린','쯔위','다현','한효주','이영애','최유정','세정','윤보미','사나','예린','박보영','김태희','한가인','아이린']:
    time.sleep(60)
    baseUrl = 'https://www.google.com/search?q='

    # plusUrl = input('검색어를 입력하세요 (입력 안하고 엔터시 종료) : ')
    if plusUrl is '':
        break;
    # 한글 검색 자동 변환
    url = baseUrl + quote_plus(plusUrl) + '&tbm=isch'

    driver = webdriver.Chrome("D:\\Study\\miniProject\\team\\1step\\chromedriver.exe")
    driver.get(url)

    body = driver.find_element_by_tag_name("body")
    num_of = 100

    while num_of:
        num_of -= 1
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)
        try:
            driver.find_element_by_class_name('mye4qd').click()
        except:
            None

    # <input jsaction="Pmjnye" class="mye4qd" type="button" value="결과 더보기">

    html = driver.page_source

    soup = bs(html, "html.parser")
    # soup = bs(html, "lxml")
    # soup = bs(html, "html5lib")
    img = soup.find_all(class_='rg_i Q4LuWd', limit=100)


    n = 1
    download_path = os.path.dirname(os.path.realpath(__file__))+ '/img/' + plusUrl
    print(download_path)
    for i in img:
        print(i)
        try:
            imgUrl = i['data-src']
        except:
            imgUrl = i['src']
        with urlopen(imgUrl) as f:
            if not os.path.isdir(download_path):
                os.mkdir(download_path)
            with open(download_path + '/' + plusUrl + str(n)+'.jpg','wb') as h: # w - write b - binary
                img = f.read()
                h.write(img)
        n += 1
    print('다운로드 완료')
