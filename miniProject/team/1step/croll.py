import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}

url = 'https://datalab.naver.com/keyword/realtimeList.naver?where=main'
res = requests.get(url, headers = headers)
print(res)
soup = BeautifulSoup(res.content, 'html.parser')
print(soup)
data = soup.select('span.item_title')
print(data)
for i, item in enumerate(data):
    print(i+1,'위',':',item.get_text())