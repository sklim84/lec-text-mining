from bs4 import BeautifulSoup
import requests
import re
from selenium import webdriver
import pandas as pd
import time
import numpy as np

# setting
wd = webdriver.Chrome('..\..\selenium_server\chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)

# pip install xlrd
df_news = pd.read_excel('./news.xlsx')

# full text column 확인 및 추가
if 'full text' not in df_news.columns:
    print('##### there is no \'full text\' column')
    df_news["full text"] = None
print(df_news.info())

for index, url in enumerate(df_news['DocumentURL']):

    full_text = df_news.at[index, 'full text']
    # 이미 크롤링된 경우는 생략(nan check)
    if type(full_text) == str:
        print('##### index {} is not full text empty'.format(index))
        continue

    try:
        wd.get(url)
        time.sleep(2)
    except:
        # request 오류 발생 시 생략(추후 재시도 시 확인)
        print('##### request error - index: {}, url: {}'.format(index, url))
        continue

    try:
        html = wd.page_source
        soup = BeautifulSoup(html, "html.parser")

        # Full Text
        div_fulltextzone = soup.find('div', {'id': 'fullTextZone'})
        full_text = div_fulltextzone.get_text()
        if len(full_text) == 0:
            div_readableContent = soup.find('div', {'id': 'readableContent'})
            full_text = div_readableContent.get_text()
        print(index, full_text)
        if len(full_text) == 0:
            print(soup)
        df_news.at[index, 'full text'] = str(full_text)
    except:
        # 파싱 오류 발생 시 생략(추후 재시도하면서 포맷 재확인)
        print(html)
        print('##### no full text - index: {}, url: {}'.format(index, url))
        time.sleep(30) # 캡차인 경우를 위해

    if index % 100 == 0:
        # save
        print('##### save data: ~{}'.format(index))
        df_news.to_excel('./news.xlsx', index=False, encoding='utf-8-sig')

# save
df_news.to_excel('./news.xlsx', index=False, encoding='utf-8-sig')

# import pandas as pd
#
# df_news = pd.read_excel('./news.xlsx', index_col=None, header=0)
# print(df_news[df_news['full text'].isna()])
