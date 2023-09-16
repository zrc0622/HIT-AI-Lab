# 使用爬虫从网络上爬取Html源代码，解析并抽取指定信息，并封装为Json格式

import requests
from bs4 import BeautifulSoup
import json
import bs4

response = requests.get('http://www.hit.edu.cn') # 从网页中获取内容

web_dict = {} # 初始化dick
json_dir = "title_link.json"

# 根据html网页字符串创建BeautifulSoup对象
soup = BeautifulSoup(response.content,'html.parser') # 将获取到的网页内容转为BeautifulSoup对象（BeautifulSoup对象是一种特殊的数据结构，它含有很多属性）

#从文档中找到所有标签的链接
for link in soup.find_all('a'): # 找到标签'a'，并依次循环
    if link.get('href') == '': # 判断链接是否为空
        continue
    if type(link.next) is bs4.element.NavigableString: # title存在于link中的title字段或link的next属性三
        web_dict[str(link.next)]=link.get('href')
    else:
        web_dict[link.get('title')]=link.get('href')

with open(json_dir, 'w', encoding='utf-8') as json_file:
    json.dump(web_dict, json_file, ensure_ascii=False)
