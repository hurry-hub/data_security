#!/usr/bin/env python
# coding=utf-8

"""
@author: Richard Huang
@license: WHU
@contact: 2539444133@qq.com
@file: check.py
@date: 22/03/18 19:54
@desc: 
"""
import csv
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
import requests
import json
import time
import pandas as pd

# 防止https证书校验不正确
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

API_KEY = 'dtPcGb1jfXuxHso4LOyleaDV'
SECRET_KEY = '8HGEFt12mjSk0GlIDYojHA73uQ8XoF5Z'
TEXT_CENSOR = "https://aip.baidubce.com/rest/2.0/solution/v1/text_censor/v2/user_defined"

"""  TOKEN start """
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'

def fetchURL(url):
    """
    功能：访问 url 的网页，获取网页内容并返回
    参数：
        url ：目标网页的 url
    返回：目标网页的 html 内容
    """
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    }

    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        print(r.url)
        return r.text
    except requests.HTTPError as e:
        print(e)
        print("HTTPError")
    except requests.RequestException as e:
        print(e)
    except:
        print("Unknown Error !")

def dec(x):
    # bv，av互换算法
    table = 'fZodR9XQDSUm21yCkr6zBqiveYah8bt4xsWpHnJE7jL5VG3guMTKNPAwcF'
    tr = {}
    for i in range(58):
        tr[table[i]] = i
    s = [11, 10, 3, 8, 4, 6]
    xor = 177451812
    add = 8728348608
    r = 0
    for i in range(6):
        r += tr[x[s[i]]] * 58 ** i
    return (r - add) ^ xor


def parserHtml(html):
    """
    功能：根据参数 html 给定的内存型 HTML 文件，尝试解析其结构，获取所需内容
    参数：
            html：类似文件的内存 HTML 文本对象
    """
    try:
        sentence = json.loads(html)
    except:
        print('error')
    commentlist = []

    for i in range(10):
        comment = sentence['data']['replies'][i]
        words = comment['content']['message']
        commentlist.append(words)

    writePage(commentlist)
    print('---' * 30)

def writePage(urating):
    """
       将数据写入文件
    """
    dataframe = pd.DataFrame(urating)
    dataframe.to_csv('Bilibili-500条.csv', encoding="utf_8_sig", mode='a', index=False, sep=',', header=False)

def fetch_token():
    """
         获取token
    """
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        print(err)
    result_str = result_str.decode()
    result = json.loads(result_str)

    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not 'brain_all_scope' in result['scope'].split(' '):
            print('please ensure has check the  ability')
            exit()
        return result['access_token']
    else:
        print('please overwrite the correct API_KEY and SECRET_KEY')
        exit()


def read_file(file_path):
    """
        读取文件
    """
    f = None
    content = []
    try:
        f = open(file_path, 'r', encoding="utf-8")
        data_origin = csv.reader(f)
        for item in data_origin:
            content.append(item[0])
        return content
    except:
        print('read image file fail')
        return None
    finally:
        if f:
            f.close()

def request(url, data):
    """
        调用远程服务
    """
    req = Request(url, data.encode('utf-8'))
    has_error = False
    try:
        f = urlopen(req)
        result_str = f.read()
        result_str = result_str.decode()
        return result_str
    except  URLError as err:
        print(err)


if __name__ == '__main__':
    bv = 'BV1hD4y1X7Rm'
    oid = dec(bv)
    for page in range(0, 50):
        url = 'https://api.bilibili.com/x/v2/reply?type=1&sort=2&oid=' + str(oid) + '&pn=' + str(page)
        html = fetchURL(url)
        parserHtml(html)

        # 为了降低被封ip的风险，每爬20页便歇5秒。
        if page % 20 == 0:
            time.sleep(5)

    # 获取access token
    token = fetch_token()
    # 拼接文本审核url
    text_url = TEXT_CENSOR + "?access_token=" + token

    data = read_file('Bilibili-500条.csv')
    for i in range(len(data)):
        content = data[i]
        result = request(text_url, urlencode({'text': content}))
        print(result)