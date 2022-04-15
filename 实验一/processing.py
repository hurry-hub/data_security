import requests
import logging as lg
import time
from bs4 import BeautifulSoup
import openpyxl
import numpy

lg.basicConfig(level=lg.DEBUG, format=" %(message)s ")


def top_baidu():
    url = 'http://top.baidu.com/buzz?b=7&fr=topindex'
    html = requests.get(url)
    html.encoding = html.apparent_encoding
    bs = BeautifulSoup(html.text, 'html.parser')
    top_list = bs.select(".list-title")
    for i in range(len(top_list)):
        print("%d : %s" % ((i + 1), top_list[i].text))


def write_data_to_excel(list_num, list_book_name, list_publisher, list_recommend):
    # write the data in the file
    wb_export = openpyxl.Workbook()
    sheet = wb_export['Sheet']
    for i in range(len(list_num)):
        sheet.append([list_num[i].text, list_book_name[i].text, list_publisher[i * 2].text, list_recommend[i].text])
    wb_export.save('dangdangbook.xlsx')


def book_list():
    list_num = []
    list_book_name = []
    list_publisher = []
    list_recommend = []
    request_header = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Cookie": "BIGipServerPools_Web_ssl=1514973376.47873.0000; Hm_lvt_c01558ab05fd344e898880e9fc1b65c4=1616163138; qimo_seosource_578c8dc0-6fab-11e8-ab7a-fda8d0606763=%E7%BB%94%E6%AC%8F%E5%94%B4; qimo_seokeywords_578c8dc0-6fab-11e8-ab7a-fda8d0606763=; accessId=578c8dc0-6fab-11e8-ab7a-fda8d0606763; pageViewNum=8; Hm_lpvt_c01558ab05fd344e898880e9fc1b65c4=1616163639",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
    }
    for i in range(50):
        url = 'http://bang.dangdang.com/books/bestsellers/01.54.00.00.00.00-24hours-0-0-1-' + str(i + 1)
        lg.debug(url)

        html = requests.get(url, headers=request_header)
        try:
            html.raise_for_status()

        except:
            print("Cannot get the html")
        else:
            html.encoding = 'gb2312'
            bs = BeautifulSoup(html.text, 'html.parser')
            # top_list_ul = bs.select(".bang_list.clearfix.bang_list_mode")
            # print("there are %d ul" % len(top_list_ul))
            # for child in top_list_ul[0].children:
            #	print(child)
            # list_num = bs.select(".bang_list.clearfix.bang_list_mode .list_num.red")
            list_num.extend(bs.select(".bang_list.clearfix.bang_list_mode .list_num"))
            list_publisher.extend(bs.select(".bang_list.clearfix.bang_list_mode .publisher_info"))
            list_book_name.extend(bs.select(".bang_list.clearfix.bang_list_mode .name"))
            list_recommend.extend(bs.select(".bang_list.clearfix.bang_list_mode .tuijian"))
        time.sleep(3)
    write_data_to_excel(list_num, list_book_name, list_publisher, list_recommend)
    lg.debug("there are %d book" % len(list_book_name))


# book_list()
top_baidu()
