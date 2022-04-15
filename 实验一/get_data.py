import requests
import json
import time
import pandas as pd


def fetchURL(url):
    '''
    功能：访问 url 的网页，获取网页内容并返回
    参数：
        url ：目标网页的 url
    返回：目标网页的 html 内容
    '''
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
    '''
    功能：根据参数 html 给定的内存型 HTML 文件，尝试解析其结构，获取所需内容
    参数：
            html：类似文件的内存 HTML 文本对象
    '''
    try:
        s = json.loads(html)
    except:
        print('error')

    commentlist = []
    hlist = []

    hlist.append("序号")
    hlist.append("名字")
    hlist.append("性别")
    hlist.append("时间")
    hlist.append("评论")
    hlist.append("点赞数")
    hlist.append("回复数")

    # commentlist.append(hlist)

    # 楼层，用户名，性别，时间，评价，点赞数，回复数
    for i in range(10):
        comment = s['data']['replies'][i]
        blist = []

        username = comment['member']['uname']
        sex = comment['member']['sex']
        ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(comment['ctime']))
        content = comment['content']['message']
        likes = comment['like']
        rcounts = comment['rcount']

        blist.append(username)
        blist.append(sex)
        blist.append(ctime)
        blist.append(content)
        blist.append(likes)
        blist.append(rcounts)

        commentlist.append(blist)

    writePage(commentlist)
    print('---' * 20)


def writePage(urating):
    '''
        Function : To write the content of html into a local file
        html : The response content
        filename : the local filename to be used stored the response
    '''
    dataframe = pd.DataFrame(urating)
    dataframe.to_csv('Bilibili_comment5-1000条.csv', encoding="utf_8_sig", mode='a', index=False, sep=',', header=False)


if __name__ == '__main__':
    bv = 'BV1hD4y1X7Rm'
    oid = dec(bv)
    for page in range(0, 10):
        url = 'https://api.bilibili.com/x/v2/reply?type=1&sort=2&oid=' + str(oid) + '&pn=' + str(page)
        html = fetchURL(url)
        parserHtml(html)

        # 为了降低被封ip的风险，每爬20页便歇5秒。
        if page % 20 == 0:
            time.sleep(5)
