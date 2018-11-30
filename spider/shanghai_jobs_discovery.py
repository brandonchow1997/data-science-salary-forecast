# common imports
import requests
from lxml import etree
import time
import random
import pymongo
from retrying import retry


# ---------------------

# 页面获取函数
def get_page(page, keyword):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/69.0.3497.12 Safari/537.36 '
    }
    print('正在爬取第', page, '页')
    url = 'https://www.zhipin.com/c101020100/?query={k}&page={page}&ka=page-{page}'.format(page=page, k=keyword)
    response = requests.get(url, headers=header)
    return response.text


# --------------
@retry(wait_fixed=8000)
def job_detail(link):
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/69.0.3497.12 Safari/537.36 '
    }
    response = requests.get(link, headers=header)
    data = etree.HTML(response.text)

    # ---检验是否出现验证码
    tips = data.xpath('/html/head/title/text()')
    tips_title = 'BOSS直聘验证码'
    if tips[0] == tips_title:
        print('检查是否弹出验证码')
        # 弹出验证码则引发IOError来进行循环
        raise IOError
    # ----------------------
    job_desc = data.xpath('//*[@id="main"]/div[3]/div/div[2]/div[3]/div[@class="job-sec"][1]/div/text()')

    jd = "".join(job_desc).strip()
    return jd


def parse_page(html, keyword, page):
    # 观察数据结构可得
    data = etree.HTML(html)
    if page == 1:
        items = data.xpath('//*[@id="main"]/div/div[3]/ul/li')
    else:
        items = data.xpath('//*[@id="main"]/div/div[2]/ul/li')
    for item in items:
        district = item.xpath('./div/div[1]/p/text()[1]')[0]
        job_links = item.xpath('./div/div[1]/h3/a/@href')[0]
        job_title = item.xpath('./div/div[1]/h3/a/div[1]/text()')[0]
        job_salary = item.xpath('./div/div[1]/h3/a/span/text()')[0]
        job_company = item.xpath('./div/div[2]/div/h3/a/text()')[0]
        job_experience = item.xpath('./div/div[1]/p/text()[2]')[0]
        job_degree = item.xpath('./div/div[1]/p/text()[3]')[0]
        fin_status = item.xpath('./div/div[2]/div/p/text()[2]')[0]
        try:
            company_scale = item.xpath('./div/div[2]/div/p/text()[3]')[0]
        except Exception:
            company_scale = item.xpath('./div/div[2]/div/p/text()[2]')[0]
        job_link = host + job_links
        # print(job_link)
        # 获取职位描述
        detail = job_detail(job_link)
        # ---------------
        job = {
            'Keyword': keyword,
            '地区': district,
            '职位名称': job_title,
            '职位薪资': job_salary,
            '公司名称': job_company,
            '工作经验': job_experience,
            '学历要求': job_degree,
            '公司规模': company_scale,
            '融资情况': fin_status,
            '职位描述': detail,
        }
        print(job)
        save_to_mongo(job)
        time.sleep(random.randint(6, 9))
        # ---------------------------------------


# 连接到MongoDB
MONGO_URL = 'localhost'
MONGO_DB = 'Graduation_project'
MONGO_COLLECTION = 'shanghai_discovery'
client = pymongo.MongoClient(MONGO_URL, port=27017)
db = client[MONGO_DB]


def save_to_mongo(data):
    # 保存到MongoDB中
    try:
        if db[MONGO_COLLECTION].insert(data):
            print('存储到 MongoDB 成功')
    except Exception:
        print('存储到 MongoDB 失败')


if __name__ == '__main__':
    MAX_PAGE = 10
    host = 'https://www.zhipin.com'
    keywords = ['数据分析', '数据挖掘', '商业分析', '机器学习']
    for keyword in keywords:
        for i in range(1, MAX_PAGE + 1):
            html = get_page(i, keyword)
            # ------------ 解析数据 ---------------
            parse_page(html, keyword, i)
            print('-' * 100)
            # -----------------
            timewait = random.randint(15, 18)
            time.sleep(timewait)
            print('等待', timewait, '秒')
