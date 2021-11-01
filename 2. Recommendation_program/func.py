import io
import os
import pickle
import joblib
import numpy as np
import plot as plot
from matplotlib import pyplot
from numpy import dot
from numpy.linalg import norm
from sqlalchemy import create_engine
import pandas as pd
import pymysql


WAIT_TIME = 4


def wait_element_ready(_driver: webdriver, xpath: str, wait: int = WAIT_TIME) -> WebElement:
    WebDriverWait(_driver, wait).until(
        expected_conditions.presence_of_all_elements_located(
            (By.XPATH, xpath)))
    web_element = _driver.find_element_by_xpath(xpath)
    return web_element


def wait_elements_ready(_driver: webdriver, xpath: str, wait: int = WAIT_TIME) -> List:
    WebDriverWait(_driver, wait).until(
        expected_conditions.presence_of_all_elements_located(
            (By.XPATH, xpath)))
    web_elements = _driver.find_elements_by_xpath(xpath)
    return web_elements

# IT 기업 추천 (Item based collaborative filtering)
def job_recomendation(user, mean_sal, mean_star, com_review_seg, welfare_sal, wo_la_bal, com_cul, opportunity, com_head,
                      growth_pos_seg, com_rec_seg, CEO_sup_seg):
    pymysql.install_as_MySQLdb()
    engine = create_engine("mysql+mysqldb://root:" + "2000" + "@127.0.0.1:3306/testdb", encoding='utf-8')
    conn = pymysql.connect(host='127.0.0.1', user='root', db='testdb', passwd='2000', charset='utf8')
    # 추천 할 값과 회사
    rec_com_list = []
    # 초기 상태
    if mean_sal == '' or mean_star == '' or welfare_sal == '' or com_review_seg == '':
        rec_com_list = [['']]
    else:
        arr = []
        with conn.cursor() as curs:
            sql = "select * from job_planet"
            curs.execute(sql)
            rs = curs.fetchall()
            for row in rs:
                arr.append(row)

        df = pd.DataFrame(
            columns=['id', 'com_name', 'com_relation', 'mean_star', 'com_review', 'mean_sal', 'welfare_sal', \
                     'wo_la_bal', 'com_cul', 'opportunity', 'com_head', 'com_rec', 'CEO_sup', \
                     'growth_pos'], data=arr
         )
        labels = np.arange(1, 6, 1)

        # 평균 연복은 크게 5개로 나눔 cut
        mean_sal_seg_series = pd.Series(list(pd.cut(df['mean_sal'], 5, labels=labels)),
                                        name='mean_sal_seg')  # 2800씩 등차함수

        # 나머지는 각 수에 맞춰 qcut
        com_rec_seg_series = pd.Series(list(pd.cut(df['com_rec'], 5, labels=labels)), name='com_rec_seg')
        CEO_sup_seg_series = pd.Series(list(pd.qcut(df['CEO_sup'], 5, labels=labels)), name='CEO_sup_seg')
        growth_pos_seg_series = pd.Series(list(pd.qcut(df['growth_pos'], 5, labels=labels)), name='growth_pos_seg')
        # com_review 인지도로 나타낸다 (리뷰수)
        com_review_seg_series = pd.Series(list(pd.qcut(df['com_review'], 5, labels=labels)), name='com_review_seg')
        # 데이터 병합 1~5점으로 변환
        df = pd.concat([df, com_review_seg_series, growth_pos_seg_series, com_rec_seg_series, mean_sal_seg_series,
                        CEO_sup_seg_series], axis=1)
        df.reset_index(drop=True, inplace=True)

        # 연봉 범위 정할 때
        if mean_sal >= 0 and mean_sal <= 1.5:
            df = df[(df['mean_sal_seg'] == 1)]
        if mean_sal <= 2.5 and mean_sal > 1.5:
            df = df[(df['mean_sal_seg'] == 2)]
        if mean_sal <= 3.5 and mean_sal > 2.5:
            df = df[(df['mean_sal_seg'] == 3)]
        if mean_sal <= 4.5 and mean_sal > 3.5:
            df = df[(df['mean_sal_seg'] == 4)]
        if mean_sal > 4.5:
            df = df[(df['mean_sal_seg'] == 5)]

        df.reset_index(drop=True, inplace=True)

        # Choose mean_star 평균 별점 선택시 +-0.5로 범위 설정
        if mean_star <= 1.5:
            df = df[(df['mean_star'] >= 0) & (df['mean_star'] <= 1.5)]
        if mean_star <= 2.5 and mean_star > 1.5:
            df = df[(df['mean_star'] > 1.5) & (df['mean_star'] <= 2.5)]
        if mean_star <= 3.5 and mean_star > 2.5:
            df = df[(df['mean_star'] > 2.5) & (df['mean_star'] <= 3.5)]
        if mean_star <= 4 and mean_star > 3.5:
            df = df[(df['mean_star'] > 3.5) & (df['mean_star'] <= 4.5)]
        if mean_star > 4.5:
            df = df[(df['mean_star'] >= 4)]
        # user가 선택한 값들
        user_1 = [int(com_review_seg), int(welfare_sal), int(wo_la_bal), int(com_cul), int(opportunity), int(com_head),
                  int(growth_pos_seg), int(com_rec_seg), int(CEO_sup_seg)]

        com_df = df.drop(['CEO_sup', 'com_rec', 'growth_pos', 'com_review', 'mean_sal_seg', 'com_relation'], axis=1)
        com_df.reset_index(drop=True, inplace=True)
        com_df = com_df[['id', 'com_name', 'mean_star', 'com_review_seg', 'mean_sal', 'welfare_sal', 'wo_la_bal', \
                         'com_cul', 'opportunity', 'com_head', 'growth_pos_seg', 'com_rec_seg',
                         'CEO_sup_seg']]
        # 행을 잘라 list로 붙임
        com_list = []
        for i in range(len(com_df)):
            com_list.append(list(com_df.loc[i]))

        # 함수 호출
        sim = cos_sim(user_1, com_list)
        for i, j in sim:
            j.insert(0, i)
            rec_com_list.append(j)

        print(rec_com_list)
        com_result = []
        for i in rec_com_list:
            com_result.append(i[1])
        com_name = ','.join(com_result)
        user_choice = [int(user), mean_sal, mean_star, com_review_seg, welfare_sal, wo_la_bal, com_cul, opportunity,
                       com_head,
                       growth_pos_seg, com_rec_seg, CEO_sup_seg, com_name
                      ]

        sql_col = ['user_id', 'mean_sal', 'mean_star', 'com_review_seg', 'welfare_sal', 'wo_la_bal', \
                   'com_cul', 'opportunity', 'com_head', 'growth_pos_seg', 'com_rec_seg',
                   'CEO_sup_seg', 'com_result'
                  ]
        user_choice = tuple(user_choice)

        df = pd.DataFrame([user_choice], columns=sql_col)
        print(df)
        df.reset_index(drop=True, inplace=True)

        df.to_sql(name='user_rec', con=engine, if_exists='append', index=False)
        print('user_choice', user_choice)
        return rec_com_list


# list 간 코사인 유사도 계산
def cos_sim(user, com_list_):
    com_id = []
    cos_sim_list = []
    com_name = []
    com_mean_star = []
    com_mean_sal = []
    for i in range(len(com_list_)):
        com_id.append(com_list_[i][0])
        com_mean_star.append(com_list_[i][2])
        com_mean_sal.append(com_list_[i][4])
        del com_list_[i][0]
        del com_list_[i][1]
        del com_list_[i][2]

    for i in com_list_:
        com_name.append(i[0])
    new_list = com_list_
    for i in range(len(com_list_)):
        com_list_[i].pop(0)

        try:
            cos_sim_list.append(np.dot(user, com_list_[i]) / (np.linalg.norm(user) * (np.linalg.norm(com_list_[i]))))
        except Exception:
            print("insert 0")
            cos_sim_list.append(0)
        #  평균 별 데이터 넣어주기
        new_list[i].insert(0, com_mean_star[i])
        new_list[i].insert(2, com_mean_sal[i])
    # 회사와 cos sim 묶은후 정렬
    for i in range(len(new_list)):
        new_list[i].insert(0, cos_sim_list[i])
        new_list[i].insert(0, com_name[i])

    rec_dic = dict(zip(com_id, new_list))
    rec_dic = sorted(rec_dic.items(), key=lambda x: x[1][1], reverse=True)
    rec_dic = rec_dic[:10]

    return rec_dic

# 기업정보 클릭시 자동 크롤링, 크롤링 데이터 자동 db저장 및 웹상으로 
def check_com_info(com_name, com_id):
    pymysql.install_as_MySQLdb()
    engine = create_engine("mysql+mysqldb://root:" + "2000" + "@127.0.0.1:3306/testdb", encoding='utf-8')
    conn = pymysql.connect(host='127.0.0.1', user='root', db='testdb', passwd='2000', charset='utf8')
    arr1 = []
    check = []
    try:
        with conn.cursor() as curs:
            sql = '''SELECT com_id FROM com_info;'''
            curs.execute(sql)
            rs = curs.fetchall()
            for e in rs:
                temp = {'id': e[0]}
                check.append(int(e[0]))
            print(check)
            if int(com_id) in check:
                print("there is id")
                sql = '''SELECT com_name, com_info.* FROM com_info LEFT JOIN job_planet ON com_info.com_id=job_planet.id where job_planet.id =''' + "'" + str(
                    com_id) + "';"
                curs.execute(sql)
                rs = curs.fetchall()
                for e in rs:
                    temp = {'com_name': e[0], 'com_bis': e[3], 'com_emp': e[4], 'com_div': e[5],
                            'com_est': e[6], 'com_capital': e[7], 'com_rev': e[8], 'com_sal': e[9], 'com_ceo': e[10],
                            'com_main_bis': e[11], 'com_en': e[12], 'com_page': e[13], 'com_address': e[14],
                            'com_rel_com': e[15]
                           }
                    arr1.append(temp)
                print(arr1)
                column_kor = ['기업명', '산업', '사원수', '기업구분', '설립일', '자본금', '매출액', '대졸초임', 
                              '대표자', '주요사업', '4대보험', '홈페이지', '주소', '계열사'
                             ]
                column_en = ['com_name', 'com_bis', 'com_emp', 'com_div', 'com_est', 'com_capital', 'com_rev', \
                             'com_sal', 'com_ceo', 'com_main_bis', 'com_en', 'com_page', 'com_address', 'com_rel_com'
                            ]
                df = pd.DataFrame(columns=column_en, data=arr1)
                df.columns = column_kor
                df.reset_index(drop=True, inplace=True)
                print(df)
            else:
                url = 'https://www.jobkorea.co.kr/'
                options = Options()
                # 화면 안보이게
                # options.add_argument("headless")
                driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)
                driver.get(url)

                # 기업명 클릭
                wait_element_ready(driver, '//*[@id="stext"]').send_keys(com_name)
                # click search
                wait_element_ready(driver, '//*[@id="common_search_btn"]').click()
                # 기업정보 클릭
                wait_element_ready(driver, '//*[@id="content"]/div/div/div[1]/div/div[1]/div/button[2]').click()
                try:
                    wait_element_ready(driver,
                                       '//*[@id="content"]/div/div/div[1]/div/div[3]/div[2]/div/div[1]/ul/li[1]/div/div[1]/div/a').click()
                except Exception:
                    print('No company')
                    column = [
                             'com_id', 'com_bis', 'com_emp', 'com_div', 'com_est', 'com_capital', 'com_rev', \
                             'com_sal', 'com_ceo', 'com_main_bis', 'com_en', 'com_page', 'com_address', 'com_rel_com'
                             ]
                    com_info_list = tuple([com_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    df = pd.DataFrame([com_info_list], columns=column)
                    print(df)
                    df.reset_index(drop=True, inplace=True)
                    conn = engine.connect()
                    df.to_sql(name='com_info', con=engine, if_exists='append', index=False)
                    driver.close()
                    return 0

                # 새로 바뀐 창으로 가기
                driver.switch_to.window(driver.window_handles[-1])
                print(driver.current_url)
                try:
                    print('바로 기업정보')
                    com_data = wait_element_ready(driver, '//*[@id="company-body"]/div[1]/div[1]/div/table/tbody').text

                except Exception:
                    # 기업 정보 클릭
                    print("기업 정보 클릭")
                    try:
                        wait_element_ready(driver, '/html/body/div[2]/div[4]/div[2]/div[2]/div/a[2]/div[1]').click()
                        com_data = wait_element_ready(driver, 
                                                      '//*[@id="company-body"]/div[1]/div[2]/div/table/tbody').text
                    except Exception:
                        print('예외처리무슨...')
                        com_data = wait_element_ready(driver,
                                                      '//*[@id="company-body"]/div[1]/div[2]/div/table/tbody').text

                com_data_list = []
                # 데이터 전처리
                com_data = com_data.replace('\n', "|")
                com_data = com_data.split('|')
                for i in range(len(com_data)):
                    arr = []
                    if '산업' in com_data[i] and len(com_data[i]) == 2:
                        arr.append('com_bis')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)
                    if '사원수' in com_data[i]:
                        arr.append('com_emp')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)
                    if '기업구분' in com_data[i]:
                        arr.append('com_div')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)

                    if '설립일' in com_data[i]:
                        arr.append('com_est')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)

                    if '자본금' in com_data[i]:
                        print("자본금 있다")
                        arr.append('com_capital')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)

                    if '매출액' in com_data[i]:
                        arr.append('com_rev')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)

                    if '대졸초임' in com_data[i]:
                        arr.append('com_sal')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)

                    if '대표자' in com_data[i]:
                        arr.append('com_ceo')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)

                    if '주요사업' in com_data[i]:
                        arr.append('com_main_bis')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)

                    if '4대보험' in com_data[i]:
                        arr.append('com_en')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)
                    if '홈페이지' in com_data[i]:
                        arr.append('com_page')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)
                    if '주소' in com_data[i]:
                        arr.append('com_address')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)
                    if '계열사' in com_data[i]:
                        arr.append('com_rel_com')
                        if i + 1 < len(com_data):
                            arr.append(com_data[i + 1])
                        com_data_list.append(arr)

                driver.close()
                # 초기화면으로 돌아오기
                driver.switch_to.window(driver.window_handles[0])

                driver.close()
                com_info_list = []
                com_current_col = []
                for i in com_data_list:
                    com_current_col.append(i[0])
                    com_info_list.append(i[1])
                com_info_list.insert(0, int(com_id))
                com_current_col.insert(0, 'com_id')
                com_info_list = tuple(com_info_list)
                print(com_info_list)
                df = pd.DataFrame([com_info_list], columns=com_current_col)
                df.reset_index(drop=True, inplace=True)
                print(df)
                # html = df.to_html(index=False, justify='center')
                conn = engine.connect()
                df.to_sql(name='com_info', con=engine, if_exists='append', index=False)

    finally:
        conn.close()

    return df
