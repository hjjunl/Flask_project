import io
import os
import pickle
import time

import joblib
import numpy as np
# import plot as plot
from matplotlib import pyplot
from numpy import dot
from numpy.linalg import norm
from sqlalchemy import create_engine
import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from openpyxl import load_workbook
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webelement import WebElement
from webdriver_manager.chrome import ChromeDriverManager  # 'webdriver_manager' 패키지모듈 다운로드 필요
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from typing import List
import pandas as pd

path = 'static/model/model.pkl'
dbCon = pymysql.connect(host='127.0.0.1', user='root', db='testdb', passwd='2000', charset='utf8')

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


# CRUD 기능
class MyEmpDao:
    def __init__(self):
        pass

    def getEmps(self):
        ret = []
        db = pymysql.connect(host="127.0.0.1", user="root", passwd="2000", db="testdb", charset="utf8")

        curs = db.cursor()

        sql = "SELECT * from user_info"
        curs.execute(sql)

        rows = curs.fetchall()
        for e in rows:
            temp = {'BADGE': e[0], 'name': e[1], 'department': e[2], 'join_date': e[3], 'gender': e[4],
                    'position': e[5]}
            ret.append(temp)

        db.commit()
        db.close()
        return ret

    def insEmp(self, BADGE, name, department, gender, position):
        db = pymysql.connect(host="127.0.0.1", user="root", passwd="2000", db="testdb", charset="utf8")
        curs = db.cursor()

        sql = '''insert into user_info (BADGE, name, department, gender, position) values(%s,%s,%s,%s,%s)'''
        curs.execute(sql, (BADGE, name, department, gender, position))
        db.commit()
        db.close()

    def updEmp(self, name, department, join_date, gender, position, BADGE):
        db = pymysql.connect(host="127.0.0.1", user="root", passwd="2000", db="testdb", charset="utf8")
        curs = db.cursor()

        sql = "update user_info set name=%s, department=%s, join_date=%s,  position=%s where BADGE=%s"
        curs.execute(sql, (name, department, join_date, gender, position, BADGE))
        db.commit()
        db.close()

    def delEmp(self, BADGE):
        db = pymysql.connect(host="127.0.0.1", user="root", passwd="2000", db="testdb", charset="utf8")
        curs = db.cursor()

        sql = "delete from user_info where BADGE=%s"
        curs.execute(sql, BADGE)
        db.commit()
        db.close()

    ##################
    def updEmp1(self, name, department, position, BADGE):
        db = pymysql.connect(host="127.0.0.1", user="root", passwd="2000", db="testdb", charset="utf8")
        curs = db.cursor()

        sql = "update user_info set name=%s, department=%s, position=%s where BADGE=%s"
        curs.execute(sql, (name, department, position, BADGE))
        db.commit()
        db.close()


# 조회
def select_all():
    conn = pymysql.connect(host='127.0.0.1', user='root', db='testdb', passwd='2000', charset='utf8')
    arr = []
    try:
        with conn.cursor() as curs:
            sql = "select * from user_info"
            curs.execute(sql)
            rs = curs.fetchall()
            for row in rs:
                arr.append(row)

            db_df = pd.DataFrame(columns=['BADGE', 'name', 'department', 'join_date', 'gender', 'position'],
                                 data=arr)
    finally:
        conn.close()

    return db_df


# insert datas into Table 엑셀 데이터 db에 dataframe형식으로 바로 저장
def insert_excel_to_db(df):
    pymysql.install_as_MySQLdb()

    engine = create_engine("mysql+mysqldb://root:" + "2000" + "@127.0.0.1:3306/testdb", encoding='utf-8')
    df.to_sql(name='user_info', con=engine, if_exists='append', index=False)


# 조회: 여러 경우에 수에 따라 chart 시각화 및 조회
def select_emp(BADGE, name, department):
    conn = pymysql.connect(host='127.0.0.1', user='root', db='testdb', passwd='2000', charset='utf8')
    arr = []
    try:
        with conn.cursor() as curs:
            if BADGE == '' and name == '' and department == '':
                sql = '''select u.BADGE, u.name, d.department_name, u.join_date, u.gender, u.position, sum(p.payment) as payment
                        from user_info u, department d, payment_info p WHERE u.department=d.department_key AND u.BADGE=p.badge_id GROUP BY badge_id;'''
                curs.execute(sql)
                rs = curs.fetchall()
            elif BADGE != '' and name != '':
                sql = '''select u.BADGE, u.name, d.department_name, u.join_date, 
                u.gender, u.position, sum(p.payment) as payment
                from user_info u, department d, payment_info p 
                WHERE u.department=d.department_key AND u.BADGE=p.badge_id 
                AND u.name=''' + '\'' + name + '\' and u.BADGE=' + '\'' + BADGE + '\''
                curs.execute(sql)
                rs = curs.fetchall()
            elif BADGE == '' and name != '':
                sql = '''select u.BADGE, u.name, d.department_name, u.join_date, 
                u.gender, u.position, sum(p.payment) as payment
                from user_info u, department d, payment_info p 
                WHERE u.department=d.department_key AND u.BADGE=p.badge_id 
                AND u.name=''' + '\'' + name + '\'' + 'group by payment'
                curs.execute(sql)
                rs = curs.fetchall()
            elif BADGE != '' and name == '':
                sql = '''select u.BADGE, u.name, d.department_name, u.join_date, 
                u.gender, u.position, sum(p.payment) as payment
                from user_info u, department d, payment_info p 
                WHERE u.department=d.department_key AND u.BADGE=p.badge_id 
                AND u.BADGE=''' + '\'' + BADGE + '\''
                curs.execute(sql)
                rs = curs.fetchall()
            elif department != '':
                sql = '''select u.BADGE, u.name, d.department_name, u.join_date, 
                u.gender, u.position, sum(p.payment) as payment
                from user_info u, department d, payment_info p 
                WHERE u.department=d.department_key AND u.BADGE=p.badge_id 
                AND d.department_name=''' + '\'' + department + '\'' + 'group by u.name'
                curs.execute(sql)
                rs = curs.fetchall()

            for e in rs:
                temp = {'BADGE': e[0], 'name': e[1], 'department': e[2], 'join_date': e[3], 'gender': e[4],
                        'position': e[5], 'payment': e[6]}
                arr.append(temp)
            db_df = pd.DataFrame(
                columns=['BADGE', 'name', 'department', 'join_date', 'gender', 'position', 'payment'],
                data=arr)
            # print(db_df)
    finally:
        conn.close()
    return arr, db_df


# 개인 정보 상세 조회: BADGE(사번)에 따라 조회
def personal_data(BADGE):
    conn = pymysql.connect(host='127.0.0.1', user='root', db='testdb', passwd='2000', charset='utf8')
    arr = []
    try:
        with conn.cursor() as curs:
            sql = '''SELECT user_info.name, user_score.score, user_score.test_date, user_score.test_type from user_info
            LEFT JOIN user_score ON user_info.BADGE = user_score.badge_id
            WHERE user_info.BADGE=''' + str(BADGE)
            curs.execute(sql)
            rs = curs.fetchall()
            for row in rs:
                arr.append(row)
            arr2 = []
            for i in range(len(arr)):
                arr1 = [arr[i][0], arr[i][1], arr[i][2], arr[i][3]]
                arr2.append(arr1)
            print(arr2)
            for i in range(3):
                if arr1[i] is None:
                    arr1[i] = 'No data'
    finally:
        conn.close()
    return arr2


# 데이터 학습 (여기서 사용하지는 않음)
def employee_train(df):
    df.reset_index(drop=True, inplace=True)
    df.drop(['name', 'BADGE', 'join_date'], axis=1, inplace=True)
    sex = {'남': 1, '여': 0}
    df['gender'] = df['gender'].map(lambda x: sex[x])

    department = {'IT': 1, 'R&D': 2, 'HR': 3, 'Manufacturing': 4, 'Dataanalyist': 5}
    df['department'] = df['department'].map(lambda x: department[x])

    position = {'사원': 1, '책임': 2, '수석': 3}
    df['position'] = df['position'].map(lambda x: position[x])
    df['payment'] = df['payment'].apply(pd.to_numeric)
    df.reset_index(drop=True, inplace=True)
    # X를 통해 y를 예측하고자 함으로 X 와 y를 나눠줘야 함
    X = df.drop('payment', axis=1)
    # 예측을 하고자 하는 y값
    y = df['payment']
    # 학습데이터와 테스트 데이터 분리 0.5는 50%로 나누겠다는 뜻
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    my_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
    # 모델 학습
    model = my_model.fit(X_train, y_train)
    # 생성된 모델을 해당 path에 저장
    joblib.dump(model, path)


# 예시로 예측 값과 실제 값을 비교 (의미는 없음)
def emp_pre_ex(df):
    print(df)
    df.reset_index(drop=True, inplace=True)
    # 데이터 전처리, 활용할 변수: 성별, 직급, 부서 nominal variable(명목변수), ordinal variable: 순위 변수
    name_df = df['name']
    df.drop(['name', 'BADGE', 'join_date'], axis=1, inplace=True)
    sex = {'남': 1, '여': 0}
    df['gender'] = df['gender'].map(lambda x: sex[x])

    department = {'IT': 1, 'R&D': 2, 'HR': 3, 'Manufacturing': 4, 'Dataanalyist': 5}
    df['department'] = df['department'].map(lambda x: department[x])

    position = {'사원': 1, '책임': 2, '수석': 3}
    df['position'] = df['position'].map(lambda x: position[x])
    df['payment'] = df['payment'].apply(pd.to_numeric)
    df.reset_index(drop=True, inplace=True)

    print(df)
    X = df.drop('payment', axis=1)
    y = df['payment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # 여기서 validation set을 따로 생성하여 MSE, MAE등
    # 정확성에 대한 질적인 척도가 필요하나 어짜피 임이의 값이기 때문에 확인하지 않음
    # 100개의 tree를 생성하겠다는 뜻 rf의 동작원리를 알아야 함 (지금 같이 몇개 없을 경우는 100개까지 없어도 모든 경우의 수 계산이 가능하다)
    # bagging이라고 예를 들면 1000개의 속성 중 임의로 100개씩 골라서 의사결정 나무를 생성한다. 중복을 허용하여 생성한 후 결과 값의 평균을 낸다. 지금은 속성이 3개...
    my_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
    col = ['department', 'gender', 'position']
    index = np.arange(len(col))
    path = 'static/model/model.pkl'
    # 모델이 해당 경로에 있으면 기존 모델 사용
    if os.path.isfile(path):
        model = joblib.load(path)

        # 데이터 간의 상관 관계 분석
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i in range(3):
            print('Feature: %s, Score: %.5f' % (col[i], importance[i]))

        # plot feature importance
        # pyplot.bar([x for x in range(len(importance))], importance)
        # pyplot.title('Feature Importance')
        # pyplot.xlabel = ('Features')
        # pyplot.ylabel = ('Importance')
        # pyplot.xticks(index, col, fontsize = 15)
        # pyplot.show()

        rf_predicted = model.predict(X_test)
        print("Train score: ", model.score(X_train, y_train))
        print("Test score: ", model.score(X_test, y_test))
    # 모델이 해당 경로에 없을 시 새로 생성
    else:
        # 학습
        model = my_model.fit(X_train, y_train)
        # 설정한 path에 모델 저장
        joblib.dump(model, path)

        # 데이터 간의 상관 관계 분석
        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i in range(3):
            print('Feature!!: %0str, Score: %.5f' % (col[i], importance[i]))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.title('Feature Importance')
        pyplot.xlabel = ('Features')
        pyplot.ylabel = ('Importance')
        pyplot.xticks(index, col, fontsize=15)
        pyplot.show()
        rf_predicted = model.predict(X_test)

    pyplot.close()
    real_data = list(y_train)
    predicted_data = rf_predicted
    name = []
    for i in list(y_train.index):
        name.append(name_df[i])
    real_data = list(map(int, real_data))
    predicted_data = list(map(int, predicted_data))

    return real_data, predicted_data, name


# 신입 직원 연봉 예측  (여기서 입력받은 df는 excel에서 업로드한 데이터를 dataframe화 한 데이터)
def emp_prediction(df):
    print(df)
    # 데이터 전처리
    df.reset_index(drop=True, inplace=True)
    # 데이터 전처리, 활용할 변수: 성별, 직급, 부서 nominal variable(명목변수), ordinal variable: 순위 변수
    BADGE_df = df['BADGE']
    name_df = df['name']
    join_date_df = df['join_date']
    gender_df = df['gender']
    department_df = df['department']
    position_df = df['position']
    df.drop(['name', 'BADGE', 'join_date'], axis=1, inplace=True)
    sex = {'남': 1, '여': 0}
    df['gender'] = df['gender'].map(lambda x: sex[x])

    department = {'IT': 1, 'R&D': 2, 'HR': 3, 'Manufacturing': 4, 'Dataanalyist': 5}
    df['department'] = df['department'].map(lambda x: department[x])

    position = {'사원': 1, '책임': 2, '수석': 3}
    df['position'] = df['position'].map(lambda x: position[x])
    # 기존 모델을 해당 path에서 불러오기
    model = joblib.load(path)
    # rf_predicted는 random forest regressor 모델로 예측한 값
    rf_predicted = model.predict(df)
    # decimal 형태로 나타나므로 int형으로 형변환
    rf_predicted = list(map(int, rf_predicted))

    arr = []
    # javascript에 출력하기 위한 데이터 처리
    for i in range(len(list(BADGE_df))):
        temp = {'BADGE': list(BADGE_df)[i], 'name': list(name_df)[i], 'department': list(department_df)[i],
                'join_date': list(join_date_df)[i],
                'gender': list(gender_df)[i], 'position': list(position_df)[i], 'payment': rf_predicted[i]}
        arr.append(temp)
    print(df)
    return rf_predicted, list(name_df), arr


########################################################################################

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
                     'growth_pos'],
            data=arr)
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
        user_1 = [float(com_review_seg), float(welfare_sal), float(wo_la_bal), float(com_cul), float(opportunity), float(com_head),
                  float(growth_pos_seg), float(com_rec_seg), float(CEO_sup_seg)]

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
                       growth_pos_seg, com_rec_seg, CEO_sup_seg, com_name]

        sql_col = ['user_id', 'mean_sal', 'mean_star', 'com_review_seg', 'welfare_sal', 'wo_la_bal', \
                   'com_cul', 'opportunity', 'com_head', 'growth_pos_seg', 'com_rec_seg',
                   'CEO_sup_seg', 'com_result']
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
                # print("there is id")
                sql = '''SELECT job_planet.com_name, com_info.* FROM com_info LEFT JOIN job_planet ON com_info.com_id=job_planet.id where job_planet.id ='''\
                      + "'" + str(com_id) + "';"
                curs.execute(sql)
                rs = curs.fetchall()
                for e in rs:
                    temp = {'com_name': e[0], 'com_bis': e[6], 'com_emp': e[7], 'com_div': e[8],
                            'com_est': e[9], 'com_capital': e[10], 'com_rev': e[11], 'com_sal': e[12], 'com_ceo': e[13],
                            'com_main_bis': e[14], 'com_en': e[15], 'com_page': e[16], 'com_address': e[17],
                            'com_rel_com': e[18]}
                    arr1.append(temp)
                # print(arr1)
                column_kor = ['기업명', '산업', '사원수', '기업구분', '설립일', '자본금', '매출액', '대졸초임', '대표자', '주요사업', '4대보험', '홈페이지',
                              '주소', '계열사']
                column_en = ['com_name', 'com_bis', 'com_emp', 'com_div', 'com_est', 'com_capital', 'com_rev', \
                             'com_sal', 'com_ceo', 'com_main_bis', 'com_en', 'com_page', 'com_address', 'com_rel_com']
                df = pd.DataFrame(columns=column_en, data=arr1)
                df.columns = column_kor
                df.reset_index(drop=True, inplace=True)
                # print(df)
            else:
                url = 'https://www.jobkorea.co.kr/'
                options = Options()
                # 화면 안보이게
                options.add_argument("headless")
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
                    column = ['com_id', 'com_bis', 'com_emp', 'com_div', 'com_est', 'com_capital', 'com_rev', \
                              'com_sal', 'com_ceo', 'com_main_bis', 'com_en', 'com_page', 'com_address', 'com_rel_com']
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
                        print(driver.current_url)
                        try:
                            com_data = wait_element_ready(driver,
                                                      '//*[@id="company-body"]/div[1]/div[2]/div/table/tbody').text
                        except Exception:
                            com_data = wait_element_ready(driver,'//*[@id="company-body"]/div[1]/div[1]/div/table/tbody').text

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
                column_kor = ['번호', '산업', '사원수', '기업구분', '설립일', '자본금', '매출액', '대졸초임', '대표자', '주요사업', '4대보험', '홈페이지',
                              '주소', '계열사']
                column_en = ['com_name', 'com_bis', 'com_emp', 'com_div', 'com_est', 'com_capital', 'com_rev', \
                             'com_sal', 'com_ceo', 'com_main_bis', 'com_en', 'com_page', 'com_address', 'com_rel_com']
                # df.columns = column_kor
                for df_name in list(df.columns):
                    if df_name == 'com_name':
                        df['회사명'] = df['com_name']
                        df = df.drop('com_name', axis = 1)
                    if df_name == 'com_bis':
                        df['사업'] = df['com_bis']
                        df = df.drop('com_bis', axis = 1)
                    if df_name == 'com_emp':
                        df['사원수'] = df['com_emp']
                        df = df.drop('com_emp', axis = 1)
                    if df_name == 'com_div':
                        df['기업구분'] = df['com_div']
                        df = df.drop('com_div', axis = 1)
                    if df_name == 'com_est':
                        df['설립일'] = df['com_est']
                        df = df.drop('com_est', axis = 1)
                    if df_name == 'com_capital':
                        df['자본금'] = df['com_capital']
                        df = df.drop('com_capital', axis = 1)
                    if df_name == 'com_rev':
                        df['매출액'] = df['com_rev']
                        df = df.drop('com_rev', axis = 1)
                    if df_name == 'com_sal':
                        df['대졸초임'] = df['com_sal']
                        df = df.drop('com_sal', axis = 1)
                    if df_name == 'com_ceo':
                        df['CEO'] = df['com_ceo']
                        df = df.drop('com_ceo', axis = 1)
                    if df_name == 'com_main_bis':
                        df['주요사업'] = df['com_main_bis']
                        df = df.drop('com_main_bis', axis = 1)
                    if df_name == 'com_en':
                        df['보험'] = df['com_en']
                        df = df.drop('com_en', axis = 1)
                    if df_name == 'com_page':
                        df['Page'] = df['com_page']
                        df = df.drop('com_page', axis = 1)
                    if df_name == 'com_address':
                        df['주소'] = df['com_address']
                        df = df.drop('com_address', axis = 1)
                    if df_name == 'com_rel_com':
                        df['관련회사'] = df['com_rel_com']
                        df = df.drop('com_rel_com', axis = 1)
                df.reset_index(drop=True, inplace=True)


    finally:
        conn.close()

    return df


# com = pd.read_excel('job_planet_mod.xlsx')
# company = list(com['com_name'][1168:])
# print(company)
# for i, j in enumerate(company):
#     check_com_info(j, i+1)
    # print(i)
    # print(j)

# if __name__ == '__main__':
    # mean_sal, mean_star, welfare_sal, wo_la_bal, com_cul, opportunity, com_head,
    # com_review_seg, growth_pos_seg, com_rec_seg, CEO_sup_seg
    # job_recomendation(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
