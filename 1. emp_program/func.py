import io
import os
import pickle
import joblib
import numpy as np
import plot as plot
from matplotlib import pyplot
from sqlalchemy import create_engine
import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

path = 'static/model/model.pkl'
dbCon = pymysql.connect(host = '127.0.0.1', user = 'root', db = 'testdb', passwd = '2000', charset = 'utf8')


# CRUD 기능
class MyEmpDao:
    def __init__(self):
        pass

    def getEmps(self):
        ret = []
        db = pymysql.connect(host = "127.0.0.1", user = "root", passwd = "2000", db = "testdb", charset = "utf8")

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
        db = pymysql.connect(host = "127.0.0.1", user = "root", passwd = "2000", db = "testdb", charset = "utf8")
        curs = db.cursor()

        sql = '''insert into user_info (BADGE, name, department, gender, position) values(%s,%s,%s,%s,%s)'''
        curs.execute(sql, (BADGE, name, department, gender, position))
        db.commit()
        db.close()

    def updEmp(self, name, department, join_date, gender, position, BADGE):
        db = pymysql.connect(host = "127.0.0.1", user = "root", passwd = "2000", db = "testdb", charset = "utf8")
        curs = db.cursor()

        sql = "update user_info set name=%s, department=%s, join_date=%s,  position=%s where BADGE=%s"
        curs.execute(sql, (name, department, join_date, gender, position, BADGE))
        db.commit()
        db.close()

    def delEmp(self, BADGE):
        db = pymysql.connect(host = "127.0.0.1", user = "root", passwd = "2000", db = "testdb", charset = "utf8")
        curs = db.cursor()

        sql = "delete from user_info where BADGE=%s"
        curs.execute(sql, BADGE)
        db.commit()
        db.close()

    ##################
    def updEmp1(self, name, department, position, BADGE):
        db = pymysql.connect(host = "127.0.0.1", user = "root", passwd = "2000", db = "testdb", charset = "utf8")
        curs = db.cursor()

        sql = "update user_info set name=%s, department=%s, position=%s where BADGE=%s"
        curs.execute(sql, (name, department, position, BADGE))
        db.commit()
        db.close()


# 조회
def select_all():
    conn = pymysql.connect(host = '127.0.0.1', user = 'root', db = 'testdb', passwd = '2000', charset = 'utf8')
    arr = []
    try:
        with conn.cursor() as curs:
            sql = "select * from user_info"
            curs.execute(sql)
            rs = curs.fetchall()
            for row in rs:
                arr.append(row)

            db_df = pd.DataFrame(columns = ['BADGE', 'name', 'department', 'join_date', 'gender', 'position'],
                                 data = arr)
    finally:
        conn.close()

    return db_df


# insert datas into Table 엑셀 데이터 db에 dataframe형식으로 바로 저장
def insert_excel_to_db(df):
    pymysql.install_as_MySQLdb()

    engine = create_engine("mysql+mysqldb://root:" + "2000" + "@127.0.0.1:3306/testdb", encoding = 'utf-8')
    df.to_sql(name = 'user_info', con = engine, if_exists = 'append', index = False)


# 조회: 여러 경우에 수에 따라 chart 시각화 및 조회
def select_emp(BADGE, name, department):
    conn = pymysql.connect(host = '127.0.0.1', user = 'root', db = 'testdb', passwd = '2000', charset = 'utf8')
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
                columns = ['BADGE', 'name', 'department', 'join_date', 'gender', 'position', 'payment'],
                data = arr)
            # print(db_df)
    finally:
        conn.close()
    return arr, db_df


# 개인 정보 상세 조회: BADGE(사번)에 따라 조회
def personal_data(BADGE):
    conn = pymysql.connect(host = '127.0.0.1', user = 'root', db = 'testdb', passwd = '2000', charset = 'utf8')
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
    df.reset_index(drop = True, inplace = True)
    df.drop(['name', 'BADGE', 'join_date'], axis = 1, inplace = True)
    sex = {'남': 1, '여': 0}
    df['gender'] = df['gender'].map(lambda x: sex[x])

    department = {'IT': 1, 'R&D': 2, 'HR': 3, 'Manufacturing': 4, 'Dataanalyist': 5}
    df['department'] = df['department'].map(lambda x: department[x])

    position = {'사원': 1, '책임': 2, '수석': 3}
    df['position'] = df['position'].map(lambda x: position[x])
    df['payment'] = df['payment'].apply(pd.to_numeric)
    df.reset_index(drop = True, inplace = True)
    # X를 통해 y를 예측하고자 함으로 X 와 y를 나눠줘야 함
    X = df.drop('payment', axis = 1)
    # 예측을 하고자 하는 y값
    y = df['payment']
    # 학습데이터와 테스트 데이터 분리 0.5는 50%로 나누겠다는 뜻
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)
    my_model = RandomForestRegressor(n_estimators = 100, criterion = 'mae', random_state = 0)
    # 모델 학습
    model = my_model.fit(X_train, y_train)
    # 생성된 모델을 해당 path에 저장
    joblib.dump(model, path)


# 예시로 예측 값과 실제 값을 비교 (의미는 없음)
def emp_pre_ex(df):
    print(df)
    df.reset_index(drop = True, inplace = True)
    # 데이터 전처리, 활용할 변수: 성별, 직급, 부서 nominal variable(명목변수), ordinal variable: 순위 변수
    name_df = df['name']
    df.drop(['name', 'BADGE', 'join_date'], axis = 1, inplace = True)
    sex = {'남': 1, '여': 0}
    df['gender'] = df['gender'].map(lambda x: sex[x])

    department = {'IT': 1, 'R&D': 2, 'HR': 3, 'Manufacturing': 4, 'Dataanalyist': 5}
    df['department'] = df['department'].map(lambda x: department[x])

    position = {'사원': 1, '책임': 2, '수석': 3}
    df['position'] = df['position'].map(lambda x: position[x])
    df['payment'] = df['payment'].apply(pd.to_numeric)
    df.reset_index(drop = True, inplace = True)

    print(df)
    X = df.drop('payment', axis = 1)
    y = df['payment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)

    # 여기서 validation set을 따로 생성하여 MSE, MAE등
    # 정확성에 대한 질적인 척도가 필요하나 어짜피 임이의 값이기 때문에 확인하지 않음
    # 100개의 tree를 생성하겠다는 뜻 rf의 동작원리를 알아야 함 (지금 같이 몇개 없을 경우는 100개까지 없어도 모든 경우의 수 계산이 가능하다)
    # bagging이라고 예를 들면 1000개의 속성 중 임의로 100개씩 골라서 의사결정 나무를 생성한다. 중복을 허용하여 생성한 후 결과 값의 평균을 낸다. 지금은 속성이 3개...
    my_model = RandomForestRegressor(n_estimators = 100, criterion = 'mae', random_state = 0)
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
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.title('Feature Importance')
        pyplot.xlabel = ('Features')
        pyplot.ylabel = ('Importance')
        pyplot.xticks(index, col, fontsize = 15)
        pyplot.show()

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
        pyplot.xticks(index, col, fontsize = 15)
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
    df.reset_index(drop = True, inplace = True)
    # 데이터 전처리, 활용할 변수: 성별, 직급, 부서 nominal variable(명목변수), ordinal variable: 순위 변수
    BADGE_df = df['BADGE']
    name_df = df['name']
    join_date_df = df['join_date']
    gender_df = df['gender']
    department_df = df['department']
    position_df = df['position']
    df.drop(['name', 'BADGE', 'join_date'], axis = 1, inplace = True)
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


if __name__ == '__main__':
    select_all()
