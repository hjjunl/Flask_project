
import io
import pandas as pd
from sqlalchemy import create_engine
import MySQLdb
import pandas as pd
from datetime import datetime
import pymysql
import schedule
import time
from apscheduler.schedulers.blocking import BlockingScheduler

dbCon = pymysql.connect(host='127.0.0.1', user='root', db='testdb', passwd='2000', charset='utf8')


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

            db_df = pd.DataFrame(columns=['BADGE', 'name', 'department', 'join_date', 'gender', 'position'], data=arr)
    finally:
        conn.close()
    # data = pd.read_excel('test.xlsx')
    #
    # print(list(data['BADGE']))
    # print(list(db_df['BADGE']))
    # for i in list(data['BADGE']):
    #     if i not in list(db_df['BADGE']):
    #         print("there is num")
    return db_df


# insert datas into Table
def insert_excel_to_db(df):
    pymysql.install_as_MySQLdb()

    engine = create_engine("mysql+mysqldb://root:" + "2000" + "@127.0.0.1:3306/testdb", encoding='utf-8')
    conn = engine.connect()
    df.to_sql(name='stock_predict', con=engine, if_exists='append', index=False)


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
            db_df = pd.DataFrame(columns=['BADGE', 'name', 'department', 'join_date', 'gender', 'position', 'payment'],
                                 data=arr)
            # print(db_df)
    finally:
        conn.close()
    return arr, db_df


def personal_data(BADGE):
    conn = pymysql.connect(host='127.0.0.1', user='root', db='testdb', passwd='2000', charset='utf8')
    arr = []
    try:
        with conn.cursor() as curs:
            sql = '''SELECT user_info.name, user_score.score, user_score.test_date from user_info
            LEFT JOIN user_score ON user_info.BADGE = user_score.badge_id
            WHERE user_info.BADGE=''' + str(BADGE)
            curs.execute(sql)
            rs = curs.fetchall()
            for row in rs:
                arr.append(row)
            arr2 = []
            for i in range(len(arr)):
                arr1 = []
                arr1.append(arr[i][0])
                arr1.append(arr[i][1])
                arr1.append(arr[i][2])
                arr2.append(arr1)
            print(arr2)
            for i in range(3):
                if arr1[i] is None:
                    arr1[i] = 'No data'
    finally:
        conn.close()
    return arr2




if __name__ == '__main__':
    select_all()

