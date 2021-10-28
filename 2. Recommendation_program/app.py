import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask.templating import render_template
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
# from sqlalchemy.dialects.mysql import pymysql
import pymysql

import func
from models import user_info

app = Flask(__name__)

# database 설정파일
app = Flask(__name__, static_url_path="", static_folder='static',
            template_folder='templates')
Bootstrap(app)

##############################################
rec_arr = []


@app.route('/rec_data.ajax', methods = ['POST'])
def rec_data_ajax():
    data = request.get_json()
    if data != None:
        mean_sal = data['mean_sal']
        mean_star = data['mean_star']
        welfare_sal = data['welfare_sal']
        wo_la_bal = data['wo_la_bal']
        com_cul = data['com_cul']
        opportunity = data['opportunity']
        com_head = data['com_head']
        com_review_seg = data['com_review_seg']
        growth_pos_seg = data['growth_pos_seg']
        com_rec_seg = data['com_rec_seg']
        CEO_sup_seg = data['CEO_sup_seg']
        user = data['user']
        input_list = [mean_sal, mean_star, com_review_seg, welfare_sal, wo_la_bal, com_cul, opportunity, com_head, \
                      growth_pos_seg, com_rec_seg, CEO_sup_seg]
        print(input_list)
        for k in range(len(input_list)):
            if input_list[k] == '':
                input_list[k] = 0
        print(input_list)
        cnt = func.job_recomendation(int(user), float(input_list[0]), float(input_list[1]), float(input_list[2]),
                                     float(input_list[3]),
                                     float(input_list[4]),
                                     float(input_list[5]), float(input_list[6]), float(input_list[7]),
                                     float(input_list[8]), float(input_list[9]), float(input_list[10]))

        rec_arr.append(cnt)
    result = 1
    return jsonify(result = result)


com_information = []


@app.route('/com_name.ajax', methods = ['POST'])
def com_name_ajax():
    data = request.get_json()
    if data != None:
        com_name = data['com_name']
        com_id = data['com_id']
        print(com_name)
        print(com_id)
        cnt = func.check_com_info(com_name, com_id)
        com_information.append(cnt)
    result = 1
    return jsonify(result = result)


# 추천 페이지
@app.route("/recommendation", methods = ['GET', 'POST'])
def recommendation():
    conn = pymysql.connect(host = '127.0.0.1', user = 'root', db = 'testdb', passwd = '2000', charset = 'utf8')
    job_planet_data = []
    rec_result = rec_arr
    user_data = []
    df = ''
    # db에 rkqtdl dlTsmswl ghkrdls
    with conn.cursor() as curs:
        user_sql = '''SELECT BADGE, NAME FROM user_info;'''
        curs.execute(user_sql)
        rss = curs.fetchall()
        for row in rss:
            user_data.append(row)
        rss = curs.fetchall()
    for e in rss:
        temp1 = {'BADGE': e[0], 'NAME': e[1]}
        user_data.append(temp1)
    if len(com_information) == 0:
        print('no df')
    else:
        print('yes df')
        df = com_information[0]
        try:
            df.reset_index(drop = True, inplace = True)
            df = df.to_html(index = False, justify = 'center')
        except Exception:
            df = df.to_html(index = False, justify = 'center')
        if len(com_information) != 0:
            com_information.pop(0)

    if len(rec_result) == 0:
        print("no input yet")
        with conn.cursor() as curs:
            sql = "select * from job_planet"
            curs.execute(sql)
            rs = curs.fetchall()
            for row in rs:
                job_planet_data.append(row)
            rs = curs.fetchall()
        for e in rs:
            temp = {'id': e[0], 'com_name': e[1], 'com_relation': e[2], 'mean_star': e[3], 'com_review': e[4],
                    'mean_sal': e[5],
                    'welfare_sal': e[6], 'wo_la_bal': e[7], 'com_cul': e[8], \
                    'opportunity': e[9], 'com_head': e[10], 'com_rec': e[11], \
                    'CEO_sup': e[12], 'growth_pos': e[13]}
            job_planet_data.append(temp)
    else:
        print("there is input")
        job_planet_data = rec_result[0]
        if len(rec_arr) != 0:
            rec_arr.pop(0)
    return render_template('recommendation.html', job_planet_data = job_planet_data, df = df, user_data = user_data)


if __name__ == '__main__':
    app.run(debug = True)
