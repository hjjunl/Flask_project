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
app = Flask(__name__, static_url_path = "", static_folder = 'static',
            template_folder = 'templates')
Bootstrap(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:2000@localhost:3306/testdb"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


@app.route('/ins.ajax', methods = ['POST'])
def ins_ajax():
    data = request.get_json()
    BADGE = data['BADGE']
    name = data['name']
    department = data['department']
    gender = data['gender']
    position = data['position']
    cnt = func.MyEmpDao().insEmp(BADGE, name, department, gender, position)
    result = "success" if cnt == 1 else "fail"
    return jsonify(result = result)


@app.route('/mod.ajax', methods = ['POST'])
def mod_ajax():
    data = request.get_json()
    BADGE = data['BADGE']
    name = data['name']
    department = data['department']
    join_date = data['join_date']
    gender = data['gender']
    position = data['position']
    cnt = func.MyEmpDao().updEmp(name, department, join_date, gender, position, BADGE)
    result = "success" if cnt == 1 else "fail"
    return jsonify(result = result)


@app.route('/del.ajax', methods = ['POST'])
def del_ajax():
    data = request.get_json()
    BADGE = data['BADGE']
    print(BADGE)
    cnt = func.MyEmpDao().delEmp(BADGE)
    result = "success" if cnt == 1 else "fail"
    return jsonify(result = result)


##################
@app.route('/mod1.ajax', methods = ['POST'])
def mod1_ajax():
    data = request.get_json()
    BADGE = data['BADGE']
    name = data['name']
    department = data['department']
    position = data['position']
    cnt = func.MyEmpDao().updEmp1(name, department, position, BADGE)
    result = "success" if cnt == 1 else "fail"
    return jsonify(result = result)


@app.route('/ins1.ajax', methods = ['POST'])
def ins1_ajax():
    data = request.get_json()
    BADGE = data['BADGE']
    name = data['name']
    department = data['department']
    gender = data['gender']
    position = data['position']
    cnt = func.MyEmpDao().insEmp(BADGE, name, department, gender, position)
    result = "success" if cnt == 1 else "fail"
    return jsonify(result = result)


@app.route("/db_list", methods = ['POST', 'GET'])
def db_list():
    check = '파일 확인'
    if request.method == 'POST':
        if len(request.form['upload_file']) == 0:
            check = 'Select Excel file first and press this!'
            pass
        else:
            if len(request.form['upload_file']) != 0:
                file = request.form['upload_file']
                data = pd.read_excel(file)
                checking = 0

                if checking == 0:
                    func.insert_excel_to_db(data)
                else:
                    check = 'There is same PK check your excel file!'
                    pass

    empList = func.MyEmpDao().getEmps();

    return render_template('db_list.html', empList = empList, check = check)


empList = []
#################
k = 0
cnt = 0
arr = []
send = []


#############
@app.route('/send_BADGE.ajax', methods = ['POST', 'GET'])
def send_BADGE():
    data = request.get_json()
    BADGE = data['BADGE']
    cnt1 = func.personal_data(BADGE)
    send.append(cnt1)
    result = "success" if cnt1 == 1 else "fail"
    return jsonify(result = result)


@app.route('/select.ajax', methods = ['POST'])
def select_ajax():
    data = request.get_json()
    if data != None:
        BADGE = data['BADGE']
        name = data['name']
        department = data['department']
        cnt = func.select_emp(BADGE, name, department)
        arr.append(cnt)
    result = "success" if cnt == 1 else "fail"
    return jsonify(result = result), cnt


# 상세 화면
@app.route('/individual.html', methods = ['Get', 'POST'])
def individual():
    if len(send) != 0:
        send_data = send[0]
        send.pop()
        print(send_data)
    else:
        send_data = []
    return render_template('individual.html', send_data1 = send_data)


# 메인 화면
@app.route("/", methods = ['GET', 'POST'])
def index():
    user_info_ = user_info.query.all()
    emp_analysis = []
    if len(arr) != 0:
        empList = arr[0]
        arr.pop()
        print(empList)
        name = list(empList[1]['name'])
        payment = list(empList[1]['payment'].astype(int))
    else:
        empList = func.select_emp(BADGE = '', name = '', department = '');
        name = list(empList[1]['name'])
        payment = list(empList[1]['payment'].astype(int))
        df = empList[1]
        emp_analysis = func.emp_pre_ex(df)
        print(empList[0])
    return render_template('index.html', empList = empList[0], name = name, payment = payment, user_info = user_info_,
                           emp_analysis = emp_analysis)


# 예측 화면
@app.route('/prediction', methods = ['Get', 'POST'])
def prediction():
    empList = func.select_emp(BADGE = '', name = '', department = '')
    value = list(empList[1]['payment'].astype(int))
    name = list(empList[1]['name'])
    data_list = empList[0]
    if request.method == 'POST' and len(request.form['upload_file_prediction']) != 0:
        if len(request.form['upload_file_prediction']) != 0:
            file = request.form['upload_file_prediction']
            data = pd.read_excel(file)
            prediction = func.emp_prediction(data)
            value = prediction[0]
            name = prediction[1]
            data_list = prediction[2]

    return render_template('prediction.html', prediction_value = value, name = name, data_list = data_list)


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
