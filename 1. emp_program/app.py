import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask.templating import render_template
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
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
                new_df = func.select_all()
                checking = 0
                for i in list(data['BADGE']):
                    if i not in list(new_df['BADGE']):
                        checking == 0
                    else:
                        checking = 1
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


if __name__ == '__main__':
    app.run(debug = True)
