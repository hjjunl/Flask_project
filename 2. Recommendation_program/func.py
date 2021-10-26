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

# IT 기업 추천 (Item based collaborative filtering)
def job_recomendation(mean_sal, mean_star, com_review_seg, welfare_sal, wo_la_bal, com_cul, opportunity, com_head,
                      growth_pos_seg, com_rec_seg, CEO_sup_seg):
    # 추천 할 값과 회사
    rec_com_list = []
    # 초기 상태
    if mean_sal == '' or mean_star == '' or welfare_sal == '' or com_review_seg == '':
        rec_com_list = [['']]
    else:
        conn = pymysql.connect(host = '127.0.0.1', user = 'root', db = 'testdb', passwd = '2000', charset = 'utf8')
        arr = []
        with conn.cursor() as curs:
            sql = "select * from job_planet"
            curs.execute(sql)
            rs = curs.fetchall()
            for row in rs:
                arr.append(row)

        df = pd.DataFrame(
            columns = ['id', 'com_name', 'com_relation', 'mean_star', 'com_review', 'mean_sal', 'welfare_sal', \
                       'wo_la_bal', 'com_cul', 'opportunity', 'com_head', 'com_rec', 'CEO_sup', \
                       'growth_pos'],
            data = arr)
        labels = np.arange(1, 6, 1)

        # 평균 연복은 크게 5개로 나눔 cut
        mean_sal_seg_series = pd.Series(list(pd.cut(df['mean_sal'], 5, labels = labels)),
                                        name = 'mean_sal_seg')  # 2800씩 등차함수

        # 나머지는 각 수에 맞춰 qcut
        com_rec_seg_series = pd.Series(list(pd.cut(df['com_rec'], 5, labels = labels)), name = 'com_rec_seg')
        CEO_sup_seg_series = pd.Series(list(pd.qcut(df['CEO_sup'], 5, labels = labels)), name = 'CEO_sup_seg')
        growth_pos_seg_series = pd.Series(list(pd.cut(df['growth_pos'], 5, labels = labels)), name = 'growth_pos_seg')

        # com_review 인지도로 나타낸다 (리뷰수)
        com_review_seg_series = pd.Series(list(pd.qcut(df['com_review'], 5, labels = labels)), name = 'com_review_seg')
        # 데이터 병합 1~5점으로 변환
        df = pd.concat([df, com_review_seg_series, growth_pos_seg_series, com_rec_seg_series, mean_sal_seg_series,
                        CEO_sup_seg_series], axis = 1)
        df.reset_index(drop = True, inplace = True)

        # 연봉 범위 정할 때
        if mean_sal >= 0 and mean_sal <= 1.5:
            df = df[(df['mean_sal_seg'] == 1)]
        if mean_sal <= 2.5 and mean_sal > 1.5:
            df = df[(df['mean_sal_seg'] == 2)]
        if mean_sal <= 3.5 and mean_sal > 2.5:
            df = df[(df['mean_sal_seg'] == 3)]
            print("check")
        if mean_sal <= 4.5 and mean_sal > 3.5:
            df = df[(df['mean_sal_seg'] == 4)]
        if mean_sal > 4.5:
            df = df[(df['mean_sal_seg'] == 5)]

        df.reset_index(drop = True, inplace = True)

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

        com_df = df.drop(['CEO_sup', 'com_rec', 'growth_pos', 'com_review', 'mean_sal_seg', 'com_relation'], axis = 1)
        com_df.reset_index(drop = True, inplace = True)
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
        return rec_com_list


# list 간 코사인 유사도 계산
def cos_sim(user, com_list_):
    # mean_sal, mean_star, com_review_seg, welfare_sal, wo_la_bal, com_cul, opportunity, com_head,
    #  growth_pos_seg, com_rec_seg, CEO_sup_seg
    com_id = []
    cos_sim_list = []
    com_name = []
    com_mean_star = []
    com_mean_sal = []
    for i in range(len(com_list_)):
        com_id.append(com_list_[i][0])
        com_mean_star.append(com_list_[i][2])
        # print(com_list_[ i ][ 1 ])
        com_mean_sal.append(com_list_[i][4])
        del com_list_[i][0]
        del com_list_[i][1]
        del com_list_[i][2]

    for i in com_list_:
        com_name.append(i[0])
        # del i[0]
    new_list = com_list_
    for i in range(len(com_list_)):
        com_list_[i].pop(0)
        # print('print com_list_', com_list_[ i ])
        # print('user', user)
        # print("len of user", len(user))
        # print("len of compare", len(com_list_[i]))
        try:
            # cos_sim_list.append(dot(user, com_list_[ i ]) / (norm(user) * norm(com_list_[ i ])))
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
    rec_dic = sorted(rec_dic.items(), key = lambda x: x[1][1], reverse = True)
    rec_dic = rec_dic[:10]

    return rec_dic


if __name__ == '__main__':
    # mean_sal, mean_star, welfare_sal, wo_la_bal, com_cul, opportunity, com_head,
    # com_review_seg, growth_pos_seg, com_rec_seg, CEO_sup_seg
    job_recomendation(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
