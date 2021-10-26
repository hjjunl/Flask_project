# Flask_project

### 1. Simple Employee management and analysis web (emp_program)
- CRUD function
- Excel uploads to application database (test.xlsx)
- Shows employees' payment chart using chart.js
- Predict new employees' salary (department, gender, position)
- simple random forest regressor model
- DB (mariadb)
- DB schema: department, payment_info, user_info, user_score

<index.html>
![image](https://user-images.githubusercontent.com/50603209/137441756-753b118c-d583-468b-88bd-a3d471802648.png)
<직원 예측>
![image](https://user-images.githubusercontent.com/50603209/137441825-b6ee45be-c097-4bfb-a5bc-fac7fdd615c2.png)

### 2. Job recommendation program 
(check https://github.com/hjjunl/DataProjects/blob/main/Data_Python/recommendation.py)
- Used cosine similarity and recommend the closest top 10 company
- Data is collected by crawling jobplanet.co.kr 2110 data
- data set: 연봉 범위 선택 2800 5600, 평균 별점 선택, 복지 및 급여, 업무와 삶의 균형, 사내문화, 승진 기회 및 가능성, 경영진, 기업 조회: 기업 인지도, 성장 가능성, 기업 추천율, CEO 지지율
- mean salary, mean_star, com_review_seg, welfare_sal, work_life_balance, company_culture, promotion opportunity, company head, company growth posibility_seg, company_recommendation_seg, CEO_support_seg
![image](https://user-images.githubusercontent.com/50603209/138800184-1635eb66-07af-4999-b6dd-827db6762e97.png)

![image](https://user-images.githubusercontent.com/50603209/138800242-acce6554-4aad-4f06-8e05-98c5ee57a332.png)
![image](https://user-images.githubusercontent.com/50603209/138800273-0d47acdf-b09c-46dc-a3a3-2739b6270d41.png)
