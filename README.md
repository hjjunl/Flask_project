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

### 2. Job recommendation program (User based collaborative filtering)
(check https://github.com/hjjunl/DataProjects/blob/main/Data_Python/recommendation.py)

(How did I get the data? check https://github.com/hjjunl/DataProjects/blob/main/Data_Python/job_planet_info.py)
- Used cosine similarity and recommend the closest top 10 company
- Data is collected by crawling jobplanet.co.kr 2110 data
- data set: 평균 연봉, 평균 별점 선택, 복지 및 급여, 업무와 삶의 균형, 사내문화, 승진 기회 및 가능성, 경영진, 기업 조회: 기업 인지도, 성장 가능성, 기업 추천율, CEO 지지율
- mean salary, mean_star, com_review_seg, welfare_sal, work_life_balance, company_culture, promotion opportunity, company head, company growth posibility_seg, company_recommendation_seg, CEO_support_seg
- When you click the company name it shows you the company information like CEO, total revenue, home-page, company address, capital...etc from database.
- However if there is no company data you clicked it will crawl data from jobkorea.co.kr and save company data in the database and show it in website.

![image](https://user-images.githubusercontent.com/50603209/139201076-17658e3f-6375-4b5d-9d9c-6986229edc2b.png)
![image](https://user-images.githubusercontent.com/50603209/139201195-cbc7895f-063a-436f-9eee-b7cc172e30e1.png)
![image](https://user-images.githubusercontent.com/50603209/139201275-405787b1-baba-432f-bb1f-71b3fa2345ea.png)

