#시각화
# 연도별 총 학비 추이
plt.scatter(education.Year,education.Value)
plt.show()

#공립, 사립, 주립별 가격 추이  /사립이 비싸고 주립이저렴/주내와 주외의 학생 가격도 다름
sns.displot(data=education,kde=True,x='Value',hue='Type',palette='Set1',)
plt.show()


#지도로 표현
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

df_copy = education.copy()

df_copy.rename(columns={'Expense': 'Expense_type','Value':'Expense'}, inplace=True)

us_state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
    'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',"District of Columbia":"DC"
}
df_copy['State_abbr'] = df_copy['State'].apply(lambda x: us_state_abbrev[x])

import plotly.express as px
exp_type = list(df_copy["Expense_type"].value_counts().index)[0]
fig = px.choropleth(
    df_copy[df_copy["Expense_type"] == exp_type], 
    locations='State_abbr', 
    locationmode='USA-states',
    color='Expense', 
    scope='usa',
    color_continuous_scale="PuBu",
    template = "plotly_dark",
    title = f"Map of Expense ({exp_type}) per state",
)
fig.show()


# 주별 학비 탑 10

edu = (education['Expense'] == 'Fees/Tuition') 
a=education[edu]
b=a.groupby(['State']).mean()
c=b.sort_values(by=["Value"], ascending=[False]).head(10).reset_index()
plt.figure(figsize=(5, 5))
sns.barplot(y=c.State, x=c.Value)



# 인코딩
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

education['Length'] = education['Length'].str[0]
education['Length'] = education['Length'].astype(int)

education = pd.read_csv('./dataset/education.csv')


features=education.drop("Value", axis=1)
y=education["Value"]

one_hot_features = pd.get_dummies(features)

one_hot_features


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(one_hot_features, y, test_size=0.2, random_state=42)

#머신러닝 모델훈련
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor(random_state=42)
rf_reg = RandomForestRegressor(random_state=42)
svr_reg=SVR(kernel='rbf',gamma='scale')
knn_reg=KNeighborsRegressor(n_neighbors=7,weights="distance")
rl_reg= Ridge()

#선형
lin_scores = cross_val_score(lin_reg,X_train, y_train, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
lin_rmse = np.sqrt(-lin_scores.mean())
lin_rmse


#트리
tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
tree_rmse = np.sqrt(-tree_scores.mean())
tree_rmse


#랜덤포레스트(배깅 앙상블)
rf_scores = cross_val_score(rf_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
rf_rmse = np.sqrt(-rf_scores.mean())
rf_rmse


#SVM(svr)

svr_scores=rf_scores = cross_val_score(svr_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
svr_rmse = np.sqrt(-svr_scores.mean())
svr_rmse


#knn

knn_scores = cross_val_score(knn_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
knn_rmse = np.sqrt(-knn_scores.mean())
knn_rmse




#튜닝

from sklearn.model_selection import GridSearchCV

rf_reg = RandomForestRegressor(random_state=42)

param_grid = {'n_estimators' : [30, 50, 100], 'max_features' : [2, 4, 6, 8]} # 3 * 4 = 12가지 조합의 파라미터로 설정된 모델 준비

grid_search = GridSearchCV(rf_reg, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1) # 3 * 4 * 5 = 60번의 학습과 검증
%time grid_search.fit(X_train, y_train)

grid_search.best_params_

cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(np.sqrt(-mean_score), params)


grid_search.best_estimator_


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'n_estimators' : randint(low=1, high=200),
                  'max_features' : randint(low=1, high=8)}

rnd_search = RandomizedSearchCV(rf_reg, param_distribs, n_iter=10, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)                  
%time rnd_search.fit(X_train, y_train)

rnd_search.best_estimator_

cv_results = rnd_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(np.sqrt(-mean_score), params)


best_model = grid_search.best_estimator_

X_test.shape, y_test.shape

from sklearn.metrics import mean_squared_error

final_predictions = best_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_predictions, squared=False) # RMSE
final_rmse


from scipy.stats import t

# 추정량 (오차의 제곱들의 합)
squared_erros = (final_predictions - y_test)**2

# 95% 신뢰구간
confidence = 0.95

# 표본의 크기
n = len(squared_erros)

# 자유도 (degree of freedom)
dof = n-1

# 추정량의 평균
m_squared_error = np.mean(squared_erros)

# 표본의 표준편차 (비편상 분산으로 구함)
sample_std = np.std(squared_erros, ddof=1) # n-1로 나눔 (그림에서 U)

# 표준 오차
std_err = sample_std/n**0.5 # (그림에서 U/n**0.5)

mse_ci = t.interval(confidence, dof, mean_squared_error, std_err)
rmse_ci = np.sqrt(mse_ci)
rmse_ci


