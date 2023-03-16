## Average cost of undergraduate student by state USA 
## 미국 주별 학부생 평균 비용
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

education = pd.read_csv('./dataset/education.csv')

- target :  value 수치 예측 문제

```
Year 연도

State 주

Type 대학 유형

Length 종류

Expense 생활비용 Fees/Tuition:수업료/수수료, Room/Board:숙식제공하는 기숙사 

Value 값 평균 비용

```
education

education.info()

# 수치형 특성 탐색
education.describe()

#null값 확인
education.isnull().sum()

education.columns

# 연도별 총 평균 학비 추이
plt.scatter(education.Year,education.Value)
plt.show()

education.hist(bins=50, figsize=(10, 10))
plt.show()

df_1 = education.groupby(by=['State']).mean().drop('Year',axis=1)
plot_order = df_1.sort_values(by='Value',ascending=False).index.values

# 사립 공립 별 가격 추이 /공립이 저렴하고 사립이 비싸다
# 주내와 주외의 학생 가격도 다름
sns.displot(data=education, kde=True, x='Value', hue='Type',palette='Set2')
plt.show()
#사립 또는 공립, 주 내 또는 주 외부.

df_1 = education.groupby(by=['State']).mean().drop('Year',axis=1)
plot_order = df_1.sort_values(by='Value',ascending=False).index.values

sns.set(font_scale = 2)
plt.figure(figsize=(20,100))
sns.barplot(data=education, x='Value', y='State',order=plot_order)

education['State'].value_counts()

#Room/Board인 주 별 비용 값
play=education.groupby(['State','Expense']).mean()

e = education['Expense'] == 'Room/Board'

a = education[e]

plt.figure(figsize=(10, 10))
sns.barplot( x =a.Value, y = a.State )

a = education[e]
a

play=education.groupby(['State','Expense']).mean()

edu = (education['Expense'] == 'Fees/Tuition') & (education['Length'] == '4-year') # 생활비 : 수강료/수업료 4년제
edu2 = (education['Expense'] == 'Fees/Tuition') & (education['Length'] == '2-year') #생활비 : 수강료/수업료 2년제

a = education[edu]
b = education[edu2]

plt.figure(figsize=(20, 10))

sns.barplot(y=b.State, x=b.Value)

cat_columns = ['Year','Type','Length','Expense']
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 4))
for i, column in enumerate(cat_columns):
    sns.barplot(data=education, x=column, y='Value', ax=axes[i])
plt.show()

education

## 데이터 전처리
X = education.drop('Value', axis=1)
y = education['Value']

X.shape ,y.shape
X,y

# 원핫 인코딩
one_hot_features = pd.get_dummies(data=education,columns=['State','Type','Length','Expense'])

one_hot_features

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 데이터 전처리
## 1.OneHotEncoding
- LinearRegression
- Ridge
- DecisionTreeRegressor
- RandomForestRegressor

## 2.LabelEncoding
- LinearRegression
- Ridge
- DecisionTreeRegressor
- RandomForestRegressor

## 3.Grid Search
## 4.Random Search

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

### OneHotEncoding with pandas

df = education #함수를 새로 만들어서 저장
df['Length'] = df['Length'].str[0]
df['Length'] = df['Length'].astype(int)

df= pd.get_dummies(df, prefix='Expense', columns=['Expense'], drop_first=False)
df= pd.get_dummies(df, prefix='Type', columns=['Type'], drop_first=False)
df= pd.get_dummies(df, prefix='State', columns=['State'], drop_first=False)

X=df.drop("Value", axis=1)
y=df["Value"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_scores = cross_val_score(lin_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lin_rmse = np.sqrt(-lin_scores.mean())
lin_rmse

ri_reg = Ridge()
ri_scores = cross_val_score(ri_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
ri_rmse = np.sqrt(-ri_scores.mean())
ri_rmse

de_reg = DecisionTreeRegressor()
de_scores = cross_val_score(de_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
de_rmse = np.sqrt(-de_scores.mean())
de_rmse

rf_reg = RandomForestRegressor()
rf_scores = cross_val_score(rf_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rf_rmse = np.sqrt(-rf_scores.mean())
rf_rmse

## Labelencoding with sklearn
lb_df = education
col = lb_df.select_dtypes(include = 'object').columns
model = LabelEncoder()
le_col = lb_df[col].apply(model.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(le_col,y,test_size=0.2, random_state=42)

lb_lin_reg = LinearRegression()
lb_lin_scores = cross_val_score(lb_lin_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lb_lin_rmse = np.sqrt(-lb_lin_scores.mean())
lb_lin_rmse

lb_ri_reg = Ridge()
lb_ri_scores = cross_val_score(lb_ri_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lb_ri_rmse = np.sqrt(-lb_ri_scores.mean())
lb_ri_rmse

lb_de_reg = DecisionTreeRegressor()
lb_de_scores = cross_val_score(lb_de_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lb_de_rmse = np.sqrt(-lb_de_scores.mean())
lb_de_rmse

lb_rf_reg = RandomForestRegressor(random_state=42)
lb_rf_scores = cross_val_score(lb_rf_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lb_rf_rmse = np.sqrt(-lb_rf_scores.mean())
lb_rf_rmse

# Tuning
- Grid Search
param_grid = {'n_estimators' : [30, 50, 100], 'max_features' : [2, 4, 6, 8]} 

rf_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(rf_reg, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1) 
%time grid_search.fit(X_train, y_train)

grid_search.best_params_

grid_search.best_estimator_
cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
  print(np.sqrt(-mean_score), params)

- Random Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'n_estimators' : randint(low=1, high=200),
                  'max_features' : randint(low=1, high=8)}

rnd_search = RandomizedSearchCV(rf_reg, param_distribs, n_iter=10, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)                  
%time rnd_search.fit(X_train, y_train)

rnd_search.best_params_

rnd_search.best_estimator_

cv_results = rnd_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
  print(np.sqrt(-mean_score), params)

best_model = grid_search.best_estimator_

8. 모델 예측과 성능 평가

X_test.shape, y_test.shape

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
mean_squared_error = np.mean(squared_erros)

# 표본의 표준편차 (비편상 분산으로 구함)
sample_std = np.std(squared_erros, ddof=1) # n-1로 나눔 (그림에서 U)

# 표준 오차
std_err = sample_std/n**0.5 # (그림에서 U/n**0.5)

mse_ci = t.interval(confidence, dof, mean_squared_error, std_err)
rmse_ci = np.sqrt(mse_ci)
rmse_ci

fig, ax = plt.subplots()
ax.scatter(y_test, final_predictions, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()
