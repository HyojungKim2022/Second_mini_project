미국 주별 대학 기숙사 학비 및 기숙사비 예측

- State = 주
- Type = 공립(주 안, 주 밖)/사립
- Length = 2 / 4 year
- Expense = Fees/Tuition // Room/Board
- Value = 값
--------------------------------------------------
- 문제 유형:수치예측
- 성능 평가지표: RMSE


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

education = pd.read_csv('./dataset/admission.csv')

education('Wa')
education.info()
education.isnull().sum()
education['State].nunique()
education['State'].value_counts()
education['Year'].value_counts()
education.boxplot(column=['Value'])

# 데이터 시각화
sns.catplot(data=education, y="Type", x="Value",
            col='State', kind='bar', col_wrap=3, sharex=False)

sns.set(font_scale = 2)
plt.figure(figsize=(20,100))
sns.barplot(data=education, x='Value', y='State',order=a)

edu = (education['Expense'] == 'Fees/Tuition') 
a=education[edu]
b=a.groupby(['State']).mean()
c=b.sort_values(by=["Value"], ascending=[False]).head(10).reset_index()
plt.figure(figsize=(5, 5))
sns.barplot(y=c.State, x=c.Value)

edu = (education['Expense'] == 'Room/Board') 
a=education[edu]
b=a.groupby(['State']).mean()
c=b.sort_values(by=["Value"], ascending=[False]).head(10).reset_index()
plt.figure(figsize=(5, 5))
sns.barplot(y=c.State, x=c.Value)

labels = ["Type","Length"]
fig = px.parallel_categories(
    education[labels],
    template= 'plotly_dark',
    title = "Type and length Diagram",
    )
fig.update_layout(
    font = dict(size=17,family="Franklin Gothic")
)
fig.show()

#모델 선택

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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

# feature = education.drop('Value',axis=1)
# drop_y = education['Value']
# encoding = pd.get_dummies(feature)
# X = encoding
# y = drop_y
# X.shape,y.shape

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_test_predict = model.predict(X_test)
# model.score(X_test,y_test)

# rf_reg = RandomForestRegressor(random_state=42)
# rf_reg.fit(X_train,y_train)
# y_test_predict = model.predict(X_test)
# model.score(X_test,y_test)

df = education
df['Length'] = df['Length'].str[0]
df['Length'] = df['Length'].astype(int)

df= pd.get_dummies(df, prefix='Expense', columns=['Expense'], drop_first=False)
df= pd.get_dummies(df, prefix='Type', columns=['Type'], drop_first=False)
df= pd.get_dummies(df, prefix='State', columns=['State'], drop_first=False)

X=df.drop("Value", axis=1)
y=df["Value"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# LinearRegression

lin_reg = LinearRegression()
lin_scores = cross_val_score(lin_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lin_rmse = np.sqrt(-lin_scores.mean())
lin_rmse

# Ridge

ri_reg = Ridge()
ri_scores = cross_val_score(ri_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
ri_rmse = np.sqrt(-ri_scores.mean())
ri_rmse

# DecisionTreeRegressor()

de_reg = DecisionTreeRegressor()
de_scores = cross_val_score(de_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
de_rmse = np.sqrt(-de_scores.mean())
de_rmse

# RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_scores = cross_val_score(rf_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rf_rmse = np.sqrt(-rf_scores.mean())
rf_rmse

## Tuning
# Grid Search

param_grid = {'n_estimators' : [30, 50, 100], 'max_features' : [2, 4, 6, 8]} 

rf_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(rf_reg, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1) 
%time grid_search.fit(X_train, y_train)

grid_search.best_params_

grid_search.best_estimator_

cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
  print(np.sqrt(-mean_score), params)


# 모델 예측과 성능 평가

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

