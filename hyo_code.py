import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

admission = pd.read_csv('./dataset/admission.csv')

admission('Wa')
admission.info()
admission.isnull().sum()
admisiion['State].nunique()
admission['State'].value_counts()
admission['Year'].value_counts()
admission.boxplot(column=['Value'])

# 데이터 시각화
sns.catplot(data=admission, y="Type", x="Value",
            col='State', kind='bar', col_wrap=3, sharex=False)

sns.set(font_scale = 2)
plt.figure(figsize=(20,100))
sns.barplot(data=admission, x='Value', y='State',order=a)

#모델 선택

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

feature = admission.drop('Value',axis=1)
drop_y = admission['Value']
encoding = pd.get_dummies(feature)
X = encoding
y = drop_y
X.shape,y.shape

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_test_predict = model.predict(X_test)
model.score(X_test,y_test)

rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train,y_train)
y_test_predict = model.predict(X_test)
model.score(X_test,y_test)