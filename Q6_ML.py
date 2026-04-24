import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data=pd.read_csv("student_scores.csv")

MS=data.isnull().sum()
print(MS)

x=data[['Hours']]
print(x)
y=data['Scores']
print()
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
predicted_score=lr.predict(x_test)
print(predicted_score)
print(mean_squared_error(y_test,predicted_score))
print(lr.coef_)
print(lr.intercept_)