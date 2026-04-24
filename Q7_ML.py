import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


data = pd.read_csv('insurance.csv')
print(data.head())



data = pd.get_dummies(data, drop_first=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
coefficients = model.coef_
intercept = model.intercept_
mse = mean_squared_error(y_test, y_pred)

print("Coefficients:", coefficients)
print("Intercept:", intercept)
print("Mean Squared Error:", mse)
print("Predicted Values:", y_pred[:10])
print("r2_score",r2_score(y_test,y_pred))