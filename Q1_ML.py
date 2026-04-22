#  For x = np.array([5, 15, 25, 35, 45, 55]) and y = np.array([5, 20, 14, 32, 22,
# 38]), apply simple linear regression using scikit learn library and calculate calculate R
# squared, coeficient and intercept. Predict the y values for x = np.arange(5). (Don’t
# split data for training/testing)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

x = np.array([5, 15, 25, 35, 45, 55]).reshape(-1,1)
print (x)
y = np.array([5, 20, 14, 32, 22, 38])
print (y)

model=LinearRegression()
model.fit(x,y)

r_squared=model.score(x,y)
coeff=model.coef_[0]
intercept=model.intercept_

x_pred=np.arange(5).reshape(-1,1)
print (x_pred)
y_pred=model.predict(x_pred)

print(f"R-squared: {r_squared}")
print(f"Cofficient: {coeff}")
print(f"Intercept: {intercept}")
print(f"predict y :{y_pred}")
