import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

x = np.arange(0, 30)
y = np.array([3, 4, 5, 7, 10, 8, 9, 10, 10, 23, 27, 44, 50, 63, 67, 60,
             62, 70,75, 88, 81, 87, 95, 100, 108, 135, 151, 160, 169, 179])


poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(x.reshape(-1,1))

model=LinearRegression()
model.fit(x_poly,y)


r_squared=model.score(x_poly,y)
coefficients=model.coef_
intercept=model.intercept_

print(f"R-squared: {r_squared}")
print(f"Coefficients: {coefficients}")  
print(f"Intercept: {intercept}")    

new_=np.arange(5)
x_new_poly=poly_reg.fit_transform(new_.reshape(-1,1))
y_pred=model.predict(x_new_poly)

print(f"Predicted values: {y_pred}")