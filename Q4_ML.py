# Error in dataSet no error but not proper
import pandas as pd
from sklearn.linear_model import LinearRegression


data = pd.read_csv("Placement.csv")
print(data)


X = data['cgpa'].values.reshape(-1, 1)
y = data['package'].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)


def predict_y(x_value):
    return lin_reg.predict([[x_value]])


intercept = lin_reg.intercept_
coefficient = lin_reg.coef_[0]
print("Intercept (a):", intercept)
print("Coefficient (b):", coefficient)

x_value = 5
predicted_y = predict_y(x_value)
print("Predicted y for x =", x_value, ":", predicted_y[0])