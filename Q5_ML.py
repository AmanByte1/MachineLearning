import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

d={"x1":[60,62,76,70,72,72,75,78],
   "y":[140,155,159,179,192,200,212,215]}

df=pd.DataFrame(d)
x=df[["x1"]]
y=df["y"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=4)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
m=lr.coef_
i=lr.intercept_
print(i)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))

plt.plot(x_test,y_test)
plt.plot(x_train,y_train)
plt.plot(x_test,y_pred)
# plt.plot(x_test,y_train)

plt.show()
