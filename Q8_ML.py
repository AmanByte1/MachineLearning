import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv("winequalityN.csv")
print(df)
df.fillna(df.mean(), inplace=True)

x=df.drop(columns=["quality"])
y=df["quality"]
# y1=df.iloc[:, -1]
# print(y1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)

coefficients=model.coef_
intercept=model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:", mse)   

new_data=[[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]
predicted_quality=model.predict(new_data)
print("Predicted Quality:", predicted_quality)

