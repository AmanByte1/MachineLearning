import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv("real_estate.csv")

x=df[['size','year']]
y=df['price']

model=LinearRegression()
model.fit(x,y)


r_squared=model.score(x,y)
cofficients=model.coef_
intercept=model.intercept_

prediction=model.predict([[750,2009]])

print(f"R-squrared:{r_squared}")
print(f"Cofficients:{cofficients}")
print(f"intercept: {intercept}")

print(f"Predicted price for a 750 sq.ft. apartment in 2009: {prediction[0]}")
