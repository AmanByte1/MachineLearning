import pandas as pd
import numpy as np
a = {
'area': [500, 750, 1000, 1200, 1500],
'price': [1500000, 2500000, 3000000, 3500000, 1800000]
}
df = pd.DataFrame(a)

df["category"]=np.where(df["price"]>3000000,"high",np.where(df["price"]<2000000,"low","medium"))
print(df)