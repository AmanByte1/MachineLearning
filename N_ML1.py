import pandas as pd
data={
    'area':[500,750,1000,1200,1500],
    'price': [1500000, 2500000, 3000000, 3500000, 1800000]
}
df =pd.DataFrame(data)
def categorize_price(price):
    if price>3000000:
        return 'High'
    elif price<2000000:
        return 'Low'
    else:
        return 'Medium'
df['category']=df['price'].apply(categorize_price)
print(df)
