import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.preprocessing import OneHotEncoder

data=pd.read_csv("PlayTennis.csv")

x=data.drop('play',axis=1)
y=data['play']
x=pd.get_dummies(x,drop_first=True)
# print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)
DT=DecisionTreeClassifier()
DT.fit(x_train,y_train)
y_pred=DT.predict(x_test)

print("------------------")     
acc=accuracy_score(y_test,y_pred)   
print(acc)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

print("------------------")
new_data = pd.DataFrame({'outlook': ['sunny'], 'temp': ['hot'], 'humidity': ['high'], 'windy': ['False']})
new_data = pd.get_dummies(new_data, drop_first=True)
missing_cols = set(x.columns)- set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0
new_data = new_data[x.columns]
predicted_play = DT.predict(new_data)
print("Predicted Play:", predicted_play[0])