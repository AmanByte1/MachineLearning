import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

wine=load_wine()
x=wine.data
y=wine.target
print(wine)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

dT=DecisionTreeClassifier(criterion='entropy',random_state=42)

dT.fit(x_train,y_train)
y_pred=dT.predict(x_test)

print("------------------")
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import classification_report, accuracy_score
# Calculate and print the classification report for the true labels and predicted labels
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

print("------------------")
acc_score = accuracy_score(y_test, y_pred)
print('Accuracy Score:')
print(acc_score)