from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris=load_iris()
x=iris.data
y=iris.target
# print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)

KNN=KNeighborsClassifier(n_neighbors=3)

KNN.fit(x_train,y_train)
y_pred=KNN.predict(x_test)

acc=accuracy_score(y_test,y_pred)
print(acc)
