import numpy as np
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as pit

x=np.array([1,2,3,4,5,6]).reshape(-1,1)

y=np.array([1,2,3,4,5,6])

model=LogisticRegression()

model.fit(x,y)

# hours=[7]
for h in range (1,100):
    predicted_marks=model.predict([[h]])
    print("if studies",h,"hours, predicted marks:",predicted_marks[0])




      
pit.scatter(x,y)
pit.plot(x,model.predict(x))
pit.xlabel("study Hours")
pit.ylabel("marks")
pit.show()