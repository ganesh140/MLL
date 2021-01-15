from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np 
from sklearn.model_selection import train_test_split 
iris_dataset=load_iris() 
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0) 
kn = KNeighborsClassifier() 
kn.fit(X_train, y_train) 
prediction = kn.predict(X_test)
print("ACCURACY:"+ str(kn.score(X_test, y_test)))
target_names = iris_dataset.target_names
for pred,actual in zip(prediction,y_test):
    print("Prediction is "+str(target_names[pred])+", Actual is "+str(target_names[actual]))
