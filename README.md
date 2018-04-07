# iris_python
iris dataset - reading data
# importing required packages
import numpy as np
import panda as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# reading file in kaggle
iris = pd.read_csv('../input/Iris.csv')
# Looking at data
iris.describe()
iris.type()
iris.info() # checking inconsistency in the data
iris.head() # looking top 5 rows
# Separating dependent and independent variables
X = iris.drop('Species',axis=1)
y = iris['Species']
# split data for crossvalidation
Xtrain,ytrain,Xtest,ytest = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# creating k-NN classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)
# classifier learning from training data
knn.fit(X_train,y_train)
# prediction on test data
knn.predict(X_test)
# comparing prediction with actual response
knn.score(X_test,y_test)


