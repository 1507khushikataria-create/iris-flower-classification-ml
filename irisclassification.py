import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

iris=load_iris()

data=pd.DataFrame(iris.data,columns=iris.feature_names)
data['SPECIES']=iris.target

print(data.head())

plt.scatter(data['sepal length (cm)'],data['sepal width (cm)'])
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('sepal length vs  sepal width')
plt.show()

plt.scatter(data['petal length (cm)'],data['petal width (cm)'])
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('petal length vs  petal width')
plt.show()

sns.pairplot(data,hue='SPECIES')
plt.show()

X=data.drop('SPECIES',axis=1)
y=data['SPECIES']

X_train,X_test,Y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model =KNeighborsClassifier()
model.fit(X_train,Y_train)

predictions=model.predict(X_test)

score=accuracy_score(y_test,predictions)
print(" Accuracy score:",score)

report=classification_report(y_test,predictions)
print("classification report:",report)

matrix=confusion_matrix(y_test,predictions)
print("confusion matrix:",matrix)

