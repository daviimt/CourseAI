import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargamos los datos
col_names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
peoples =  pd.read_csv('E:\AI\Modulo 2\CourseAI\Module 3 (Machine Learning)\Reto\\train.csv')

# iris['label']= iris.Class.map({'Iris-setosa' : 0, 'Iris-versicolor':1,'Iris-virginica':2})
print(peoples.head())

# ¿Hay valores perdidos? Tratarlos aquí.

X= peoples[col_names].to_numpy()
y= peoples[['Survived']].to_numpy()

print('Tamaño del dataset: ', X.shape)
C= len(np.unique(y))
print('Número de clusters: ', C)

plt.scatter(X[:, 2], X[:, 3],X[:, 4],X[:,5], X[:, 6],X[:, 7],X[:, 8], X[:, 9],X[:, 10],
            color='blue', marker='+')

# División en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.25)
# ¿Es necesario hacer preprocesamiento? Hacerlo aquí
# from sklearn.preprocessing import StandardScaler
# se= StandardScaler()
# se.fit(X_train)
# X_train= se.transform(X_train)
# X_test= se.transform(X_test)

# Selección del algoritmo de aprendizaje
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.7, min_samples=7)
model.fit(X_train)
labels = model.labels_
# Cálculo de métricas de evaluación en TRAIN (Homogeneidad y completitud en nuestro caso)
print('Número de clusters encontrados: ', len(np.unique(labels)))
from sklearn import metrics
clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score
    ]
results= []
results += [m(y_train.reshape(-1), model.labels_) for m in clustering_metrics]
print('homogeneity_score', results[0])
print('completeness_score', results[1])
print('v_measure_score', results[2])
print('adjusted_rand_score', results[3])

# Cálculo de métricas de evaluación en TEST
plt.figure()
for x, y in zip(X_train, labels):
    if y==0:
        plt.plot(x[0], x[1],x[2], marker='x', color='red')
    elif y==1:
        plt.plot(x[0], x[1],x[2], marker='o', color='blue')
plt.show()