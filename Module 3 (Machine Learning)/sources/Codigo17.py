#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:10:16 2023

@author: manupc

Agrupación de datos en un conjunto de tipos de erupciones de geyses
(usamos 2 variables sólo por motivos de visualización posterior)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargamos los datos
data= pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv")
data['label']= data.kind.map({'long' : 0, 'short':1})
print(data.head())

# Cogemos datos
X= data[['duration', 'waiting']].to_numpy()
y= data[['label']].to_numpy()

print('Tamaño del dataset: ', X.shape)
C= len(np.unique(y))
print('Número de clusters: ', C)

# Mostrar puntos
## Límites de la figura
plt.scatter(X[:, 0], X[:, 1],
            color='blue', marker='+')



# División en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.25)


# Preprocesamiento
from sklearn.preprocessing import StandardScaler
se= StandardScaler()
se.fit(X_train)
X_train= se.transform(X_train)
X_test= se.transform(X_test)


# Entrenamiento
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.3, min_samples=10)
model.fit(X_train)
labels = model.labels_

# Número de clusters
print('Número de clusters encontrados: ', len(np.unique(labels)))


# Validación
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


# Ilustración de los clusters generados
plt.figure()
for x, y in zip(X_train, labels):
    if y==0:
        plt.plot(x[0], x[1], marker='o', color='red')
    elif y==1:
        plt.plot(x[0], x[1], marker='o', color='green')
    else:
        plt.plot(x[0], x[1], marker='x', color='blue')
plt.show()