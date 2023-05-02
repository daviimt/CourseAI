#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:10:16 2023

@author: manupc

Agrupación de datos en un conjunto de tipos de erupciones de geyser
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
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 2)
kmeans.fit(X_train)


# Validación
from sklearn import metrics
clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score
    ]
results= []
results += [m(y_train.reshape(-1), kmeans.labels_) for m in clustering_metrics]
print('homogeneity_score', results[0])
print('completeness_score', results[1])
print('v_measure_score', results[2])
print('adjusted_rand_score', results[3])


# Ilustración de los clusters generados
h = 0.02  # Puntos en la malla

# Creación de la malla
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtener clusters
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Mostrar en color
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(X_train[:, 0], X_train[:, 1], "k.", markersize=2)

# Mostrar centroides
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()