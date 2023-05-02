#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:37:01 2023

@author: manupc
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Cargamos los datos del geyser
data= pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv")

# Añadimos columna nueva igual a 0
data['ceros']= 0.0
print(data.head())
data= data.to_numpy()
X= data[:, [0, 1, 3]]
Y= data[:, 2]


le= LabelEncoder()
le.fit(Y)
Y= le.transform(Y)

print('Tamaño de los datos antes de selección: ', X.shape)

# Eliminación de datos con baja varianza
from sklearn.feature_selection import VarianceThreshold


var_thr = VarianceThreshold(threshold = 0.25)
var_thr.fit(X)
X2= var_thr.transform(X)
print('Tamaño de los datos tras de selección: ', X2.shape)



# Selección de k mejores atributos
from sklearn.feature_selection import SelectKBest, chi2

k_thr = SelectKBest(score_func=chi2, k=2)
k_thr.fit(X, Y)
X2= k_thr.transform(X)
print('Tamaño de los datos tras de selección: ', X2.shape)



# Eliminación iterativa de atributos
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier


rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=2)
rfe.fit(X, Y)
X2= rfe.transform(X)
print('Tamaño de los datos tras de selección: ', X2.shape)



# Análisis PCA

from sklearn.decomposition import PCA
pca= PCA(n_components=2)
pca.fit(X)
X2 = pca.transform(X)
print('Tamaño de los datos tras de selección: ', X2.shape)
print('PCA Explained variance: ', pca.explained_variance_ratio_)