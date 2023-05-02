#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:37:01 2023

@author: manupc
"""

import numpy as np


# Cargamos los datos
datos= np.random.rand(100)
datos2= 2*datos+1
orig= np.vstack((datos, datos2)).T

# Ponemos valores perdidos en la segunda columna, un total de 10
p= np.random.permutation((len(orig)))[:10]
datos= orig.copy()
datos[p, 1]= np.nan


print('Datos originales:\n', datos)

# SimpleImpouter (Mean)
from sklearn.impute import SimpleImputer

si= SimpleImputer(strategy='mean')
si.fit(datos)
datos2= si.transform((datos))
print('Valores perdidos rellenos con media:\n', 
      np.hstack((datos[np.isnan(datos[:,1]), :], 
                 datos2[np.isnan(datos[:,1]), :])))


# SimpleImpouter (median)
from sklearn.impute import SimpleImputer

si= SimpleImputer(strategy='median')
si.fit(datos)
datos2= si.transform((datos))
print('Valores perdidos rellenos con mediana:\n', 
      np.hstack((datos[np.isnan(datos[:,1]), :], 
                 datos2[np.isnan(datos[:,1]), :])))

# SimpleImpouter (moda)
from sklearn.impute import SimpleImputer

si= SimpleImputer(strategy='most_frequent')
si.fit(datos)
datos2= si.transform((datos))
print('Valores perdidos rellenos con moda:\n', 
      np.hstack((datos[np.isnan(datos[:,1]), :], 
                 datos2[np.isnan(datos[:,1]), :])))


# SimpleImpouter (valor constante)
from sklearn.impute import SimpleImputer

si= SimpleImputer(strategy='constant', fill_value=0.0)
si.fit(datos)
datos2= si.transform((datos))
print('Valores perdidos rellenos con mediana:\n', 
      np.hstack((datos[np.isnan(datos[:,1]), :], 
                 datos2[np.isnan(datos[:,1]), :])))


# KnnImputer (valor de los 3 más parecidos)
from sklearn.impute import KNNImputer

knni= KNNImputer(n_neighbors=3)
knni.fit(datos)
datos2= knni.transform((datos))
print('Valores perdidos rellenos con KNN:\n', 
      np.hstack((datos[np.isnan(datos[:,1]), :], 
                 datos2[np.isnan(datos[:,1]), :])))


