#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:37:01 2023

@author: manupc
"""

import numpy as np

# Cargamos los datos
datos= np.random.rand(100)
print('Datos originales, media y std: ', np.mean(datos), np.std(datos))

# StandardScaler
from sklearn.preprocessing import StandardScaler

ss= StandardScaler()
ss.fit(datos.reshape(-1, 1))

datos2= ss.transform(datos.reshape(-1, 1))
print('Datos tras StandardScaler, media y std: ', np.mean(datos2), np.std(datos2))

# MaxMinScaler
from sklearn.preprocessing import MinMaxScaler

mms= MinMaxScaler(feature_range=(1, 10))
mms.fit(datos2)

datos3= mms.transform(datos2)
print('Datos tras MinMaxScaler, media y std: ', np.mean(datos3), np.std(datos3))

# Transformación logarítmica
from sklearn.preprocessing import FunctionTransformer

def logTransform(x):
    return np.log(x)

logt= FunctionTransformer(logTransform)
logt.fit(datos3)

datos4= logt.transform(datos3)
print('Datos tras Log transform, media y std: ', np.mean(datos4), np.std(datos4))
