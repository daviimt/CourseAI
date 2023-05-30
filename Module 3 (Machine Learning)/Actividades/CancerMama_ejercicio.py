#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:03:12 2023

Se trata de un conjunto de pacientes (mujeres) a las que se le detecta cáncer de mama.
El problema consiste en clasificar si es benigno o maligno conociendo mediciones 
médicas realizadas como el radio del tumor, su textura, perímetro, área, concavidad, etc.


@author: manupc
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# Cargamos el dataset de cáncer de mama y vemos sus atributos
dataset= load_breast_cancer()
for key in dataset:
    print(key)

# Mostramos descripción (Descomentar para ver)
#print(dataset['DESCR'])

# Convertimos a pandas juntando entradas (data) y salidas (target)
df= pd.DataFrame(data= dataset['data'], columns= dataset['feature_names'])
df['target']= pd.Series(dataset['target'])

# Muestra de los datos
print('Muestra de los datos:')
print(df.sample(10))

# Descripción de los datos
print('Descripción de los datos:')
print(df.describe())

# Información de los datos
print('Información de los datos:')
print(df.info())

# Número de clases:
print('Número de clases a predecir: ', len(df.target.unique()), 'que son:', df.target.unique())



# ¿Es necesario quitar valores perdidos? Si es el caso, hacerlo aquí:
# NO
    
# ¿Se debería hacer alguna transformación de los datos? Si es es caso, hacerlo aquí:
from sklearn.preprocessing import MinMaxScaler

#scaler=MinMaxScaler()
#scaler.fit(X)
#X=scaler.transform(X)

# División ALEATORIA en training (50%) y test (50%) . Ejemplo:

datos= df.to_numpy()
X= datos[:, :-1] # Entradas: Todas las columnas salvo la última
Y= datos[:, -1].astype(int) # Salidas: La última columna

# División en train y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, shuffle=True)




    
# ¿Qué modelo se va a usar para aprender? Instanciarlo aquí y entrenarlo en TRAIN
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train,Y_train)

model.fit(X_train, Y_train)

# Comprobación del Score (accuracy) en training y en test
acuracy_train=model.score(X_train,Y_train)
acuracy_test=model.score(X_test,Y_test)

print('train',acuracy_train*100)
print('train',acuracy_test*100)