#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:09:26 2023

@author: manupc
Dataset: Diabetes

Hay 10 atributos de entrada (edad, sexo, índice de masa corporal,
presión sanguínea promedia), 6 variables de análisis de sangre) sobre
442 pacientes de diabetes. Se trata de predecir la evolución de la
diabetes de forma cuantitativa.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carga del dataset y pasarlo a DataFrame
dataset= load_diabetes()


df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['Progression'] = dataset.target # Insertamos también el objetivo 
print(df.head())

print('Valores perdidos: ', df.isna().sum()) # No hay
print('Descripción del dataset:\n', df.describe())
    
# Matriz de correlación entre variables
corr = df.corr()

plt.subplots(figsize=(8,8))
sns.heatmap(corr,cmap= 'RdYlGn',annot=True)
plt.show()


# Separamos en X/y

#independent variables / explanatory variables
X = df.drop(labels='Progression', axis=1).to_numpy()  #axis=1 quitamos columnas
y = df['Progression'].to_numpy() # target


# Creamos train/test
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25)



##############################################################
# Modelo de regresión lineal
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# Predicción
train_p= model.predict(x_train)
test_p= model.predict(x_test)

# Gráfica de resultados
plt.figure()
plt.scatter(y_train.reshape(-1), train_p.reshape(-1), color='blue')
plt.scatter(y_test.reshape(-1), test_p.reshape(-1), color='red')
x= np.linspace(np.min(y_train), np.max(y_train), 1000)
plt.plot(x,x)

# Métricas de resultados
from sklearn import metrics as mt
print("El modelo lineal explica ", np.round(mt.explained_variance_score(y_test,test_p)*100,2),"% de la varianza del objetivo con respecto a los atributos")
print("El error promedio absoluto del modelo lineal es: ", np.round(mt.mean_absolute_error(y_test, test_p),2))
print("El coeficiente de determinación R^2 del modelo lineal es: " , np.round(mt.r2_score(y_test,test_p),2))
print("El error cuadrático medio del modelo lineal es " , np.round(mt.mean_squared_error(y_test, test_p) ,2))




##############################################################
# Modelo de MLP
from sklearn.neural_network import MLPRegressor
model = MLPRegressor([100, 100], activation='relu', solver='lbfgs', 
                     max_iter=20000)
model.fit(x_train, y_train)

# Predicción
train_p= model.predict(x_train)
test_p= model.predict(x_test)

# Gráfica de resultados
plt.figure()
plt.scatter(y_train.reshape(-1), train_p.reshape(-1), color='blue')
plt.scatter(y_test.reshape(-1), test_p.reshape(-1), color='red')
x= np.linspace(np.min(y_train), np.max(y_train), 1000)
plt.plot(x,x)

# Métricas de resultados
print("El modelo MLP explica ", np.round(mt.explained_variance_score(y_test,test_p)*100,2),"% de la varianza del objetivo con respecto a los atributos")
print("El error promedio absoluto del modelo MLP es: ", np.round(mt.mean_absolute_error(y_test, test_p),2))
print("El coeficiente de determinación R^2 del modelo MLP es: " , np.round(mt.r2_score(y_test,test_p),2))
print("El error cuadrático medio del modelo MLP es " , np.round(mt.mean_squared_error(y_test, test_p) ,2))

