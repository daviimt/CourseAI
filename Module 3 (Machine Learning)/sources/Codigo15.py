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


# Separamos en X/y

#independent variables / explanatory variables
X = df.drop(labels='Progression', axis=1).to_numpy()  #axis=1 quitamos columnas
y = df['Progression'].to_numpy() # target


# Creamos train/test
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25)



##############################################################
# Modelo de regresión lineal SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

# Muestra de ExplainedVariance para Linear Regression
muestraLR= []
numMuestras= 30
for i in range(numMuestras): # 30 veces mínimo, estadísticamente significativo
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    test_p= model.predict(x_test)
    muestraLR.append(explained_variance_score(y_test.reshape(-1), test_p.reshape(-1)))



##############################################################
# Modelo de MLP
from sklearn.neural_network import MLPRegressor

# Muestra de MSE para MLP
muestraMLP= []
for i in range(numMuestras): # 30 veces mínimo, estadísticamente significativo
    model = MLPRegressor([100, 100], activation='relu', solver='lbfgs', 
                     max_iter=2000, random_state=1000+i)
    model.fit(x_train, y_train)

    test_p= model.predict(x_test)
    muestraMLP.append(explained_variance_score(y_test.reshape(-1), test_p.reshape(-1)))


# Estadístico básico
plt.figure()
plt.boxplot([muestraLR, muestraMLP])
print('LR: Mejor Varianza Explicada= ', np.min(muestraLR))
print('LR: Peor Varianza Explicada= ', np.max(muestraLR))
print('LR: Promedio Varianza Explicada= ', np.mean(muestraLR))
print('MLP: Mejor Varianza Explicada= ', np.min(muestraMLP))
print('MLP: Peor Varianza Explicada= ', np.max(muestraMLP))
print('MLP: Promedio Varianza Explicada= ', np.mean(muestraMLP))


# Tests
from scipy.stats import shapiro # Normalidad
from scipy.stats import ttest_ind # Test parametrico
from scipy.stats import wilcoxon # Test no parametrico

_, p_valueLR= shapiro(muestraLR)
_, p_valueMLP= shapiro(muestraMLP)
print('p-value LR Shapiro-Wilk: ', p_valueLR)
print('p-value MLP Shapiro-Wilk: ', p_valueMLP)

print('Con intervalo de confianza del 95% (p-value 0.05) se acepta que el dato no es normal (p-value<0.05) o sí (p-value>0.05)')
print('SI SON NORMALES:')
_, p_valuettest= ttest_ind(muestraLR, muestraMLP)
print('ttest (LR vs MLP): (<0.05 hay diferencias)', p_valuettest)

print('SI NO SON NORMALES:')
_, p_valuewilcoxon= wilcoxon(muestraLR, muestraMLP)
print('Wilcoxon (LR vs MLP): (<0.05 hay diferencias)', p_valuewilcoxon)

print('EL MEJOR ALGORITMO SI HAY DIFERENCIAS ES EL QUE MEJOR SCORE PROMEDIO TENGA')