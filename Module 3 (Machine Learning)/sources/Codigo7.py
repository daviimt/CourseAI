#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:41:54 2023

@author: manupc

Ejemplo de cómo los modelos lineales no van bien con datasets no lineales
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Obtenemos datos (entrada=X/salida= y)
X, y= make_circles(n_samples=100)

# Insertamos error en datos
X+= np.random.randn(X.shape[0], X.shape[1])*0.01
class0= X[y==0, :]
class1= X[y==1, :]

## Límites de la figura
X1_min, X1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
X2_min, X2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

plt.scatter(class0[:, 0], class0[:, 1],
            color='red', marker='o', label='long')
plt.scatter(class1[:, 0],class1[:, 1],
            color='blue', marker='x', label='short')

plt.xlim(X1_min, X1_max)
plt.ylim(X2_min, X2_max)
plt.show()


# División en train y test
x_train, x_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.25)

# Creamos modelo de regresión logística
logisticRegr = LogisticRegression()

# Lo entrenamos
logisticRegr.fit(x_train, y_train)

# Comprobamos predicciones en training
train_p= logisticRegr.predict(x_train)
print('Predicciones correctas en training (predicción, correcto):\n', np.vstack((train_p, y_train)).T)
score = logisticRegr.score(x_train, y_train)
print('Accuracy en training: ', score)


# Comprobamos predicciones en test
test_p= logisticRegr.predict(x_test)
print('Predicciones correctas en test (predicción, correcto):\n', np.vstack((test_p, y_test)).T)
score = logisticRegr.score(x_test, y_test)
print('Accuracy en test: ', score)



# Mostramos parámetros del clasificador
print('Los parámetros del clasificador son:')
print('\tCoefs: ', logisticRegr.coef_)
print('\tIntercept: ', logisticRegr.intercept_)

# Mostramos gráficamente con mejor visualización
markers = ('s', 'x', 'o', '^', 'v') # Marcadores a usar
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') # COlores a usar

# Mapa de colores
from matplotlib.colors import ListedColormap
cmap = ListedColormap(colors[:2]) # Cogemos mapa para 2 clases

# Superficie de decisión
## Límites de la figura
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

## Creamos grid
resolucion= 0.02
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolucion),
                           np.arange(x2_min, x2_max, resolucion))

# Predecimos para cada valor del mesh su clase
Z = logisticRegr.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape) # Lo ponemos en forma de mesh

# Mostramos contorno y aplicamos límites
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

    
## Salidas reales
plt.scatter(class0[:, 0], class0[:, 1],
            color='red', marker='o', label='long')
plt.scatter(class1[:, 0],class1[:, 1],
            color='blue', marker='x', label='short')


# X2: Agregación de datos
X2= X[:, 0]**2 + X[:, 1]**2


# Mostramos nuevo
class0= X2[y==0]
class1= X2[y==1]
plt.figure()
plt.scatter(class0, np.zeros(len(class0)),
            color='red', marker='o', label='long')
plt.scatter(class1, np.ones(len(class1)),
            color='red', marker='o', label='long')

# División en train y test
x_train, x_test, y_train, y_test = \
    train_test_split(X2, y,
                     test_size=0.25)

x_train= x_train.reshape(-1, 1)    
x_test= x_test.reshape(-1, 1)    

# Creamos modelo de regresión logística
logisticRegr = LogisticRegression()

# Lo entrenamos
logisticRegr.fit(x_train, y_train)

# Comprobamos predicciones en training
train_p= logisticRegr.predict(x_train)
print('Predicciones correctas en training (predicción, correcto):\n', np.vstack((train_p, y_train)).T)
score = logisticRegr.score(x_train, y_train)
print('Accuracy en training: ', score)


# Comprobamos predicciones en test
test_p= logisticRegr.predict(x_test)
print('Predicciones correctas en test (predicción, correcto):\n', np.vstack((test_p, y_test)).T)
score = logisticRegr.score(x_test, y_test)
print('Accuracy en test: ', score)
