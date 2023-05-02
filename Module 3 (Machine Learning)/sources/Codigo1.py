#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:59:54 2023

@author: manupc
"""

# Implementación de un modelo Perceptrón y si aprendizaje con gradiente descendente

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Lectura de datos del dataset iris
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
iris =  pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                    names = col_names)
print(iris.sample(10))

# Nos vamos a quedar sólo con las clases que sean de setosa y de versicolor
# Como atributos, sólo vamos a predecir con las longitudes (pétalo y sépalo)
x= iris[['Sepal_Length','Petal_Length']].values

# Escalamos los datos por facilitar el aprendizaje
x= (x-x.mean(axis=0))


y = iris['Class'].values
y[y=='Iris-setosa']= -1
y[y=='Iris-versicolor']= 1
patrones= y!='Iris-virginica'
y= y[patrones].astype(int)
x= x[patrones, :].astype(float)


# Mostramos el dataset
class0= (y==-1)
class1= (y==1)
plt.scatter(x[class0, 0], x[class0, 1], color='red')
plt.scatter(x[class1, 0], x[class1, 1], color='blue')
plt.xlabel('Long. petalo')
plt.ylabel('Long. sepalo')
plt.legend(['Setosa', 'Versicolor'])
plt.show()


# Modelo de perceptrón
class Perceptron:
    def __init__(self, numEntradas, lmbda=0.01, epochs=50):
        self.lmbda= lmbda
        self.numEntradas= numEntradas
        self.epochs= epochs
        #self.W= np.random.randn(self.numEntradas+1) # 3 parámetros: w0 + x1*w1 + x2*w2
        self.W= np.ones(self.numEntradas+1)
        self.coste = []

    # calcula la salida de la neurona    
    def forward_pass(self, X):
        return self.W[0] + np.dot(X, self.W[1:])


    # Devuelve la etiqueta deseada
    def predict(self, X):
        return np.where(self.forward_pass(X) >= 0.0, 1, -1)


    # Calcula Dw, los cambios de los gradientes y la función de coste
    def backward_pass(self, output, y):
        errors = (y - output)
        cost= (errors**2).sum() / 2.0
        
        Dw= np.zeros(self.numEntradas+1)
        Dw[1:]= output.T.dot(errors)
        Dw[0]= errors.sum()
        return Dw, cost

    # Actualización de parámetros
    def update_weights(self, Dw):
        self.W+= self.lmbda*Dw

    # Función para hacer el aprendizaje
    def fit(self, X, y):
        self.coste = []

        for i in range(self.epochs):
            
            # Paso adelante
            output = self.forward_pass(X)
            Dw, cost= self.backward_pass(output, y)
            self.update_weights(Dw)
            
            self.coste.append(cost) # Para llevar la cuenta del coste por iteraciones
        return self






# Creación del modelo y aprendizaje
modelo = Perceptron(numEntradas= x.shape[1], epochs=30, lmbda=0.001)

# Aprendizaje
modelo.fit(x, y)

# Mostrar resultados
plt.figure()
plt.plot(range(1, len(modelo.coste) + 1), modelo.coste, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Coste')
plt.show()

# Cálculo del porcentaje de aciertos
yp= modelo.predict(x)
aciertos= 100*(yp==y).sum()/len(yp)
print('Porcentaje de aciertos en clasificación tras entrenamiento: ', aciertos)