#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:55:45 2023

@author: manupc
"""

import tensorflow as tf


########################################################
## Capa lineal
class CapaLineal(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(num_outputs, num_inputs), dtype="float32"),
            trainable=True
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(num_outputs,1), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(self.w, inputs) + self.b



# Capa de ejemplo que recoge 2 entradas y da 3 salidas
layer= CapaLineal(2, 3) 
y= layer(tf.ones((2, 1)))
print('Salida de la capa lineal: ', y.numpy())



########################################################
## Capa sigmoide
class CapaSigmoide(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.sigmoid(inputs)


layer= CapaSigmoide() 
entradas= tf.convert_to_tensor([[-1],
                               [0],
                               [1]], dtype=tf.float64)
y= layer(entradas)
print('Salida de la capa sigmoide: ', y)




#######################################
## Modelo (versión como clase)
class MLP(tf.keras.Model):
    def __init__(self, num_inputs, lista_hidden, num_outputs):
        super().__init__()
        self.capas= []
        current_in= num_inputs

        # Capas ocultas
        if lista_hidden is not None:
            for h in lista_hidden:
                self.capas.append( CapaLineal(current_in, h) )
                self.capas.append( CapaSigmoide() )
                current_in= h
        
        # Capa de salida
        self.capas.append( CapaLineal(current_in, num_outputs) )
        self.capas.append( CapaSigmoide() )


    def call(self, inputs):
        curr_in= inputs
        for c in self.capas:
            output= c(curr_in)
            curr_in= output
        return output
        

# Ejemplo de modelo perceptrón con 2 entradas, 1 capa oculta (3 neuronas) y 1 salida
model= MLP(2, [3], 1)
entradas= tf.convert_to_tensor([[-1, 1],
                               [0, -1]], dtype=tf.float64)
salidas= model(entradas)
print('Salidas del MLP: ', salidas)

entradas= tf.convert_to_tensor([[-1],
                               [0]], dtype=tf.float64)
salidas= model(entradas)
print('Salidas del MLP: ', salidas)

print('Parámetros del modelo:')
print(model.weights)



#######################################
## Modelo (versión con Sequential)

model= tf.keras.Sequential()
model.add(CapaLineal(2, 3))
model.add(CapaSigmoide())
model.add(CapaLineal(3, 1))
model.add(CapaSigmoide())

entradas= tf.convert_to_tensor([[-1, 1],
                               [0, -1]], dtype=tf.float64)
salidas= model(entradas)
print('Salidas del MLP (sequential): ', salidas)

entradas= tf.convert_to_tensor([[-1],
                               [0]], dtype=tf.float64)
salidas= model(entradas)
print('Salidas del MLP (sequential): ', salidas)

print('Parámetros del modelo (sequential):')
print(model.weights)
