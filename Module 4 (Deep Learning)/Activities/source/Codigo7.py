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




########################################################
## Capa sigmoide
class CapaSigmoide(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.sigmoid(inputs)



########################################################
## Capa sigmoide
class CapaReLU(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.maximum(0, inputs)
    




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
                self.capas.append( CapaReLU() )
                current_in= h
        
        # Capa de salida
        self.capas.append( CapaLineal(current_in, num_outputs) )
        self.capas.append( CapaReLU() )


    def call(self, inputs):
        curr_in= inputs
        for c in self.capas:
            output= c(curr_in)
            curr_in= output
        return output
        


# Cargamos los datos (regresión)
X= tf.random.uniform(shape=(100,2))
Y= tf.reduce_sum(X, axis=1)
noise= tf.random.normal(shape=(100,), stddev=0.01)
Y= Y+noise
Y= tf.sigmoid(Y)
Y= tf.reshape(Y, shape=(-1, 1))


# Creación del modelo
model= MLP(2, [10], 1)

# Entrenamiento MANUAL: 1000 iteraciones con J(w)= error cuadrático medio
lmbda= 0.1
num_itera= 10
for i in range(num_itera):
    
    # Forward pass
    with tf.GradientTape() as tape:
        Yp= model( tf.transpose(X) )
        loss= tf.reduce_mean((tf.transpose(Y) -Yp)**2)
    print('Loss en iteración ', i+1, ':', loss.numpy())
    
    # Backward pass
    DW= tape.gradient(loss, model.weights) # Coger gradientes
    
    # Update
    for w, dw in zip(model.weights, DW):
        w.assign_add( -lmbda*dw )
    #print('W', model.weights[0])
    #print('DW', DW[0])


