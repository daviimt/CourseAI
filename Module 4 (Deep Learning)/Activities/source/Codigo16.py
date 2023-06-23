#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:07:41 2023

@author: manupc
"""


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import linregress

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'




###################################################################
# LSTM network + Dense output 1layer
###################################################################
class LSTM(tf.keras.Model):    
    # Constructor.
    # Inputs:
    #    n_inputs: Number of network inputs
    #    n_outputs: Number of network outputs
    #    layers: List of pairs [#neurons, activation] for each non-linear layer, e.g. [[30, 'relu']]
    def __init__(self, n_features, n_inputs, n_outputs, layers):
        super(LSTM, self).__init__()        
        self.n_features= n_features      
        self.n_inputs= n_inputs
        self.n_outputs= n_outputs
        self.capas= layers
        
        # Create layers
        sequence= []
        #first layer will be a lstm layer
        layer=layers[0]
        firstLayer= tf.keras.layers.LSTM(layer[0], activation=layer[1], 
                                         input_shape = (self.n_inputs, self.n_features))
        sequence.append(firstLayer)
        
        #now the rest ones
        for layer in layers[1:]:                              
            currentLayer= tf.keras.layers.Dense(layer[0], activation=layer[1])
            sequence.append(currentLayer)
        output_layer= tf.keras.layers.Dense(self.n_outputs, activation='linear')
        sequence.append(output_layer)              
        
        # Pack all layers
        self.net= tf.keras.Sequential()
        for layer in sequence:
            self.net.add(layer)
            
        self.net.summary()
                
        # Create weights
        self(tf.random.normal(shape=(1, self.n_inputs)))
                
    # forward pass for input batch x
    def call(self, x):
        x= tf.convert_to_tensor(x, dtype=tf.float32)
        return self.net(x)



# # Data loading and plot
df= pd.read_csv('AirPassengers.csv')
print(df.head())
data= df['#Passengers'].to_numpy().astype(np.float32)
plt.plot(data)


# Preprocesamiento

# Reducimos varianza de la serie con transformación logarítmica
data= np.log(data)


# Eliminar tendencia de la serie (hacer media 0)
# Asumimos tendencia lineal
t= np.array(range(1, len(data)+1))
result= linregress(t, data)
slope= result.slope
intercept= result.intercept
line= t*slope + intercept
data-= line
plt.figure()
plt.plot(data)
print('Slope: {:.3f}, Intercept: {:.3f}'.format(slope, intercept))


# train/test split
NDataTraining= len(data)-24
NDataVal= 12

# Transform into input/output patterns with time horizon T
T= 12 # Time horizon

inputsTr, outputsTr= [], []
inputsVal, outputsVal= [], []

# Create training/validation  input/output patterns
for i in range(len(data)-T):
    inPattern= data[i:(i+T)]
    outPattern= data[i+T]
    if i < NDataTraining:
        inputsTr.append(inPattern)
        outputsTr.append(outPattern)
    else:
        inputsVal.append(inPattern)
        outputsVal.append(outPattern)

# Scale inputs to [-pi/2, pi/2] as preprocessing for the quantum circuit
inputsTr= np.array(inputsTr).reshape(-1, T)
outputsTr= np.array(outputsTr).reshape(-1, 1)

# Scale inputs also in validation data
inputsVal= np.array(inputsVal).reshape(-1, T)
outputsVal= np.array(outputsVal).reshape(-1, 1)


# Time horizon
n_inputs= T
n_features= 1
n_outputs= 1

lr= 0.01 # Learning rate of training algorithm
#layers= [[10, 'relu'],[5, 'relu']]
layers= [[10, 'relu']]

MaxEpochs= 30 # Maximum number of epochs to run the algorithm
BatchSize= 64#32 # Batch size of training


# # Training

valMSE= 0.0 # History of MSE in the validation dataset for each fold
trainMSE= [] # History of MSE in the training data for each fold

# Transfor validation dataset to tensor
valIn= tf.convert_to_tensor(inputsVal, dtype='float32')
valOut= tf.convert_to_tensor(outputsVal, dtype='float32')

model= LSTM(n_features, n_inputs, n_outputs, layers)
model.compile(loss=tf.keras.losses.MSE,
          optimizer=tf.optimizers.RMSprop(learning_rate= lr),
          metrics=['mean_squared_error'])


trIn= tf.convert_to_tensor(inputsTr, dtype='float32')
trOut= tf.convert_to_tensor(outputsTr, dtype='float32')
model.fit(trIn, trOut, batch_size=BatchSize, epochs=MaxEpochs, verbose=0)

scoresTr = model.evaluate(trIn, trOut, verbose=0)
scoresVal = model.evaluate(valIn, valOut, verbose=0)
print('Scores in Training', scoresTr)
print('Scores in Val', scoresVal)


t= np.array(range(len(data)-12, len(data)))
valOuts= model.predict(valIn).squeeze()
line= t*slope + intercept
valOuts+= line
valOuts= np.exp(valOuts)


# Plot real validation time series, averaged validation network outputs, and its standard deviation
plt.figure()
plt.plot(df['#Passengers'])
plt.plot(t, valOuts, color='red')
plt.legend(['Real data', 'Prediction Avg.'])
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
