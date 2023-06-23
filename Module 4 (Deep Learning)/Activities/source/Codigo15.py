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
# NAR neural network model
###################################################################
class NAR(tf.keras.Model):
    
    # Constructor
    # Inputs:
    #   T: Time horizon of x(t)= f(x(t-1), x(t-2), ..., x(t-T))
    #   layers: List of pairs [#neurons, activation] for each non-linear layer, e.g. [[30, 'relu']]
    #   autoScaleInputs: True to weight input values; false to provide raw inputs to the quantum circuit
    #   autoScaleOutputs: True to scale and bias circuit output values; false to provide raw circuit outputs as results
    def __init__(self, T, layers):
        
        super(NAR, self).__init__()

        # set parameters
        self.n_inputs= 1
        self.T= T
        self.n_outputs= 1
        
        # Create temporal buffer
        self.Buffer = np.zeros(T, dtype=np.float32)

        
        # Create layers
        sequence= []
        for layer in layers:
            
            activ= 'linear' if layer[1] is None else layer[1]
            
            currentLayer= tf.keras.layers.Dense(layer[0], activation=activ)
            sequence.append(currentLayer)
        output_layer= tf.keras.layers.Dense(self.n_outputs, activation='linear')
        sequence.append(output_layer)
        
        
        
        # Pack all layers
        self.net= tf.keras.Sequential()
        for layer in sequence:
            self.net.add(layer)

        # Build weights
        self(np.array([0.0], dtype=np.float32))
        
    # Reset network internal state
    def reset(self):
        self.Buffer[:] = 0
            
    # Forward pass
    def call(self, x):

        self.Buffer[0]= x.numpy()
        out= self.net(tf.convert_to_tensor(self.Buffer.reshape(1, -1), dtype='float32'))
        self.Buffer[1:-1]= self.Buffer[2:]
        self.Buffer[-1]= out.numpy()[0]
        
        return out
 


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
trData= tf.convert_to_tensor(data[:-12])
valData= tf.convert_to_tensor(data[-12:])
NDataTraining= len(trData)
NDataVal= len(valData)



# Transform into input/output patterns with time horizon T
T= 12 # Time horizon


# # Training parameters and model definition

lr= 0.01 # Learning rate of training algorithm
layers= [[10, 'tanh']] # Description (number of neurons and activation function) of intermediate layers
optimizer= tf.optimizers.RMSprop(lr) # Network optimizer
MaxEpochs= 30 # Maximum number of epochs to run the algorithm
MaxExecutions= 1 # Number of experiments
kFoldSplits= 4 # value of K for time series fold validation


# # Training
# 

valMSE= 0.0 # History of MSE in the validation dataset for each fold
trainMSE= [] # History of MSE in the training data for each fold


# Create network model
model= NAR(T, layers)
    
for epoch in range(MaxEpochs):

    # Get training and test input/output patterns as tensors
    trIn= trData[:-1]
    trOut= trData[1:]

    # fit the model
    print('Training epoch {}...'.format(epoch), end='\r')


    trLoss= 0
    model.reset()
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        for t in range(len(trIn)):
            out= model(trIn[t])
            trLoss+= (out-trOut[t])**2
        trLoss/= (len(trIn)-1)
    trainMSE.append(trLoss.numpy().squeeze())
    variables = model.trainable_variables
    gradients = tape.gradient(trLoss, variables)
    optimizer.apply_gradients(zip(gradients, variables))


    print('Epoch {}, MSE tr {:.3f}'.format(epoch, trLoss.numpy().squeeze()))
    
# Generate generalization metrics in validation
model.reset()
trOuts= []
for t in range(len(trData)):
    out= model(trData[t])
    trOuts.append(out)

valOuts= []
valLoss= 0
for t in range(len(valData)-1):
    out= model(valData[t])
    valLoss+= (out-valData[t+1])**2
    valOuts.append(out.numpy())
valLoss/= (len(valData)-1)
valMSE= valLoss.numpy().squeeze()

print('End of Execution. MSE tr {:.3f}, val {:.3f}'.format(trainMSE[-1], valMSE))
        



# Un-do preprocessing
valOuts= np.array([trOuts[-1], *valOuts]).squeeze()
t= np.array(range(NDataTraining, len(data)))
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






