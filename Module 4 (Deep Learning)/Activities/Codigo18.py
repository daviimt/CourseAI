#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:20:56 2023

@author: manupc
"""

import gym
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Implementación de Q-Network como modelo de red feedforward
class QNetwork(tf.keras.Model):
    
    
    # Constructor. Como entrada tenemos:
    #  - n_inputs: El número de entradas a la red
    #  - n_hidden: El número de neuronas de la capa oculta
    #  - n_outputs: EL número de salidas de la red
    #  Se debe encargar de crear los módulos que componen el grafo del modelo
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(QNetwork, self).__init__()

        self.NumAcciones= n_outputs # Para conocer el número de acciones en el entorno
        
        # Modelo
        self.model= tf.keras.Sequential()

        # Entradas
        self.model.add( tf.keras.Input(shape=(n_inputs) ) )
        
        # Capa entradas-oculta
        self.model.add(tf.keras.layers.Dense(n_hidden, activation='relu'))
        
        # Capa oculta-salida
        self.model.add(tf.keras.layers.Dense(n_outputs, activation='linear'))

        
    
    def getNumAcciones(self):
        return self.NumAcciones
    
    # Hacemos un forward de la entrada x: Terminamos de definir cómo se conecta el grafo (si es necesario)
    def call(self, x):
        if len(x.shape) ==1:
            x= tf.reshape(x, shape=(1, -1))
        return self.model(x)



# Implementación de la política del agente
def politicaAgente(red, S):
    
    Qvals= red(S).numpy()
    
    # El agente escoge la acción de max Q
    a= np.argmax(Qvals, axis=-1)
    return a.squeeze() # Eliminamos todas las dimensiones extra
    

# Política de exploración epsilon-Greedy
def politicaEpsilonGreedy(red, S, epsilon):
    if (np.random.rand() < epsilon):
        return np.random.randint(red.getNumAcciones())
    else:
        return politicaAgente(red, S)




## ENTORNO

# Entorno
Entorno= 'CartPole-v1'
env= gym.make(Entorno) # Entorno para entrenamiento
testEnv= gym.make(Entorno)
#testEnv= gym.make(Entorno, render_mode= "rgb_array") # Entorno para test
TamEstado= env.observation_space.shape[0] # Tamaño del entorno gym
NumAcciones= env.action_space.n # Número de acciones del entorno gym
print('Entorno {} con tam. Estado={} y acciones={}'.format(Entorno, TamEstado, NumAcciones))


### REDES

# Inicialización de redes DQN, target, optimizador y función de pérdida
DQN= QNetwork(TamEstado, 100, NumAcciones)
target= QNetwork(TamEstado, 100, NumAcciones)
optimizer= tf.keras.optimizers.Adam(learning_rate= 0.001) # Algoritmo de aprendizaje
J= tf.keras.losses.MeanSquaredError() # Función de pértidad
DQN.compile(optimizer=optimizer, loss=J)


# Copiamos la red DQN en target
target.set_weights(DQN.get_weights()) 
SYNC_STEPS= 50 # Sincronizamos cada 500 pasos en el entorno


### POLÍTICA E-GREEDY
MaxEpisodes= 100 # Número de episodios a ejecutar
gamma= 0.99 # Factor de descuento
eps0= 0.5 # Valor epsilon inicial
epsf= 0.01 # Valor epsilon final
epsEpisodes= int(0.99*MaxEpisodes) # Número de episodios a generar para decaer de eps0 a epsf
epsilon= eps0 # Al comienzo

### REPLAY BUFFER

# Replay Buffer implementado como lista circular de capacidad máxima 1000
buffer= deque(maxlen=500)
BatchSize= 32 # Tamaño del batch a muestrear en cada iteración


# Inicializamos buffer con número mínimo de experiencias
S,_= env.reset() # Inicialización de entorno
while (len(buffer) < BatchSize):
    # Selección de acción y aplicación en entorno
    a= politicaEpsilonGreedy(DQN, S, epsilon)
    Sp, r, truncated, done, _= env.step(a)
    done= done or truncated
    
    # Guardar experiencia en el buffer
    buffer.append( (S, a, r, Sp, done) )
    
    # Pasamos de estado
    if not done:
        S= Sp  
    else:
        S, _= env.reset()
        

testEnv= gym.make(Entorno, render_mode= "human") # Entorno para test
for i in range(10):
    S, _= testEnv.reset()
    testEnv.render()
    for i in range(5000):
        S, r, t, d, info= testEnv.step( politicaAgente(DQN, S) )
        testEnv.render()
        if t or d:
            break


## ALGORITMO DQN
S, _= env.reset() # Inicializar entorno
steps_count= 0 # Para saber cuándo hay que sincronizar DQN y target
ep= 0
logReturn=[]
totalR= 0
logEp= []
while ep < MaxEpisodes: # Bucle principal
    
    # Sincronizar si es necesario
    if steps_count >= SYNC_STEPS :
        target.set_weights(DQN.get_weights()) 
        steps_count%= SYNC_STEPS
    
    # Generar experiencia
    steps_count+=1
    a= politicaEpsilonGreedy(DQN, S, epsilon)
    Sp, r, truncated, done, _= env.step(a)
    totalR+= r
    done= truncated or done
    buffer.append( (S, a, r, Sp, done) ) # Guardar experiencia en el buffer
    
    # Pasamos de estado
    if not done:
        S= Sp  
    else:
        S, _= env.reset()
        ep+= 1

    # Muestreamos batch
    indices = np.random.choice(len(buffer), BatchSize, replace=False)
    bS, bA, bR, bSp, bDone = zip(*[buffer[idx] for idx in indices])
    bS= np.array(bS)
    bA= np.array(bA)
    bR= np.array(bR)
    bSp= np.array(bSp)
    bDone= np.array(bDone)

    # Paso a tensores
    tS= tf.convert_to_tensor(bS, dtype='float32')
    tSp= tf.convert_to_tensor( bSp , dtype='float32')
    tA= tf.expand_dims( tf.convert_to_tensor( bA ) , axis=-1)
    tR= tf.convert_to_tensor( bR , dtype='float32')
    tD= tf.convert_to_tensor( bDone )

    # Cálculo de Qtarget(s,a)= r + gamma*max_{a'}Q(s',a')
    #maxQ_sp_ap = target(tSp).max(1)[0] 
    maxQ_sp_ap = tf.reduce_max(target(tSp), axis=-1).numpy()
    maxQ_sp_ap[bDone] = 0.0 # Ponemos valores Q de estados finales (no hay Sp siguiente)
    Qtarget= tR+gamma*maxQ_sp_ap
        
    # Cálculo de Q(s,a) para las acciones escogidas tA


    
    with tf.GradientTape() as tape:
        Q= DQN(tS)
        
        selected_a= np.stack( (np.arange(len(Q)) , bA ) ).T
        Qsa= tf.gather_nd(Q, selected_a)
        
        loss= tf.reduce_mean( (Qtarget - Qsa)**2 )
    
    gradients = tape.gradient(loss, DQN.trainable_variables)
    optimizer.apply_gradients(zip(gradients, DQN.trainable_variables))


    # Actualización de epsilon para política e-greedy
    epsilon= max(epsf, eps0+ep*(epsf-eps0)/epsEpisodes)
    
    # Comprobamos si hay que hacer test cada 10 episodios generados
    if done:
        logEp.append(ep)
        logReturn.append(totalR)
        print('Terminado episodio {} con R={}'.format(ep, totalR))
        totalR= 0
    
env.close()


plt.figure()
plt.plot(logEp, logReturn)
plt.xlabel('Episodios')
plt.ylabel('R')
plt.show()

# Ejecutamos entorno 500 pasos visualizando (OJO: la visualización tiene bug)

testEnv= gym.make(Entorno, render_mode= "human") # Entorno para test
S, _= testEnv.reset()
testEnv.render()
for i in range(5000):
    S, r, t, d, info= testEnv.step( politicaAgente(DQN, S) )
    testEnv.render()
    if t or d:
        break
testEnv.close()