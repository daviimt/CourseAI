#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:59:54 2023

@author: manupc
"""

# Implementación de Q-Learning para resolver el MDP de ejemplo

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gym


#env= gym.make('CliffWalking-v0')
env= gym.make('Taxi-v3')

# Política epsilon-Greedy
# Función que devuelve una acción en 0..NumActions-1 según la política e-greedy,
#  para el estado S, teniendo política de agente pi(s) determinista 
#  dada en "politicaAgente"
#  Como entrada también se tienen los valores aproximados Q(s,a) en Q, y epsilon 
def politicaEpsilonGreedy(env, S, politicaAgente, Q, epsilon):
    
    if np.random.rand() < epsilon: # Politica aleatoria uniforme
        APermitidas= list(range(NumActions))
        return APermitidas[ np.random.choice(APermitidas) ]
    
    else:
        return politicaAgente(env, S, Q)
    
def politicaAgente(env, S, Q):
    APermitidas= list(range(NumActions)) # Las 4 acciones del entorno
    return APermitidas [ np.argmax([Q[(S, a)] for a in APermitidas]) ]    


# Función para actualizar epsilon según el episodio en el que estemos (x)
epsilonUpdate= lambda x: max(epsf, eps0+x*(epsf-eps0)/epsEpisodes)


# Método de Q-Learning
def QLearning(env, NumActions, NumEpisodios, eps0, epsf, epsEpisodes, alpha, gamma, umbralConvergencia=1e-2):
    
    # Inicialización de parámetros
    epsilon= eps0
    todosR= []
    # Inicialización de tabla de Q(s,a)= 0 como diccionario Q[(s,a)]-> valor Q
    Qini= defaultdict(float)
    
    ep= 0
    for i in range(NumEpisodios):
        
        # Generación de episodio
        Q= Qini.copy()
        s, _= env.reset()
        done= False
        R= 0 # Recompensa total
        steps= 0
        while not done:
            a= politicaEpsilonGreedy(env, s, politicaAgente, Q, epsilon)
            sp, r, truncated, done, _= env.step(a)
            R+= r
            ADisponibles= list(range(NumActions))
            
            ap= ADisponibles[ np.argmax( [Q[(sp, a_)] for a_ in ADisponibles] ) ]
            Q[(s,a)]+= alpha*(r+gamma*Q[(sp, ap)] - Q[(s,a)])
            s= sp
            steps+= 1
            if steps>= MaxSteps:
                done= True

        
        
        # Actualización de epsilon
        epsilon= max(epsf, eps0+ep*(epsf-eps0)/epsEpisodes)
        
        # Pasamos de episodio
        Qini= Q
        ep+= 1
        print('Episodio ', ep, 'terminado con R=', R)
        todosR.append(R)

    return Qini, ep, todosR



# Ejecución de Q-Learning

# Ejemplo en el entorno
NumActions= env.action_space.n

alpha= 0.1 # Tasa de aprendizaje
MaxEpisodios= 5000 # Número de episodios a entrenar como máximo
MaxSteps= 5000 # Máximo de pasos por episodio

eps0= 1.0 # Valor epsilon inicial
epsf= 0.001 # Valor epsilon final
epsEpisodes= int(0.99*MaxEpisodios) # Número de episodios a generar para decaer de eps0 a epsf
epsilon= eps0 # Al comienzo

gamma= 0.9 # Factor de descuento

Q, epiDone, todosR= QLearning(env, NumActions, MaxEpisodios, eps0, epsf, epsEpisodes, alpha, gamma)
plt.plot(todosR)
plt.xlabel('Num. Episodios')
plt.ylabel('R')


# Ejecución de la política
"""
env= gym.make('Taxi-v3', render_mode='rgb_array')
s, _= env.reset()
env.render()
done= False
while not done:
    a= politicaAgente(env, s, Q)
    sp, r, truncated, done, _= env.step(a)
    env.render()
    s= sp
"""