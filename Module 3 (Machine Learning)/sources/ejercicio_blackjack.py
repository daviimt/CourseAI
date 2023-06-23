#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:59:54 2023

NOTA: Si tienes problemas para visualizar porque te da error de bibliotecas MESA-LOADER:
    conda install -c conda-forge libstdcxx-ng


    
CliffWalking-v0
========================

Se trata de caminar desde un punto a otro, teniendo en cuenta que al lado tenemos
un precipicio por el que podemos caer. El mapa tiene tamaño 4 filas x 12 columnas:
    - Se empieza en la esquina inferior izquierda (3,0)
    - Se desea llegar a la esquina inferior derecha (3, 11)
    - En todos los valores de la última fila desde (3,1) hasta (3,10) hay un precipicio
    
Se puede ejecutar una de 4 acciones en cada paso:
    - Ir hacia arriba (acción 0)
    - Ir hacia abajo (acción 2)
    - Ir hacia la dcha (acción 1)
    - Ir hacia la izq (acción 3)
    
Por cada paso que se da, se pierte 1 unidad de tiempo (recompensa=-1). 
Si se cae al precipicio, se obtiene recompensa de -100 y se empieza otra vez
Si se llega a la meta, se termina el juego


    
Blackjack-v1
========================

El blackjack es un juego de cargas. Consiste en obtener cartas hasta que la suma 
de todas sea 21. Los valores de las cartas son:
    
     - Sota, Caballo, Reuy (Jack, Queen, King): Valor 10
     - Ases: Pueden contar como 11 o como 1, a elección del jugador
     - Resto de cartas (2-9): Tienen su propio valor.
     
El juego tiene 2 acciones: Pedir nueva carta (acción 1) o plantarse (acción 0)

Para ganar el juego puede ocurrir que:
    
    - El jugador tenga cartas por valor >21 : El jugador pierde (recompensa -1)
    - El jugador y la banca tengan el mismo valor (empate, recompensa 0)
    - El jugador tenga más valor que la banca, sin pasarse: El jugador gana (recompensa 1)

@author: manupc
"""

# Implementación de Q-Learning para resolver el MDP de ejemplo

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gym



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



#######################
# Ejecución de Q-Learning

## Creación del entorno
env= gym.make('Blackjack-v1')



# Obtener número de acciones (env.action_space.n)
NumActions= env.action_space.n


# Establecer hiperparámetros
alpha= ??? # Tasa de aprendizaje
MaxEpisodios= ??? # Número de episodios a entrenar como máximo
MaxSteps= ??? # Máximo de pasos por episodio

eps0= ??? # Valor epsilon inicial
epsf= ?? # Valor epsilon final
epsEpisodes= int(0.99*MaxEpisodios) # Número de episodios a generar para decaer de eps0 a epsf
epsilon= eps0 # Al comienzo

gamma= ??? # Factor de descuento


# Ejecución del algoritmo de Q-Learning
Q, epiDone, todosR= QLearning(env, NumActions, MaxEpisodios, eps0, epsf, epsEpisodes, alpha, gamma)
plt.plot(todosR)
plt.xlabel('Num. Episodios')
plt.ylabel('R')


# Ejecución de la política

# Crear el entorno con opción render_mode='human'
env= gym.make(?????, render_mode='human')

# Obtener el estado inicial

# Renderizar con env.render()

# Bucle para ejecutar un episodio
while not done:
    # Obtener acción para el estado actual
    
    # Ejecutar la acción en el entorno
    
    # Renderizar
    
    # Actualizar estado siguiente
