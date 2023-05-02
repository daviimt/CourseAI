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



# Implementación del entorno
class Entorno:
    # Constructor
    def __init__(self, timeHorizon= 50):
        
        # Horizonte de tiempo para finalizar episodio
        self.timeHorizon= timeHorizon
        
        ## Estados
        self.nEstados= 3
        self.sLabels= ['Tired', 'Energetic', 'Healthier']
        
        ## Acciones
        self.nAcciones= 3 # Acciones 0, 1, 2
        self.aLabels= ['Work', 'Gym', 'Sleep']
        
        ## MDP
        
        # Acciones disponibles en cada estado
        self.actDisponibles= {} # Work para todos. gym para Tired/energetic.
        for i in range(self.nEstados):
            self.actDisponibles[i]= list(range(3-i))

        
        # Probabilidades de transición
        self.transP= {} # Probabilidades de transición (s,a)-> p(s').
        self.transP[(0,0)]= np.array([1,   0,   0]) # Tired, work -> tired
        self.transP[(0,1)]= np.array([0.5, 0.5, 0]) # Tired, gym -> tired (0.5), energetic (0.5)
        self.transP[(0,2)]= np.array([0.2, 0.8, 0]) # Tired, sleep -> tired (0.2), energetic (0.8)
        
        self.transP[(1,0)]= np.array([0.8, 0.2, 0]) # Energetic, work -> tired (0.8), energetic (0.2)
        self.transP[(1,1)]= np.array([0,   0,   1]) # Energetic, gym -> healthier
        
        self.transP[(2,0)]= np.array([1.0, 0,   0]) # Healthier, work -> tired

        # Probabilidades de comenzar el episodio en cada estado (asumimos las siguientes)
        self.initP= np.array([0.5, 0.3, 0.2]) # tired (0.5), Energetic (0.3), Healthier (0.2)
        
        
        
        # Recompensas
        self.rewards= {}
        self.rewards[(0,0,0)]= 20.0
        self.rewards[(0,1,0)]= -10.0
        self.rewards[(0,1,1)]= -10.0
        self.rewards[(0,2,0)]= 0.0
        self.rewards[(0,2,1)]= 0.0
        self.rewards[(1,0,0)]= 40.0
        self.rewards[(1,0,1)]= 30.0
        self.rewards[(1,1,2)]= -10.0
        self.rewards[(2,0,0)]= 100.0
        
        # Variables internas: En qué estado estamos
        self.S= None
        self.steps= 0
        
        
    # Devuelve las acciones disponibles para el estado S
    def accionesDisponibles(self, S):
        return self.actDisponibles[S]
    
    
    # devuelve las probabilidades de transición p(s' | s,a )
    def transitionProbs(self, S, a):
        return self.transP[(S,a)]

    # devuelve las recompensas (s, a, s')
    def rewardValue(self, S, a, Sp):
        return self.rewards[(S,a,Sp)]

    
        
    # Resetea el entorno y devuelve el estado inicial
    def reset(self):
        # Calculamos estado inicial y reiniciamos pasos ejecutados
        self.S= np.random.choice(list(range(self.nEstados)), p=self.initP)
        self.steps= 0
        return self.S.copy()
    
    
    # Ejecuta una acción sobre el entorno
    def step(self, action):
        if (self.steps >= self.timeHorizon): # Si el episodio terminó, no devolvemos nada
            return None, None, None, None
        
        # Vemos que la acción es válida
        if (action not in self.actDisponibles[self.S]):
            raise Exception("Acción no permitida")
        
        # Calculamos estado siguiente
        Sp= np.random.choice(list(range(self.nEstados)), p=self.transP[(self.S, action)])
        R= self.rewards[(self.S, action, Sp)] # Calculamos recompensa
        
        
        self.steps+= 1 # Pasamos instante de tiempo
        done= True if (self.steps >= self.timeHorizon) else False
        self.S= Sp # Pasamos de estado
        return self.S.copy(), R, done, None

    
    
    # Indica cómo va el entorno
    def render(self):
        if (self.S is None):
            return
        print('Estoy en {} en paso {}'.format(self.sLabels[self.S], self.steps))
    
    



# Política epsilon-Greedy
# Función que devuelve una acción en 0..NumActions-1 según la política e-greedy,
#  para el estado S, teniendo política de agente pi(s) determinista 
#  dada en "politicaAgente"
#  Como entrada también se tienen los valores aproximados Q(s,a) en Q, y epsilon 
def politicaEpsilonGreedy(env, S, politicaAgente, Q, epsilon):
    
    if np.random.rand() < epsilon: # Politica aleatoria uniforme
        APermitidas= env.accionesDisponibles(S)
        return APermitidas[ np.random.randint(len(APermitidas)) ]
    
    else:
        return politicaAgente(env, S, Q)


def politicaAgente(env, S, Q):
    APermitidas= env.accionesDisponibles(S)
    return APermitidas [ np.argmax([Q[(S, a)] for a in APermitidas]) ]    


# Control de exploración vs convergencia con e-greedy
# Evolución de epsilon con los episodios:
eps0= 0.3 # Valor epsilon inicial
epsf= 0.01 # Valor epsilon final
epsEpisodes= 100 # Número de episodios a generar para decaer de eps0 a epsf
epsilon= eps0 # Al comienzo

# Función para actualizar epsilon según el episodio en el que estemos (x)
epsilonUpdate= lambda x: max(epsf, eps0+x*(epsf-eps0)/epsEpisodes)


# vemos cómo evoluciona por episodios
valoresEpsilon= [epsilon]
for ep in range(1, epsEpisodes+11):
    epsilon= epsilonUpdate(ep)
    valoresEpsilon.append(epsilon)

plt.plot(valoresEpsilon)
plt.xlabel("Episodios")
plt.ylabel("Epsilon")


# Método de Q-Learning
def QLearning(env, NumActions, NumEpisodios, eps0, epsf, epsEpisodes, alpha, gamma, umbralConvergencia=1e-2):
    
    # Inicialización de parámetros
    epsilon= eps0
    
    # Inicialización de tabla de Q(s,a)= 0 como diccionario Q[(s,a)]-> valor Q
    Qini= defaultdict(float)
    
    ep= 0
    fin= False
    while not fin:
        
        # Generación de episodio
        Q= Qini.copy()
        s= env.reset()
        done= False
        while not done:
            a= politicaEpsilonGreedy(env, s, politicaAgente, Q, epsilon)
            sp, r, done, _= env.step(a)
            ADisponibles= env.accionesDisponibles(sp)
            ap= ADisponibles[ np.argmax( [Q[(sp, a_)] for a_ in ADisponibles] ) ]
            Q[(s,a)]+= alpha*(r+gamma*Q[(sp, ap)] - Q[(s,a)])
            s= sp
        
        # Actualización de epsilon
        epsilon= max(epsf, eps0+ep*(epsf-eps0)/epsEpisodes)
        
        # Pasamos de episodio
        converge= np.max([ np.fabs(Q[k]-Qini[k]) for k in Q.keys() ]) <= umbralConvergencia
        Qini= Q
        ep+= 1
        if ep>=NumEpisodios or converge:
            fin= True
    return Qini, ep, converge



# Ejecución de Q-Learning

# Ejemplo en el entorno
env= Entorno()
NumActions= env.nAcciones
eps0= 0.8 # Valor epsilon inicial
epsf= 0.01 # Valor epsilon final
epsEpisodes= 1800 # Número de episodios a generar para decaer de eps0 a epsf
epsilon= eps0 # Al comienzo

gamma= 0.99 # Factor de descuento
alpha= 0.1 # Tasa de aprendizaje
MaxEpisodios= 2000 # Número de episodios a entrenar como máximo

Q, epiDone, converge= QLearning(env, NumActions, MaxEpisodios, eps0, epsf, epsEpisodes, alpha, gamma)
print('Converge: {} en {} episodios'.format(converge, epiDone))



# Evaluación de la política

# Método que genera num_episodios del entorno env con la política aprendida por el agente
def generarNEpisodiosPolitica(env, num_episodios, Q, gamma, render= False):

    TotalReturnsS0= [] # Guardar los returns de cada episodio
    for ep in range(num_episodios):
        done= False
        S= env.reset()
        if render:
            env.render()
        ReturnS0= 0.0
        steps= 0
        while not done:
            accion= politicaAgente(env, S, Q)
            Sp, r, done, _= env.step(accion)
            if render:
                print('Aplico acción {} en estado {}'.format(int(accion), S))
                env.render()
            ReturnS0+= (gamma**steps) * r
            steps+= 1
            S= Sp
        TotalReturnsS0.append(ReturnS0)
    return TotalReturnsS0


# Generamos histograma de returns
TotalReturns= generarNEpisodiosPolitica(env, 10000, Q, gamma=1.0)
print('Recompensa promedio esperada: {}'.format(np.mean(TotalReturns)))
plt.hist(TotalReturns)
plt.xlabel('Returns (gamma={})'.format(1.0))
plt.ylabel('Frecuencia')

