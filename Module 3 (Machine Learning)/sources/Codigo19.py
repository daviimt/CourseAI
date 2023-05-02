#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:32:47 2023

@author: manupc
"""

import numpy as np
import matplotlib.pyplot as plt
# Ejemplo de implementación de Value Iteration para resolver un MDP

### Simulación del entorno

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
    
    

## Función para generar episodios
def generarNEpisodios(env, num_episodios, politica, gamma, render= False):

    TotalReturnsS0= [] # Guardar los returns de cada episodio
    for ep in range(num_episodios):
        done= False
        S= env.reset()
        if render:
            env.render()
        ReturnS0= 0.0
        steps= 0
        while not done:
            accion= politica(S, env.accionesDisponibles(S))
            Sp, r, done, _= env.step(accion)
            if render:
                print('Aplico acción {} en estado {}'.format(int(accion), S))
                env.render()
            ReturnS0+= (gamma**steps) * r
            steps+= 1
            S= Sp
        TotalReturnsS0.append(ReturnS0)
    return TotalReturnsS0




#### Algoritmo VI
def ValueIteration(env, gamma, iteraciones=20, umbralConvergencia=1e-20):
    
    # Tabla (estado, V(estado)). Inicializada a 0
    Vtable= np.zeros(env.nEstados)
    
    fin= False
    converge= False
    it= 0
    while not fin: # Ejecutamos hasta criterio de parada
        auxVtable= Vtable.copy()
        
        # Pasamos por cada estado calculando la tabla Q(s,a)= sum_{s'} p(s'|s,a)*(r(s,a,s')+\gamma*V(s))
        for s in range(env.nEstados):
            Qtable= np.zeros(env.nAcciones)
            for a in env.accionesDisponibles(s):
                transP= env.transitionProbs(s, a) # Probabilidades de transición
                for sp,prob in enumerate(transP):
                    if prob>0.0:
                        Qtable[a]+= prob*(env.rewardValue(s,a,sp) + gamma*auxVtable[sp] )
            
            # Actualizamos V(s)= max_{a} Q(s,a)
            Vtable[s]= np.max(Qtable)
        
        # Pasamos de iteración
        it+= 1
        
        # Comprobamos convergencia
        converge= np.max(np.fabs(Vtable-auxVtable)) <= umbralConvergencia
        
        # Criterio de parada
        fin= (it>=iteraciones or converge)
        
    # Devolvemos la tabla de valores, iteraciones realizadas y si ha convergido
    return Vtable, it, converge


# Paso 2
def ExtractPolicy(env, Vtable, gamma):
    policy= np.zeros(len(Vtable), dtype=int) # Creamos la política determinista para cada estado
    
    for s in range(len(Vtable)):
        Qtable= np.zeros(env.nAcciones)
        for a in env.accionesDisponibles(s):
            transP= env.transitionProbs(s, a) # Probabilidades de transición
            for sp,prob in enumerate(transP):
                if prob>0.0:
                    Qtable[a]+= prob*(env.rewardValue(s,a,sp) + gamma*Vtable[sp] )

        # Actualizamos política
        policy[s]= np.argmax(Qtable)
    return policy



# Ejemplo de ejecución de Value Iteration

env= Entorno()
gamma= 0.99 # OJO: Si se pone= 1.0 V(s) -> infinito. ValueIteration no converge
MaxIteraciones= 50000

# Paso 1
VTable, totalItera, converge= ValueIteration(env, gamma, MaxIteraciones)
print('Converge: {} en {} iteraciones'.format(converge, totalItera))
# Paso 2
policy= ExtractPolicy(env, VTable, gamma)
print('Política obtenida:')
for i in range(len(policy)):
    print('\tEn estado {} se aplica {}'.format(env.sLabels[i], env.aLabels[policy[i]]))

# Política del agente aprendida
def AgentPolicy(S, accionesDisponibles):
    return policy[S]

# Ejecutamos 1 episodio
print('\n\nEpisodio de ejemplo:')
TotalReturns= generarNEpisodios(env, 1, AgentPolicy, gamma=1.0, render=True)
print('Fin del episodio con Return={}\n'.format(TotalReturns))

# Veamos cómo se distribuyen las frecuencias

TotalReturns= generarNEpisodios(env, 10000, AgentPolicy, gamma=1.0)
print('Recompensa promedio esperada: {}'.format(np.mean(TotalReturns)))
plt.hist(TotalReturns)
plt.xlabel('Returns (gamma={})'.format(1.0))
plt.ylabel('Frecuencia')
