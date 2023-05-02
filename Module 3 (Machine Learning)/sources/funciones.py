"""
@author: manupc
Ejemplo de uso de funciones en ficheros externos
"""

from math import sqrt


# Función que calcula la media de una lista de valores
def Media(ListaValores : list[float]) -> float:
    
    # Calculamos el tamaño de la lista
    N= len(ListaValores)
    
    # Sumamos todos los valores y los dividimos por N
    suma= 0.0
    for elem in ListaValores:
        suma+= elem
    return suma/N


# Función que calcula la desviación estándar de una lista de valores
def DesvTipica(ListaValores : list[float]) -> float:
    
    # Calculamos el tamaño de la lista
    N= len(ListaValores)
    
    # Calculamos su media
    m= Media(ListaValores)
    
    # Calculamos el sumatorio
    suma= 0.0
    for elem in ListaValores:
        suma+= (elem - m)**2
    varianza= suma/(N-1)
    
    # Devolvemos la raíz cuadrada de la varianza
    return sqrt(varianza)

