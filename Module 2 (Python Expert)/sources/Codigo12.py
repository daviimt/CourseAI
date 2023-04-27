"""
@author: manupc
Ejemplo de uso de imports
"""
import math as Mates
from math import log10
from math import sqrt as RaizCuadrada
from sys import *

A= 9
print( 'Raíz cuadrada de ', A, ': ', RaizCuadrada(A))
print( 'Log_10 de ', A, ': ', log10(A))
print( 'Coseno de ', A, ': ', Mates.cos(A))

print(dir()) # dir es una función de sys, que devuelve el contenido 
             # de los nombres actualmente definidos

