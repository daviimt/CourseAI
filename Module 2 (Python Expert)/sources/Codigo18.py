"""
@author: manupc
Ejemplo de uso de clases: Clase complejo
"""

from CodigoComplejo import Complejo

# Ejemplo de uso de complejos
a= Complejo(1, 0)
b= Complejo(1, 1)
print('Complejo a: ', a.Real, "+", a.Imag, 'i')
print('Complejo a: ', a)

# Operaciones
print('Conjugado de b: ', b.Conjugado())
print('Módulo de b: ', b.Abs())
print('a+b: ', a.Suma(b))
print('a-b: ', a.Resta(b))