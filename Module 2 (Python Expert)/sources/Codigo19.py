"""
@author: manupc
Ejemplo de uso de clases: Clase complejo
"""

from CodigoAnimales import Perro, Vaca, Abeja

miPerro= Perro('Nala', 'Negro')
miVaca= Vaca('Paquita')
miAbeja= Abeja('Picantilla')

print(miPerro)
print(' y dice: ')
miPerro.hablar()
miPerro.moverse()


print(miVaca)
print(' y dice: ')
miVaca.hablar()
miVaca.moverse()



print(miAbeja)
print(' y dice: ')
miAbeja.hablar()
miAbeja.moverse()
miAbeja.picar()

