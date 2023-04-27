"""
@author: manupc
Ejemplo de sentencia while
"""

a= None

while a is None or a<=0:
    print('Dime un número positivo: ')
    a= float( input() )


print('Has escrito', a)
