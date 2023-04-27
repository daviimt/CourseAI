"""
@author: manupc
Ejemplo de sentencia if..elif..
"""

print('Dime un número entero: ')
a= int( input() )

if a < 0:
    print('El número es negativo')
elif a > 0: 
    print('El número es positivo')
else:
    print('El número es cero')

print('FIN')
