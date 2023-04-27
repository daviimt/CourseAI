"""
@author: manupc
Ejemplo de sentencia for
"""

# for range
for dato in range(7):
    print('Primer for, dato ', dato)


# for range con inicio y fin
print('Dime un número entero positivo inicial: ')
inicio= int( input() )
print('Dime un número entero positivo final: ')
fin= int( input() )

for dato in range(inicio, fin):
    print('Segundo for, dato ', dato)


# for range con inicio, fin y step
print('Dime un número entero positivo inicial: ')
inicio= int( input() )
print('Dime un número entero positivo final: ')
fin= int( input() )
print('Dime un número entero positivo para el salto: ')
salto= int( input() )

for dato in range(inicio, fin, salto):
    print('ercer for, dato ', dato)


print('FIN')
