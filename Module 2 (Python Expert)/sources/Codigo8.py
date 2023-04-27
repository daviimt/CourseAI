"""
@author: manupc
Ejemplo de sentencias break y continue
"""

# break
print('inicio del bucle 1')
for dato in range(7):
    print('Primer for, dato ', dato)
    if dato == 3: # Si estamos en el tercer dato, terminamos
        break # No se ejecuta para los datos 4, 5, 6
print('fin del bucle 1')


# continue
for dato in range(7):
    print('Segundo for, dato ', dato)
    if dato != 3:
        continue
    print('Voy por la mitad!!! : ', dato)
print('fin del bucle 2')


print('FIN')
