from ClaseMatriz import Matriz

matriz1 = Matriz(2, 3)
matriz1.asignar_valor(0, 0, 1)
matriz1.asignar_valor(0, 1, 2)
matriz1.asignar_valor(0, 2, 3)
matriz1.asignar_valor(1, 0, 4)
matriz1.asignar_valor(1, 1, 5)
matriz1.asignar_valor(1, 2, 6)

matriz2 = Matriz(2, 3)
matriz2.asignar_valor(0, 0, 7)
matriz2.asignar_valor(0, 1, 8)
matriz2.asignar_valor(0, 2, 9)
matriz2.asignar_valor(1, 0, 10)
matriz2.asignar_valor(1, 1, 11)
matriz2.asignar_valor(1, 2, 12)

resultado = matriz1.suma(matriz2)
print(resultado)

resultado = matriz1.resta
