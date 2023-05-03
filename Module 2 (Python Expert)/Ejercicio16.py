import matrices

# Ejemplo de uso de las funciones
matriz1 = [[1, 2], [3, 4]]
matriz2 = [[5, 6], [7, 8]]
resultado = matrices.sumar_matrices(matriz1, matriz2)
if resultado is not None:
    print("Suma de matrices:")
    for fila in resultado:
        print(fila)
else:
    print("Las matrices no tienen dimensiones compatibles para la suma.")
