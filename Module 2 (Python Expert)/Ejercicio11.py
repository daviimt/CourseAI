def multiplicar_matrices(matriz1, matriz2):
  
    # Verificar si es posible la multiplicación de las matrices
    if len(matriz1[0]) != len(matriz2):
        return None
    
    # Crear la matriz resultante con ceros
    resultado = [[0 for j in range(len(matriz2[0]))] for i in range(len(matriz1))]
    
    # Realizar la multiplicación de las matrices
    for i in range(len(matriz1)):
        for j in range(len(matriz2[0])):
            for k in range(len(matriz2)):
                resultado[i][j] += matriz1[i][k] * matriz2[k][j]
    
    return resultado

matriz1 = [[1, 2, 3], [4, 5, 6]]
matriz2 = [[7, 8], [9, 10], [11, 12]]
print(multiplicar_matrices(matriz1, matriz2))
