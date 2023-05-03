def sumar_matrices(matriz1, matriz2):
    """
    Esta función recibe dos matrices como entrada y devuelve su suma,
    si es posible, o None en otro caso.
    """
    # Verificar si es posible la suma de las matrices
    if len(matriz1) != len(matriz2) or len(matriz1[0]) != len(matriz2[0]):
        return None
    
    # Crear la matriz resultante con ceros
    resultado = [[0 for j in range(len(matriz1[0]))] for i in range(len(matriz1))]
    
    # Realizar la suma de las matrices
    for i in range(len(matriz1)):
        for j in range(len(matriz1[0])):
            resultado[i][j] = matriz1[i][j] + matriz2[i][j]
    
    return resultado

matriz1 = [[1, 2, 3], [4, 5, 6]]
matriz2 = [[7, 8, 9], [10, 11, 12]]
print(sumar_matrices(matriz1, matriz2))
