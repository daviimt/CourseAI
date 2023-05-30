def sumar_matrices(matriz1, matriz2):
    """Suma dos matrices."""
    if len(matriz1) != len(matriz2) or len(matriz1[0]) != len(matriz2[0]):
        return None
    resultado = []
    for i in range(len(matriz1)):
        fila = []
        for j in range(len(matriz1[0])):
            fila.append(matriz1[i][j] + matriz2[i][j])
        resultado.append(fila)
    return resultado


def restar_matrices(matriz1, matriz2):
    """Resta dos matrices."""
    if len(matriz1) != len(matriz2) or len(matriz1[0]) != len(matriz2[0]):
        return None
    resultado = []
    for i in range(len(matriz1)):
        fila = []
        for j in range(len(matriz1[0])):
            fila.append(matriz1[i][j] - matriz2[i][j])
        resultado.append(fila)
    return resultado


def multiplicar_matrices(matriz1, matriz2):
    """Multiplica dos matrices."""
    if len(matriz1[0]) != len(matriz2):
        return None
    resultado = []
    for i in range(len(matriz1)):
        fila = []
        for j in range(len(matriz2[0])):
            suma = 0
            for k in range(len(matriz2)):
                suma += matriz1[i][k] * matriz2[k][j]
            fila.append(suma)
        resultado.append(fila)
    return resultado
