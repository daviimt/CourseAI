class Matriz:
    def __init__(self, filas, columnas):
        self.filas = filas
        self.columnas = columnas
        self.matriz = [[0 for j in range(columnas)] for i in range(filas)]

    def __str__(self):
        cadena = ""
        for i in range(self.filas):
            cadena += "| "
            for j in range(self.columnas):
                cadena += str(self.matriz[i][j]) + " "
            cadena += "|\n"
        return cadena

    def asignar_valor(self, fila, columna, valor):
        self.matriz[fila][columna] = valor

    def obtener_valor(self, fila, columna):
        return self.matriz[fila][columna]

    def suma(self, otra_matriz):
        if self.filas != otra_matriz.filas or self.columnas != otra_matriz.columnas:
            return None
        resultado = Matriz(self.filas, self.columnas)
        for i in range(self.filas):
            for j in range(self.columnas):
                resultado.matriz[i][j] = self.matriz[i][j] + otra_matriz.matriz[i][j]
        return resultado

    def resta(self, otra_matriz):
        if self.filas != otra_matriz.filas or self.columnas != otra_matriz.columnas:
            return None
        resultado = Matriz(self.filas, self.columnas)
        for i in range(self.filas):
            for j in range(self.columnas):
                resultado.matriz[i][j] = self.matriz[i][j] - otra_matriz.matriz[i][j]
        return resultado

    def multiplicacion(self, otra_matriz):
        if self.columnas != otra_matriz.filas:
            return None
        resultado = Matriz(self.filas, otra_matriz.columnas)
        for i in range(self.filas):
            for j in range(otra_matriz.columnas):
                for k in range(self.columnas):
                    resultado.matriz[i][j] += self.matriz[i][k] * otra_matriz.matriz[k][j]
        return resultado
