
tabla = {
    "Caso":[1,2,3,4,5,6,7,8,9],
    "Tasacion":["+","-","+","-","-","+","+","+","+"],
    "Valoracion":["-","-","+","+","+","+","+","-","+"],
    "Entradas":["No","Si","No","Si","Si","Si","Si","Si","No"],
    "Balance":["+","-","+","+","-","-","+","-","-"],
    "Aprobar":["No","No","Si","Si","No","Si","Si","No","No"]
}

# Crear el diccionario con la clave "Caso"
diccionario = {}
for i in range(len(tabla["Caso"])):
    caso = tabla["Caso"][i]
    diccionario[caso] = {}
    for columna in tabla:
        diccionario[caso][columna] = tabla[columna][i]

# Iterar sobre las claves pares del diccionario y mostrar las filas correspondientes
for caso in diccionario:
    if caso % 2 == 0:
        fila = diccionario[caso]
        print("Caso:", caso)
        print("Tasacion:", fila["Tasacion"])
        print("Valoracion:", fila["Valoracion"])
        print("Entradas:", fila["Entradas"])
        print("Balance:", fila["Balance"])
        print("Aprobar:", fila["Aprobar"])
        print("----------------------")
