suma = 0
contador = 0
num = int(input("Introduce un número entero positivo (0 para terminar): "))
while num != 0 and contador < 10:
    if num > 0:
        suma += num
        contador += 1
    num = int(input("Introduce un número entero positivo (0 para terminar): "))

if contador > 0:
    media = suma / contador
    print(f"La media de los números positivos introducidos es {media:.2f}")
else:
    print("No se ha introducido ningún número positivo.")
