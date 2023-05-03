num1 = float(input("Introduce el primer número: "))
num2 = float(input("Introduce el segundo número: "))
num3 = float(input("Introduce el tercer número: "))

maximo = num1
minimo = num1

if num2 > maximo:
    maximo = num2
if num3 > maximo:
    maximo = num3

if num2 < minimo:
    minimo = num2
if num3 < minimo:
    minimo = num3

print(f"El máximo es {maximo}")
print(f"El mínimo es {minimo}")
