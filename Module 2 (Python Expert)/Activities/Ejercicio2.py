num1 = int(input("Introduce el primer número: "))
num2 = int(input("Introduce el segundo número: "))

if num1 % num2 == 0:
    print(f"{num2} divide a {num1}")
elif num2 % num1 == 0:
    print(f"{num1} divide a {num2}")
else:
    print("Los números no son divisibles entre sí")
