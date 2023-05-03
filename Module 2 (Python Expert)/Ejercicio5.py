def es_primo(numero):
    """
    Esta función recibe un número entero positivo como entrada y devuelve True si el número
    es primo, y False en caso contrario.
    """
    # Los números menores o iguales a 1 no son primos
    if numero <= 1:
        return False
    # Los números 2 y 3 son primos
    elif numero <= 3:
        return True
    # Los números pares mayores que 2 no son primos
    elif numero % 2 == 0:
        return False
    # Verificar si el número es divisible por algún impar mayor que 3
    i = 3
    while i <= int(numero**0.5) + 1:
        if numero % i == 0:
            return False
        i += 2
    return True


numero = int(input("Introduce un número entero positivo: "))
if es_primo(numero):
    print(f"{numero} es un número primo.")
else:
    print(f"{numero} no es un número primo.")
