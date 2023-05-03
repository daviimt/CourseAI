def es_primo(numero):
    
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
print(es_primo(7))
print(es_primo(15))
print(es_primo(2))
print(es_primo(1))
print(es_primo(73))
