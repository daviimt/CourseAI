def multiplicacion_division(num1, num2):
    """
    Esta función recibe dos números como entrada y devuelve su multiplicación
    y división (si es posible) como una tupla.
    """
    multiplicacion = num1 * num2
    division = None
    
    if num2 != 0:
        division = num1 / num2
    
    return (multiplicacion, division)

print(multiplicacion_division(5, 2))
print(multiplicacion_division(5, 0))
