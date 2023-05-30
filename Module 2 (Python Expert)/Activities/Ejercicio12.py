def multiplicacion_division(num1, num2):
  
    multiplicacion = num1 * num2
    division = None
    
    if num2 != 0:
        division = num1 / num2
    
    return (multiplicacion, division)

print(multiplicacion_division(5, 2))
print(multiplicacion_division(5, 0))
