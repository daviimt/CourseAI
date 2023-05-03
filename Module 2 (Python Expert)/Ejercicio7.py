def pedir_valor_en_rango(minimo, maximo):
    """
    Esta función recibe dos números como entrada, mínimo y máximo, y pide al usuario
    introducir un valor en ese rango. Devuelve el valor introducido (previa comprobación
    de que es válido).
    """
    # Pedir al usuario que introduzca un valor en el rango
    while True:
        valor = input(f"Introduzca un valor entre {minimo} y {maximo}: ")
        try:
            valor = float(valor)
        except ValueError:
            print("Debe introducir un valor numérico.")
            continue
        if minimo <= valor <= maximo:
            return valor
        else:
            print(f"El valor debe estar entre {minimo} y {maximo}.")

print(pedir_valor_en_rango(0, 10))
