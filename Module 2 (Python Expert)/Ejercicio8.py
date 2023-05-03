def calcular_media_y_varianza(a, b, c):
   
    # Calcular la media
    media = (a + b + c) / 3
    
    # Calcular la varianza
    varianza = ((a - media) ** 2 + (b - media) ** 2 + (c - media) ** 2) / 3
    
    return media, varianza
print(calcular_media_y_varianza(2, 4, 6))