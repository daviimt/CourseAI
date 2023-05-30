import statistics

def estadisticas(lista):
    minimo = min(lista)
    maximo = max(lista)
    media = statistics.mean(lista)
    varianza = statistics.variance(lista)
    desviacion_tipica = statistics.stdev(lista)
    return (minimo, maximo, media, varianza, desviacion_tipica)

lista = [1, 2, 3, 4, 5, 6]

print(estadisticas(lista))