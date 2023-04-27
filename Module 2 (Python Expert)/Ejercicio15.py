D={
    'A':'',
    'B':'',
    'C':'',
    'D':'',
    'F':'',
    'G':'',
    'H':'',
    'I':'',
    'J':'',
    'K':'',
    'L':'',
    'M':'',
    'N':'',
    'O':'',
    'P':'',
    'Q':'',
    'R':'',
    'S':'',
    'T':'',
    'U':'',
    'V':'',
    'W':'',
    'X':'',
    'Y':'',
    'Z':''
}

cadena=input()

salida=[]
for letra in cadena:
    salida.append(D[letra])

print('El codigo morse de la salida es ',salida)