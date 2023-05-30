
n=0
m=0
L=[]

def ejercicio9():
    n= int(input("introduce n: "))
    m=int(input("introduce m: "))
    for i in range(n):
        L.append([])
        for f in range(m):
            L[i].append(f)
    return L

ejercicio9()
print(L)