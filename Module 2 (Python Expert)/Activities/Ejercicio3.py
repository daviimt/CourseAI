for i in range(1001):
    ai = ((-1)**i * (i**2 - 1)) / (2**i)
    if ai % 2 == 0:
        print(f"a{i} = {ai}")
