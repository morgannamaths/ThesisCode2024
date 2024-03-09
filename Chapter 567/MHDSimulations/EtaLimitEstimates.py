import math

N = [32, 256, 512, 2048]

for element in N:
    print(f"For {element} dataset, eta limit is {round((element/2)/1.5, 2)}. Multiples of eta should not exceed {math.floor((element/2)/1.5)}.")