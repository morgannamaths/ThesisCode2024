import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

Re = np.linspace(10, 10000, 20)

# Laminar example
drag_laminar = 24/Re + 4/np.sqrt(Re) + 0.4

# Turbulent example
drag_turbulent = 0.027/np.power(Re, 1/7)

# NASA example
D = np.linspace(50, 10000, 20)
I = 1
V = 1
mu = 0.01
Cd = (2 * D * I)/(Re * V * mu)

# Steve Brunton video example
reynolds = [10, 15, 20, 40, 50, 80, 130, 150, 250, 500, 800, 1000, 1200, 1500, 3000, 9000, 10000, 15000, 20000, 150000, 175000, 300000, 400000, 500000, 600000, 2000000, 3000000, 3500000]
drag = [2.5, 2.2, 2, 1.65, 1.58, 1.5, 1.37, 1.35, 1.25, 1.1, 1, 0.95, 0.92, 0.9, 0.93, 1.16, 1.2, 1.24, 1.25, 1.25, 1.23, 0.5, 0.42, 0.4, 0.41, 0.6, 0.8, 0.85]

def mean_interp(reynolds, drag):
    re_new = []
    drag_new = []
    for i in range(len(reynolds) - 1):
        re_new.append(reynolds[i])
        drag_new.append(drag[i])
        re_new.append((reynolds[i] + reynolds[i+1])/2)
        drag_new.append((drag[i] + drag[i+1])/2)
    re_new.append(reynolds[len(reynolds)-1])
    drag_new.append(drag[len(drag)-1])
    return re_new, drag_new

for j in range(10):
    reynolds, drag = mean_interp(reynolds, drag)
    print(f"Done iteration {j + 1}")

xlabel1 = [20, 20]
label1 = [0, 1.95]

xlabel2 = [150, 150]
label2 = [0, 1.35]

xlabel3 = [150000, 150000]
label3 = [0, 1.25]

plt.figure(figsize=(9,6))
plt.plot(reynolds, drag)
plt.plot(xlabel1, label1, "--", color="black")
plt.plot(xlabel2, label2, "--", color="black")
plt.plot(xlabel3, label3, "--", color="black")
plt.text(6, 1, "Steady")
plt.text(8, 0.9, "Flow")
plt.text(6, 0.7, r"$\frac{\partial \hat{u}}{\partial t} = 0$")
plt.text(25, 0.8, "Laminar")
plt.text(30, 0.7, "Flow")
plt.text(1000, 0.7, "Turbulent Flow")
plt.text(900000, 0.3, "Turbulent")
plt.text(500000, 0.2, "Boundary Layer")
plt.xscale("log")
plt.xlabel("Reynold's Number", fontsize=16)
plt.ylabel("Drag", fontsize=16)
plt.title("Reynold's Number vs Drag", fontsize=20)
plt.minorticks_off()
plt.ylim(0, 3)
plt.xlim(0, 10000000)
plt.show()