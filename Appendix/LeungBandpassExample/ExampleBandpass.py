import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors

custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", 
                                                                  ["white",  "white", 
                                                                   "red", "red", "red", "red", "red", 
                                                                   "yellow", 
                                                                   "lightgreen", "lightgreen", "lightgreen", "lightgreen", 
                                                                   "blue", "darkblue"])

# Unfiltered velocity field
def VF(Omega, r, z, l):
    return Omega * r * np.exp(-2 * (r**2 + z**2)/(l**2))
    # This is in the theta direction

# Bandpass filtered velocity field
def BFVF(L, l, h, Omega, r, z):
    return ((L**2 * l**5)/(4 * pow(h, 7) * np.sqrt(L))) * Omega * r * np.exp(-1 * (r**2 + z**2)/(h**2)) * (5 - (2 * (r**2 + z**2))/(h**2))
    # This is in the theta direction

def h(L, l):
    return np.sqrt(L**2 + l**2/2)

def generate_mesh():
    r = np.arange(0, 1.5 + 0.01, 0.01)
    theta = np.arange(0, 2 * np.pi, 0.01)
    rmesh, thetamesh = np.meshgrid(r, theta)
    return rmesh, thetamesh

def graph(thetamesh, rmesh, function, title, row, col):
    plt.subplots(dpi = 70, subplot_kw=dict(projection='polar'))
    plt.contourf(thetamesh, rmesh, function, 100, cmap = custom_cmap)
    plt.axis('off')
    plt.title(title, fontsize = 20, y = -0.1)
    plt.savefig(f"{title}.png")

# Plane cut normal to z-axis, so try z = 0
z = 0.01

ell1 = 1
ell2 = 0.1 * ell1
L = 1.2 * ell1
Omega1 = 5.5
Omega2 = 10 * Omega1
rmesh, thetamesh = generate_mesh()

u0_1 = VF(Omega1, rmesh, z, ell1)
u0_2 = VF(Omega2, rmesh, z, ell2)
u0 = u0_1 + u0_2

# Unfiltered Plot
fig, ax = plt.subplots(dpi = 70, subplot_kw=dict(projection='polar'))
ax = plt.contourf(thetamesh, rmesh, u0, 100, cmap = custom_cmap)
plt.axis('off')
plt.colorbar()
plt.title("Unfiltered L = 1.2 * ell1", fontsize = 20, y = -0.1)
plt.savefig(f"(a) Unfiltered L = 1.2 * ell1.png")

# Filtered Plot (b)

L = ell1

uBL_1 = BFVF(L, ell1, h(L, ell1), Omega1, rmesh, z)
uBL_2 = BFVF(L, ell2, h(L, ell2), Omega2, rmesh, z)
uBL = uBL_1 + uBL_2

graph(thetamesh, rmesh, uBL, "(b) L = ell1", 0, 0)

# Filtered Plot (c)

L = 0.5 * ell1

uBL_1 = BFVF(L, ell1, h(L, ell1), Omega1, rmesh, z)
uBL_2 = BFVF(L, ell2, h(L, ell2), Omega2, rmesh, z)
uBL = uBL_1 + uBL_2

graph(thetamesh, rmesh, uBL, "(c) L = 0.5 * ell1", 0, 1)

# Filtered Plot (d)

L = 0.2 * ell1

uBL_1 = BFVF(L, ell1, h(L, ell1), Omega1, rmesh, z)
uBL_2 = BFVF(L, ell2, h(L, ell2), Omega2, rmesh, z)
uBL = uBL_1 + uBL_2

graph(thetamesh, rmesh, uBL, "(d) L = 0.2 * ell1", 0, 2)

# Filtered Plot (e)

L = 1.2 * ell2

uBL_1 = BFVF(L, ell1, h(L, ell1), Omega1, rmesh, z)
uBL_2 = BFVF(L, ell2, h(L, ell2), Omega2, rmesh, z)
uBL = uBL_1 + uBL_2

graph(thetamesh, rmesh, uBL, "(e) L = 1.2 * ell2", 0, 3)

# Filtered Plot (f)

L = ell2

uBL_1 = BFVF(L, ell1, h(L, ell1), Omega1, rmesh, z)
uBL_2 = BFVF(L, ell2, h(L, ell2), Omega2, rmesh, z)
uBL = uBL_1 + uBL_2

graph(thetamesh, rmesh, uBL, "(f) L = ell2", 1, 0)

# Filtered Plot (g)

L = 0.5 * ell2

uBL_1 = BFVF(L, ell1, h(L, ell1), Omega1, rmesh, z)
uBL_2 = BFVF(L, ell2, h(L, ell2), Omega2, rmesh, z)
uBL = uBL_1 + uBL_2

graph(thetamesh, rmesh, uBL, "(g) L = 0.5 * ell2", 1, 1)

# Filtered Plot (h)

L = 0.2 * ell2

uBL_1 = BFVF(L, ell1, h(L, ell1), Omega1, rmesh, z)
uBL_2 = BFVF(L, ell2, h(L, ell2), Omega2, rmesh, z)
uBL = uBL_1 + uBL_2

graph(thetamesh, rmesh, uBL, "(h) L = 0.2 * ell2", 1, 2)

plt.show()