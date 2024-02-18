import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
from datetime import datetime
from mpl_toolkits import mplot3d

date = datetime.now()

custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", 
                                                                  ["white",  "white", 
                                                                   "red", "red", "red", "red", "red", 
                                                                   "yellow", 
                                                                   "lightgreen", "lightgreen", "lightgreen", "lightgreen", 
                                                                   "blue", "darkblue"])

# Unfiltered Velocity Field
def Unfiltered(Omega, r, z, ellr, ellz):
    return Omega * r * np.exp(-2 * r**2/(ellr**2) - 2 * z**2/(ellz**2))

def hr(L, ellr):
    return L**2 + ellr**2/2

def hz(L, ellz):
    return L**2 + ellz**2/2

# Bandpass Filtered Velocity Field
def BFVF(Omega, r, z, ellr, ellz, L, hr, hz):
    return (1/np.sqrt(L)) * (L**2 * ellr**4 * ellz)/(2 * hr**4 * hz) * Omega * r * np.exp(-1 * r**2/(hr**2) - z**2/(hz**2)) * ((2 - r**2/(hr**2))/(hr**2) + (0.5 - z**2/(hz**2))/(hz**2))

def generate_mesh():
    r = np.arange(0, 1.5 + 0.01, 0.01)
    theta = np.arange(0, 2 * np.pi, 0.01)
    rmesh, thetamesh = np.meshgrid(r, theta)
    return rmesh, thetamesh

def graph(theta, r, z, field, colour, title):
    fig, ax = plt.subplots(dpi = 70, subplot_kw=dict(projection='polar'))
    ax = plt.contourf(theta, r, field, 100, cmap = colour)
    plt.axis('off')
    plt.colorbar()
    plt.title(title)

ellr = 1
ellz = 0.1 * ellr
L = 0.5 * ellr
Omega1 = 1
Omega2 = 10 * Omega1

rmesh, thetamesh = generate_mesh()
z = 0.1

u0_1 = Unfiltered(Omega1, rmesh, z, ellr, ellz)
u0_2 = Unfiltered(Omega2, rmesh, z, ellr, ellz)

u0 = u0_1 + u0_2

title = f"Unfiltered: ellr = {ellr}, L = 0.5 * ellr, ellz = 0.1 * ellr, z = {z}, OmegaSmall = {Omega1}, OmegaLarge = 10 * OmegaSmall, Date: {date}"

# Unfiltered Plot
graph(thetamesh, rmesh, z, u0, custom_cmap, title)

# Filtered Plot
uF_1 = BFVF(Omega1, rmesh, z, ellr, ellz, L, hr(L, ellr), hz(L, ellz))
uF_2 = BFVF(Omega2, rmesh, z, ellr, ellz, L, hr(L, ellr), hz(L, ellz))

uF = uF_1 + uF_2

title2 = f"Bandpass Filtered: ellr = {ellr}, L = 0.5 * ellr, ellz = 0.1 * ellr, z = {z}, OmegaSmall = {Omega1}, OmegaLarge = 10 * OmegaSmall, Date: {date}"

graph(thetamesh, rmesh, z, uF, custom_cmap, title2)

# Try doing some 3d surface plot to analyse the shape of the fields better

X, Y = rmesh * np.cos(thetamesh), rmesh * np.sin(thetamesh)

fig, ax = plt.subplots(dpi = 70, subplot_kw=dict(projection='3d'))
ax.plot_surface(X, Y, u0, cmap = "rainbow")
plt.title(f"Unfiltered u")

fig, ax = plt.subplots(dpi = 70, subplot_kw=dict(projection='3d'))
ax.plot_surface(X, Y, uF, cmap = "rainbow")
plt.title(f"Filtered u")

# Also consider numerical solution

plt.show()