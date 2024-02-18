import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfftn, irfftn
from mpl_toolkits import mplot3d
from datetime import datetime

date = datetime.now()

N = 128
x = y = np.linspace(-1 * np.pi, np.pi - np.pi/N, N)
z = 0

xmesh, ymesh = np.meshgrid(x, y)

Omega = 1
ell = np.pi/6
L = 1.2 * ell
alpha = np.sqrt(2)

k_x = np.arange(0, N/2 + 1, 1)              # k_x >= 0
k_y_top = np.arange(0, N/2 + 1, 1)          # Goes to N/2 + 1 since it doesn't store final point, so (0, 1, ..., N/2)
k_y_bot = np.arange(-1 * N/2 + 1, 0, 1)     # Goes to 0 since it doesn't store final point, so (-N/2 + 1, ..., -2, -1)
k_y = np.concatenate((k_y_top, k_y_bot))    # Concatenates to create (0, 1, ..., N/2, -N/2 + 1, ..., -2, -1)
kxmesh, kymesh = np.meshgrid(k_x, k_y)
k = np.sqrt(kxmesh**2 + kymesh**2)

theta_ij = np.arctan2(ymesh,xmesh)          # Divides each element in ymesh by each element in xmesh, with a 2D arctan

r = np.sqrt(xmesh**2 + ymesh**2)
kap = k * L/2
T = 2 * kap**2 * np.exp(-1 * kap**2)

xhat = -1 * np.sin(theta_ij)
yhat = np.cos(theta_ij)

func = Omega * r * np.exp(-2 * ((r**2 + z**2)/(ell**2)))    # Base Function to avoid making Python recompute it several times... also works as colourmap

u_x = func * xhat
u_y = func * yhat

'''
ax = plt.axes(projection="3d")
ax.plot_surface(xmesh, ymesh, u_y, cmap = "rainbow")
plt.title(f"Unfiltered u_y, ell = pi/6, L = 1.2 * ell, N = {N}")
'''

plt.figure(figsize=(8,6))
plt.quiver(xmesh, ymesh, u_x, u_y, func)
plt.title(f"Unfiltered Velocity Field, for N = {N}^2", fontsize=20)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.ylim((-1.25, 1.25))
plt.xlim((-1.25, 1.25))
# This gives a whirlpool, as expected!!!

# Numerical Bandpass Filter

u_x_rfft = rfftn(u_x, s=None, axes=(0, 1), norm = "forward")
u_y_rfft = rfftn(u_y, s=None, axes=(0, 1), norm = "forward")

ubl_x_rfft = alpha/np.sqrt(L) * T * u_x_rfft
ubl_y_rfft = alpha/np.sqrt(L) * T * u_y_rfft

ubl_x = irfftn(ubl_x_rfft, s=None, axes=(0, 1), norm = "forward")
ubl_y = irfftn(ubl_y_rfft, s=None, axes=(0, 1), norm = "forward")

plt.figure(figsize=(8,6))
plt.quiver(xmesh, ymesh, ubl_x, ubl_y, func)
plt.title(f"Filtered (Numerical) Velocity Field, for N = {N}^2", fontsize=20)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.ylim((-1.25, 1.25))
plt.xlim((-1.25, 1.25))

# Analytical Bandpass Filter

h = np.sqrt(L**2 + ell**2/2)

ubl_x_analytic = ((L**2 * ell**5)/(4 * pow(h, 7) * np.sqrt(L))) * Omega * r * np.exp(-1 * (r**2 + z**2)/(h**2)) * (5 - (2 * (r**2 + z**2))/(h**2)) * xhat
ubl_y_analytic = ((L**2 * ell**5)/(4 * pow(h, 7) * np.sqrt(L))) * Omega * r * np.exp(-1 * (r**2 + z**2)/(h**2)) * (5 - (2 * (r**2 + z**2))/(h**2)) * yhat

plt.figure(figsize=(8,6))
plt.quiver(xmesh, ymesh, ubl_x_analytic, ubl_y_analytic, func)
plt.title(f"Filtered (Analytical) Velocity Field, for N = {N}^2", fontsize=20)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.ylim((-1.25, 1.25))
plt.xlim((-1.25, 1.25))

plt.show()
