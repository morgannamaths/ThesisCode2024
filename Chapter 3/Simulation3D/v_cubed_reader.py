import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfftn, irfftn, rfft2, irfft2

f = h5py.File("v32cubed.h5", "r")
#print(list(f.keys())) gives "mhd3d-mpi"
data = f["mhd3d-mpi"]
#print(list(data.keys())) gives "vx", "vy", "vz"
ux = data["vx"][:]
uy = data["vy"][:]
uz = data["vz"][:]

print(f"ux is shape {ux.shape}")
print(f"uy is shape {uy.shape}")
print(f"uz is shape {uz.shape} \n")

energy = 0
for i in range(32):
    for j in range(32):
        for m in range(32):
            energy += ux[i][j][m]**2 + uy[i][j][m]**2 + uz[i][j][m]**2
print(energy/(32*32*32))
print(f"{energy/(32*32*32)/2} \n")

vsqd = ux[0:31][0:31][0:31]**2 + uy[0:31][0:31][0:31]**2 + uz[0:31][0:31][0:31]**2
print(vsqd.shape)
en = np.average(vsqd)

Eb_k = [0.000000E+00, 4.506729E-04, 3.030616E-03, 4.223417E-02, 5.105386E-02, 3.980224E-02, 2.793994E-02, 2.243190E-02, 1.575576E-02, 1.405151E-02, 1.032330E-02, 7.129760E-03, 6.601347E-03, 5.975230E-03, 5.131001E-03, 4.803499E-03, 5.072352E-03, 4.215255E-03, 3.160718E-03, 2.276307E-03, 1.800095E-03, 1.309867E-03, 7.448536E-04, 4.256951E-04, 1.884412E-04, 1.160764E-04, 2.221008E-05, 1.374823E-05, 9.796379E-08]

k_mag = np.arange(0, len(Eb_k), 1)

plt.plot(k_mag, Eb_k)
plt.title("Energy Spectrum from spectra.dat for time = 1", fontsize = 20)
plt.xlabel(r"$|\vec{k}|$", fontsize = 16)
plt.ylabel(r"$E_{omni}(\vec{k})$", fontsize = 16)
plt.xscale("log")
plt.yscale("log")
plt.show()
