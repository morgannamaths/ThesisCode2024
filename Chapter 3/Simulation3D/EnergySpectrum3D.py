import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfftn, irfftn
from operations import Operations

def import_files(filename):
    f = h5py.File(filename, "r")
    #print(list(f.keys())) gives "mhd3d-mpi"
    data = f["mhd3d-mpi"]
    #print(list(data.keys())) gives "vx", "vy", "vz"
    vx = data["vx"][:]
    vy = data["vy"][:]
    vz = data["vz"][:]

    # Make data (32, 32, 32)
    ux = np.delete(vx, [32, 33], 2)
    uy = np.delete(vy, [32, 33], 2)
    uz = np.delete(vz, [32, 33], 2)
    return ux, uy, uz

def Energy(ux, uy, uz, N):
    energy = 0
    for i in range(N):
        for j in range(N):
            for m in range(N):
                energy += ux[i][j][m]**2 + uy[i][j][m]**2 + uz[i][j][m]**2
    energy = energy/(N**3 * 2)
    return energy

def setup(N):
    ky = kz = np.fft.fftfreq(N, d = 1/N)
    kx = np.fft.rfftfreq(N, d = 1/N)    # Chopping kx in half
    kz[int(N/2)] = N/2
    print(kx)   # Confirmed to be float (avoids int overflow for large data)
    return kx, ky, kz

def Energy_Spectrum_Modal(uxhat, uyhat, uzhat):
    return np.abs(uxhat)**2 + np.abs(uyhat)**2  + np.abs(uzhat)**2 # E = |uhat|^2

def Conjugation(uxhat, uyhat, uzhat, N):
    # Checking way of writing conjugation
    '''print(uxhat[0, 5])
    print(np.conj(uxhat[N/2, 5]))   # Works
    print(np.conj(uxhat)[N/2, 5])   # Only works for typed-in integers, not integer variables'''

    #print(uxhat[int(N/2), 3])              # Note N/2 is a float, even if result is an integer
    testuxhat = uxhat    # Testing conjugation
    testuyhat = uyhat    # Testing conjugation
    testuzhat = uzhat    # Testing conjugation
    for i in np.arange(1, int(N/2), 1, dtype = int):        # Goes from 0 < ky < N/2, or 1 <= ky <= N/2-1
        uxhat[0, i, 0] = np.conj(uxhat[0, i, 0])
        uxhat[0, i, int(N/2)] = np.conj(uxhat[0, i, int(N/2)])
        uxhat[int(N/2), i, 0] = np.conj(uxhat[int(N/2), i, 0])
        uxhat[int(N/2), i, int(N/2)] = np.conj(uxhat[int(N/2), i, int(N/2)])

        uyhat[0, i, 0] = np.conj(uxhat[0, i, 0])
        uyhat[0, i, int(N/2)] = np.conj(uxhat[0, i, int(N/2)])
        uyhat[int(N/2), i, 0] = np.conj(uxhat[int(N/2), i, 0])
        uyhat[int(N/2), i, int(N/2)] = np.conj(uxhat[int(N/2), i, int(N/2)])

        uzhat[0, i, 0] = np.conj(uxhat[0, i, 0])
        uzhat[0, i, int(N/2)] = np.conj(uxhat[0, i, int(N/2)])
        uzhat[int(N/2), i, 0] = np.conj(uxhat[int(N/2), i, 0])
        uzhat[int(N/2), i, int(N/2)] = np.conj(uxhat[int(N/2), i, int(N/2)])

    print(uxhat is testuxhat)
    print(uyhat is testuyhat)
    print(uzhat is testuzhat)

def efficient_abs_sqd(x):
    return np.real(x)**2 + np.imag(x)**2

def Energy_Spectrum_Omni(N, kx, ky, kz, uxhat, uyhat, uzhat):
    E_Omni = np.zeros(N)
    for m in range(N-1):
        kz_val = kz[m]
        for j in range(N-1):
            ky_val = ky[j]
            for i in range(int(N/2)):
                kx_val = kx[i]
                k_sqd = kx_val**2 + ky_val**2 + kz_val**2
                k_index = int(round(np.sqrt(k_sqd)))
                E_Omni[k_index] += efficient_abs_sqd(uxhat[j][i][m]) + efficient_abs_sqd(uyhat[j][i][m]) + efficient_abs_sqd(uzhat[j][i][m])
    #print(E_Omni)
    return E_Omni

def logplot(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title(title, fontsize = 20)
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(50E-9, 10E-2)

# Set up domain
ux, uy, uz = import_files("v32cubed.h5")
N = int(ux.shape[0])     # ux is (32, 32, 34) so ux.shape[0] returns 32
print(N)
print(Energy(ux, uy, uz, N))
kx, ky, kz = setup(N)

# Fourier Transformed Velocity Fields
uxhat = np.fft.rfftn(ux, s=None, axes=(0,2,1), norm = "forward")
uyhat = np.fft.rfftn(uy, s=None, axes=(0,2,1), norm = "forward")
uzhat = np.fft.rfftn(uz, s=None, axes=(0,2,1), norm = "forward")
print(uxhat.shape)
print(ux.ndim)
print(uxhat.ndim)

# Check conjugation of uxhat and uyhat
Conjugation(uxhat, uyhat, uzhat, N)

# E_Omni energy spectrum
E_Omni = Energy_Spectrum_Omni(N, kx, ky, kz, uxhat, uyhat, uzhat)
k_magnitude = np.arange(0, N, 1)

'''E_Omni = [2.83588012e-36, 2.10031387e-04, 2.97629160e-03, 2.06371652e-02,
 3.19769124e-02, 1.96163090e-02, 1.26207295e-02, 9.72648506e-03,
 7.42764672e-03, 5.91667436e-03, 4.25932649e-03, 3.41427334e-03,
 2.83740206e-03, 2.59220943e-03, 2.30713339e-03, 2.11048230e-03,
 2.21671916e-03, 1.77944333e-03, 1.20709482e-03, 8.64684512e-04,
 6.65734342e-04, 4.60520056e-04, 2.31945300e-04, 1.31886105e-04,
 5.63024409e-05, 2.41320492e-05, 3.35758794e-06, 2.21298063e-06,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]'''

logplot(k_magnitude, E_Omni, f"Omnidirectional Energy Spectrum for N = {N}", r"$|\vec{k}|$", r"$E_{omni}(\vec{k})$")

# Taylor Microscale lambda = 5 E/Omega = 5 <v^2>/<omega^2>
'''v_sqd = ux**2 + uy**2 + uz**2
v_av = np.average(v_sqd)

omega_av = np.average(omega**2)
taylor = 5 * v_av/omega_av

print(taylor)'''

# Integral Length Scale l_corr
# Integral is essentially a for-loop summmation
int_E = 0
for i in range(len(k_magnitude)):
    int_E += E_Omni[i]
#print(int_E)
int_kinv_E = 0
for i in range(1, len(k_magnitude)):
        int_kinv_E += E_Omni[i]/k_magnitude[i]
#print(int_kinv_E)
l_corr = 3 * (np.pi/4) * int_kinv_E/int_E
print(f"l_corr = {l_corr}")   # For 32x32 case, this lines up with the value I approximated in guess-and-check code

#plt.plot([1/l_corr, 1/l_corr], [0, 3 * 10**(-3)], "--")
#plt.text(1/l_corr+0.1, 10**(-7), r"$\frac{1}{l_{corr}}$", fontsize=14)
plt.text(1, 1e-05, r"Eddies depend on $\vec{u}$ and $L$", fontsize=12, color="darkorange")

eta = Operations().Kolmogorov(0.01, k_magnitude, E_Omni)
print(f"Kolomogorov microscale is {eta}")

#plt.plot([1/eta, 1/eta], [0, 5 * 10**(-4)], "--")
#plt.text(1/eta+1, 10**(-7), r"$\frac{1}{\eta}$", fontsize=14)
plt.text(16, 1e-05, r"Eddies depend", fontsize=12, color="darkgreen")
plt.text(18, 6e-06, r"on $\nu$", fontsize=12, color="darkgreen")

plt.plot([4, 4], [0, 3e-02], "--", color="black")
plt.plot([15, 15], [0, 2e-03], "--", color="black")
plt.text(5, 1e-05, "Inertial Subrange", fontsize=12, color="black")

plt.plot(k_magnitude, 0.25 * (k_magnitude)**(-5/3), "--")
plt.text(10, 10**(-2), r"$\propto k^{-5/3}$", fontsize=14, color="red")

plt.show()