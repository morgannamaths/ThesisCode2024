import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from parameters import Parameters
from operations import Operations
from bandpass import BandpassFilter
from scipy.integrate import simps

def main():
    # Import parameters and fields
    params = Parameters("v32cubed.h5")
    limit = 2 * np.pi
    ux, uy, uz = params.fields()
    N = params.setup()
    #print(N)    # Check if single value for N or 3 values for nx, ny, nz
    alpha = params.alpha
    visc = params.visc
    x, y, z, kx, ky, kz = params.space(limit)
    k_magnitude = np.arange(0, N, 1)
    uxhat, uyhat, uzhat = Operations().rfft_ufield(ux, uy, uz)

    Kx, Ky, Kz = np.meshgrid(kx, ky, kz) #removed ij to maintain y,x,z format, otherwise output is rotated

    u_sqd = ux**2 + uy**2 + uz**2
    u_av = np.average(u_sqd)
    L_character = limit/N
    Reynolds = u_av * L_character/visc
    print(f"The Reynolds number for this flow is {Reynolds}")

    omega = Operations().curl3D(uxhat, uyhat, uzhat, Kx, Ky, Kz)
    omega_x = omega[0]
    omega_y = omega[1]
    omega_z = omega[2]

    E_Omni = Operations().Energy_Spectrum_Omni(N, uxhat, uyhat, uzhat, kx, ky, kz)

    l_corr = Operations().l_corr(E_Omni, k_magnitude)
    print(f"Correlation length is {l_corr}")

    eta = Operations().Kolmogorov(visc, k_magnitude, E_Omni)
    print(f"Kolomogorov microscale is {eta}")

    # Redundant nodes and sheets

    # For large ell, use l_corr
    # For small ell, use Kolmogorov scale aka eta

    data_points = 64

    Pi_array_1 = []
    F_array_1 = []
    Pi_array_2 = []
    F_array_2 = []
    Pi_array_3 = []
    F_array_3 = []

    S_over_L = np.linspace(eta, 1, data_points)   # Avoids division by zero

    scale = [4, 5, 10]
    l_large = [scale[0] * eta, scale[1] * eta, scale[2] * eta]

    # For loop
    for i in range(len(l_large)):
        for element in S_over_L:
            l_small = element * l_large[i]
            # Bandpass coeffs
            T_large, T_small = BandpassFilter().coeffs(k_magnitude, l_large[i], l_small)

            # Transfer of energy from large to small scales
            Pi_LtS = BandpassFilter().Pi_LtS(uxhat, uyhat, uzhat, Kx, Ky, Kz, l_large[i], l_small, alpha, T_large, T_small)
            if i == 0:
                Pi_array_1.append(Pi_LtS)
            if i == 1:
                Pi_array_2.append(Pi_LtS)
            if i == 2:
                Pi_array_3.append(Pi_LtS)

            # Transfer of enstrophy from large to small scales
            F_LtS = BandpassFilter().F_LtS(ux, uy, uz, omega_x, omega_y, omega_z, Kx, Ky, Kz, l_large[i], l_small, alpha, T_large, T_small)
            if i == 0:
                F_array_1.append(F_LtS)
            if i == 1:
                F_array_2.append(F_LtS)
            if i == 2:
                F_array_3.append(F_LtS)
    
    # Normalise arrays
    Pi_array_1 = Pi_array_1/max(map(abs, Pi_array_1))
    Pi_array_2 = Pi_array_2/max(map(abs, Pi_array_2))
    Pi_array_3 = Pi_array_3/max(map(abs, Pi_array_3))
    F_array_1 = F_array_1/max(map(abs, F_array_1))
    F_array_2 = F_array_2/max(map(abs, F_array_2))
    F_array_3 = F_array_3/max(map(abs, F_array_3))

    # Plot
    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, (Pi_array_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, (Pi_array_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, (Pi_array_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    plt.title("Normalised transfer of energy from large to small scales", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{\Pi}_{V, b}^{r}$", fontsize=16)
    plt.legend()
    plt.savefig("3DEnergy.png")

    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, (F_array_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, (F_array_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, (F_array_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    plt.title("Normalised transfer of enstrophy from large to small scales", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{F}_{b}^{r}$", fontsize=16)
    plt.legend()
    plt.savefig("3DEnstrophy.png")

    plt.show()

if __name__ == "__main__":
    main()
