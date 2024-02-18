import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from parameters import Parameters
from parameters2D import Parameters2D
from operations import Operations
from bandpass import BandpassFilter
from Get2DArrays import Get2DArrays

def main3D():
    # Import parameters and fields
    params = Parameters("mhd3d-32/r1-mhd3-01-uno-Xsp-v.h5", "mhd3d-32/r1-mhd3-01-uno-Xsp-b.h5")
    #params = Parameters("mhd3d-n256/mhd3-00-uno-Xsp-v.h5", "mhd3d-n256/mhd3-00-uno-Xsp-b.h5")
    print("Parameters obtained")
    limit = 2 * np.pi
    ux, uy, uz, Bx, By, Bz = params.fields()
    dims = ux.ndim
    N = params.setup()
    print(N)    # Check if single value for N or 3 values for nx, ny, nz
    alpha = params.alpha
    time = params.time
    visc = params.visc
    resist = params.resist
    x, y, z, kx, ky, kz = params.space(limit)
    k_magnitude = np.arange(0, N, 1)
    uxhat, uyhat, uzhat, Bxhat, Byhat, Bzhat = Operations().rfft_field3D(ux, uy, uz, Bx, By, Bz)
    print("Found uhat and Bhat")

    Kx, Ky, Kz = np.meshgrid(kx, ky, kz) #removed ij to maintain y,x,z format, otherwise output is rotated
    K = np.sqrt(Kx**2 + Ky**2 + Kz**2)
    print("Done K")
    
    a_x_hat, a_y_hat, a_z_hat = Operations().anticurl3D(Bxhat, Byhat, Bzhat, Kx, Ky, Kz)
    print("Found ahat")

    omega = Operations().curl3D(uxhat, uyhat, uzhat, Kx, Ky, Kz)
    omega_x = omega[0]
    omega_y = omega[1]
    omega_z = omega[2]

    omega_x_hat = Operations().rfft(omega_x)
    omega_y_hat = Operations().rfft(omega_y)
    omega_z_hat = Operations().rfft(omega_z)
    print("omega and omega hat done")

    J = Operations().curl3D(Bxhat, Byhat, Bzhat, Kx, Ky, Kz)
    Jx = J[0]
    Jy = J[1]
    Jz = J[2]
    print("J done")

    E_Omni = Operations().Energy_Spectrum_Omni(N, uxhat, uyhat, uzhat, kx, ky, kz)

    #plt.figure(figsize=(9, 6))
    #plt.plot(k_magnitude, E_Omni)
    #plt.xscale("log")
    #plt.yscale("log")

    mu0 = 0.01

    l_corr = Operations().l_corr(E_Omni, k_magnitude)
    print(f"Correlation length is {l_corr}")

    eta = Operations().Kolmogorov(visc, k_magnitude, E_Omni)
    print(f"Kolomogorov microscale is {eta}")

    # Redundant nodes and sheets

    # For large ell, use l_corr
    # For small ell, use Kolmogorov scale aka eta

    data_points = 64

    # Kinetic Arrays
    Pi_array_1 = []
    F_array_1 = []
    Pi_array_2 = []
    F_array_2 = []
    Pi_array_3 = []
    F_array_3 = []
    #Pi_array_4 = []
    #F_array_4 = []

    # Magnetic Arrays
    Pi_b_array_1 = []
    F_b_array_1 = []
    Pi_b_array_2 = []
    F_b_array_2 = []
    Pi_b_array_3 = []
    F_b_array_3 = []
    #Pi_b_array_4 = []
    #F_b_array_4 = []

    # Total Energy Arrays
    Pi_T_1 = []
    Pi_T_2 = []
    Pi_T_3 = []

    S_over_L = np.linspace(eta, 1, data_points)   # Avoids division by zero

    scale = [75, 100, 150]
    l_large = [scale[0] * eta, scale[1] * eta, scale[2] * eta]

    print("Starting for-loop")
    # For loop
    j = 0
    for i in range(len(l_large)):
        for element in S_over_L:
            l_small = element * l_large[i]
            # Bandpass coeffs
            T_large, T_small = BandpassFilter().coeffs(K, l_large[i], l_small)

            # Filtering
            ux_bL, uy_bL, uz_bL, ux_bS, uy_bS, uz_bS, B_bL, B_bS, Sij_L, Sij_S = BandpassFilter().BandpassFilter(uxhat, uyhat, uzhat, Bxhat, Byhat, Bzhat, Kx, Ky, Kz, l_large[i], l_small, alpha, T_large, T_small)

            # Transfer of kinetic energy from large to small scales
            Pi_u = BandpassFilter().Pi_u(ux_bL, uy_bL, uz_bL, ux_bS, uy_bS, uz_bS, B_bL, B_bS, Sij_L, Sij_S, mu0)
            if i == 0:
                Pi_array_1.append(Pi_u)
            if i == 1:
                Pi_array_2.append(Pi_u)
            if i == 2:
                Pi_array_3.append(Pi_u)
            
            # Transer of magnetic energy
            Pi_b = BandpassFilter().Pi_b(ux, uy, uz, B_bL, B_bS, Sij_L, Sij_S, Kx, Ky, Kz)
            if i == 0:
                Pi_b_array_1.append(Pi_b)
                Pi_T_1.append(Pi_u + Pi_b)
            if i == 1:
                Pi_b_array_2.append(Pi_b)
                Pi_T_2.append(Pi_u + Pi_b)
            if i == 2:
                Pi_b_array_3.append(Pi_b)
                Pi_T_3.append(Pi_u + Pi_b)

            # Transfer of kinetic enstrophy from large to small scales
            F_u = BandpassFilter().F_u(ux, uy, uz, omega_x_hat, omega_y_hat, omega_z_hat, Jx, Jy, Jz, B_bL, B_bS, Kx, Ky, Kz, l_large[i], l_small, alpha, T_large, T_small)
            if i == 0:
                F_array_1.append(F_u)
            if i == 1:
                F_array_2.append(F_u)
            if i == 2:
                F_array_3.append(F_u)

            # Transfer of magnetic enstrophy
            F_b = BandpassFilter().F_b(ux, uy, uz, a_x_hat, a_y_hat, a_z_hat, Kx, Ky, Kz, l_large[i], l_small, alpha, T_large, T_small)
            if i == 0:
                F_b_array_1.append(F_b)
            if i == 1:
                F_b_array_2.append(F_b)
            if i == 2:
                F_b_array_3.append(F_b)
            
            j += 1
            print(f"{round(j/(len(S_over_L) * len(l_large))*100, 2)}% completed")
    
    print(f"Dimensions are {dims}D")
    # Normalise arrays
    Pi_array_1 = Pi_array_1/max(map(abs, Pi_array_1))
    Pi_array_2 = Pi_array_2/max(map(abs, Pi_array_2))
    Pi_array_3 = Pi_array_3/max(map(abs, Pi_array_3))
    #Pi_array_4 = Pi_array_4/max(map(abs, Pi_array_4))
    Pi_b_array_1 = Pi_b_array_1/max(map(abs, Pi_b_array_1))
    Pi_b_array_2 = Pi_b_array_2/max(map(abs, Pi_b_array_2))
    Pi_b_array_3 = Pi_b_array_3/max(map(abs, Pi_b_array_3))
    #Pi_b_array_4 = Pi_b_array_4/max(map(abs, Pi_b_array_4))
    F_array_1 = F_array_1/max(map(abs, F_array_1))
    F_array_2 = F_array_2/max(map(abs, F_array_2))
    F_array_3 = F_array_3/max(map(abs, F_array_3))
    #F_array_4 = F_array_4/max(map(abs, F_array_4))
    F_b_array_1 = F_b_array_1/max(map(abs, F_b_array_1))
    F_b_array_2 = F_b_array_2/max(map(abs, F_b_array_2))
    F_b_array_3 = F_b_array_3/max(map(abs, F_b_array_3))
    #F_b_array_4 = F_b_array_4/max(map(abs, F_b_array_4))
    Pi_T_1 = Pi_T_1/max(map(abs, Pi_T_1))
    Pi_T_2 = Pi_T_2/max(map(abs, Pi_T_2))
    Pi_T_3 = Pi_T_3/max(map(abs, Pi_T_3))

    # Plot
    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, -(Pi_array_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, -(Pi_array_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, -(Pi_array_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    #plt.plot(S_over_L, -(Pi_array_4), linestyle="dashdot", label=f"{scale[3]}" + r"$\eta$")
    plt.title(f"Normalised Kinetic Energy Transfer Function, N = {N}^{dims}", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{\Pi}_{r}^{u}$", fontsize=16)
    plt.legend()
    plt.savefig(f"3D_{N}_1_t{time}.png")

    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, -(Pi_b_array_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, -(Pi_b_array_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, -(Pi_b_array_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    #plt.plot(S_over_L, (Pi_b_array_4), linestyle="dashdot", label=f"{scale[3]}" + r"$\eta$")
    plt.title(f"Normalised Magnetic Energy Transfer Function, N = {N}^{dims}", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{\Pi}_{r}^{b}$", fontsize=16)
    plt.legend()
    plt.savefig(f"3D_{N}_2_t{time}.png")

    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, (F_array_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, (F_array_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, (F_array_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    #plt.plot(S_over_L, (F_array_4), linestyle="dashdot", label=f"{scale[3]}" + r"$\eta$")
    plt.title(f"Normalised Kinetic Enstrophy Transfer Function, N = {N}^{dims}", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{F}_{r}^{u}$", fontsize=16)
    plt.legend()
    plt.savefig(f"3D_{N}_3_t{time}.png")

    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, -(F_b_array_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, -(F_b_array_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, -(F_b_array_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    #plt.plot(S_over_L, -(F_b_array_4), linestyle="dashdot", label=f"{scale[3]}" + r"$\eta$")
    plt.title(f"Normalised Magnetic Enstrophy Transfer Function, N = {N}^{dims}", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{F}_{r}^{b}$", fontsize=16)
    plt.legend()
    plt.savefig(f"3D_{N}_4_t{time}.png")

    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, (Pi_T_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, (Pi_T_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, (Pi_T_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    #plt.plot(S_over_L, -(Pi_array_4), linestyle="dashdot", label=f"{scale[3]}" + r"$\eta$")
    plt.title(f"Normalised Total Energy Transfer Function, N = {N}^{dims}", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{\Pi}_{\text{Total}}$", fontsize=16)
    plt.legend()
    plt.savefig(f"3D_{N}_Total_t{time}.png")

def main2D():
    # Import parameters and fields
    params = Parameters2D("mhd2d-n2048/dyn11c-003.h5")
    #params = Parameters2D("mhd2d-n512/dyn11c-000.h5")
    limit = 2 * np.pi
    N = params.setup()
    print(f"N is {N}")    # Check if single value for N or 3 values for nx, ny, nz
    a_z_sqd, omega_z = params.fields()
    a_z_sqd_hat = Operations().rfft(a_z_sqd)
    a_z_hat = np.full((N, int(N/2)+1), 0+0j)
    for x in range(a_z_sqd_hat.shape[1]):
        for y in range(a_z_sqd_hat.shape[0]):
            a_z_hat[y][x] = np.sqrt(a_z_sqd_hat[y][x])
    a_z = Operations().irfft(a_z_hat)
    dims = a_z.ndim
    alpha = params.alpha
    visc = params.visc
    if visc == 0:
        visc = 0.00001
    time = params.time
    x, y, kx, ky = params.space(limit)
    k_magnitude = np.arange(0, N, 1)
    omega_z_hat = Operations().rfft(omega_z)

    # Zero padding arrays
    zerosReal = np.full((N, N), 0)
    zerosComplex = np.full((N, int(N/2) +1), 0 + 0j)

    Kx, Ky = np.meshgrid(kx, ky) #removed ij to maintain y,x,z format, otherwise output is rotated
    K = np.sqrt(Kx**2 + Ky**2)

    uxhat, uyhat = Operations().anticurl_zhat(zerosReal, omega_z, Kx, Ky)
    ux = Operations().irfft(uxhat)
    uy = Operations().irfft(uyhat)
    B = Operations().curl2D_zhat(zerosReal, a_z, Kx, Ky)
    Bx = B[0]
    By = B[1]
    Bxhat = Operations().rfft(Bx)
    Byhat = Operations().rfft(By)
    print("J time!")

    J = Operations().curl2D(Bx, By, Kx, Ky)

    E_Omni = Operations().Energy_Spectrum_Omni_2D(N, uxhat, uyhat, kx, ky)

    mu0 = 0.01

    l_corr = Operations().l_corr(E_Omni, k_magnitude)
    print(f"Correlation length is {l_corr}")

    eta = Operations().Kolmogorov(visc, k_magnitude, E_Omni)
    print(f"Kolomogorov microscale is {eta}")

    # Redundant nodes and sheets

    # For large ell, use l_corr
    # For small ell, use Kolmogorov scale aka eta

    data_points = 64

    # Kinetic Arrays
    Pi_array_1 = []
    F_array_1 = []
    Pi_array_2 = []
    F_array_2 = []
    Pi_array_3 = []
    F_array_3 = []

    # Magnetic Arrays
    Pi_b_array_1 = []
    F_b_array_1 = []
    Pi_b_array_2 = []
    F_b_array_2 = []
    Pi_b_array_3 = []
    F_b_array_3 = []

    # Total Energy Arrays
    Pi_T_1 = []
    Pi_T_2 = []
    Pi_T_3 = []

    S_over_L = np.linspace(eta, 1, data_points)   # Avoids division by zero

    scale = [2, 3, 5]
    l_large = [scale[0] * eta, scale[1] * eta, scale[2] * eta]

    # For loop
    j = 0
    for i in range(len(l_large)):
        for element in S_over_L:
            l_small = element * l_large[i]
            # Bandpass coeffs
            T_large, T_small = BandpassFilter().coeffs(K, l_large[i], l_small)

            # Filtering
            ux_bL, uy_bL, ux_bS, uy_bS, B_bL, B_bS, Sij_L, Sij_S = BandpassFilter().BandpassFilter2D(uxhat, uyhat, Bxhat, Byhat, Kx, Ky, l_large[i], l_small, alpha, T_large, T_small)

            # Transfer of kinetic energy from large to small scales
            Pi_u = BandpassFilter().Pi_u_2D(ux_bL, uy_bL, ux_bS, uy_bS, B_bL, B_bS, Sij_L, Sij_S, mu0)
            if i == 0:
                Pi_array_1.append(Pi_u)
            if i == 1:
                Pi_array_2.append(Pi_u)
            if i == 2:
                Pi_array_3.append(Pi_u)
            
            # Transer of magnetic energy
            Pi_b = BandpassFilter().Pi_b_2D(ux, uy, B_bL, B_bS, Sij_L, Sij_S, Kx, Ky)
            if i == 0:
                Pi_b_array_1.append(Pi_b)
                Pi_T_1.append(Pi_u + Pi_b)
            if i == 1:
                Pi_b_array_2.append(Pi_b)
                Pi_T_2.append(Pi_u + Pi_b)
            if i == 2:
                Pi_b_array_3.append(Pi_b)
                Pi_T_3.append(Pi_u + Pi_b)

            # Transfer of kinetic enstrophy from large to small scales
            F_u = BandpassFilter().F_u_2D(ux, uy, zerosReal, zerosComplex, omega_z_hat, J, B_bL, B_bS, Kx, Ky, l_large[i], l_small, alpha, T_large, T_small)
            if i == 0:
                F_array_1.append(F_u)
            if i == 1:
                F_array_2.append(F_u)
            if i == 2:
                F_array_3.append(F_u)
            
            # Transfer of magnetic enstrophy
            F_b = BandpassFilter().F_b(ux, uy, zerosReal, zerosComplex, zerosComplex, a_z_hat, Kx, Ky, 0, l_large[i], l_small, alpha, T_large, T_small)
            if i == 0:
                F_b_array_1.append(F_b)
            if i == 1:
                F_b_array_2.append(F_b)
            if i == 2:
                F_b_array_3.append(F_b)
            
            j += 1
            print(f"{round(j/(len(S_over_L) * len(l_large))*100, 2)}% completed")
    
    print(f"Dimensions are {dims}D")
    # Normalise arrays
    Pi_array_1 = Pi_array_1/max(map(abs, Pi_array_1))
    Pi_array_2 = Pi_array_2/max(map(abs, Pi_array_2))
    Pi_array_3 = Pi_array_3/max(map(abs, Pi_array_3))
    Pi_b_array_1 = Pi_b_array_1/max(map(abs, Pi_b_array_1))
    Pi_b_array_2 = Pi_b_array_2/max(map(abs, Pi_b_array_2))
    Pi_b_array_3 = Pi_b_array_3/max(map(abs, Pi_b_array_3))
    F_array_1 = F_array_1/max(map(abs, F_array_1))
    F_array_2 = F_array_2/max(map(abs, F_array_2))
    F_array_3 = F_array_3/max(map(abs, F_array_3))
    F_b_array_1 = F_b_array_1/max(map(abs, F_b_array_1))
    F_b_array_2 = F_b_array_2/max(map(abs, F_b_array_2))
    F_b_array_3 = F_b_array_3/max(map(abs, F_b_array_3))
    Pi_T_1 = Pi_T_1/max(map(abs, Pi_T_1))
    Pi_T_2 = Pi_T_2/max(map(abs, Pi_T_2))
    Pi_T_3 = Pi_T_3/max(map(abs, Pi_T_3))

    # Plot
    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, -(Pi_array_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, -(Pi_array_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, -(Pi_array_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    plt.title(f"Normalised Kinetic Energy Transfer Function, N = {N}^{dims}", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{\Pi}_{r}^{u}$", fontsize=16)
    plt.legend()
    plt.savefig(f"{N}_1_t{time}.png")

    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, (Pi_b_array_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, (Pi_b_array_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, (Pi_b_array_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    plt.title(f"Normalised Magnetic Energy Transfer Function, N = {N}^{dims}", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{\Pi}_{r}^{b}$", fontsize=16)
    plt.legend()
    plt.savefig(f"{N}_2_t{time}.png")

    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, -(F_array_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, -(F_array_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, (F_array_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    plt.title(f"Normalised Kinetic Enstrophy Transfer Function, N = {N}^{dims}", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{F}_{r}^{u}$", fontsize=16)
    plt.legend()
    plt.savefig(f"{N}_3_t{time}.png")

    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, -(F_b_array_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, -(F_b_array_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, -(F_b_array_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    plt.title(f"Normalised Magnetic Enstrophy Transfer Function, N = {N}^{dims}", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{F}_{r}^{b}$", fontsize=16)
    plt.legend()
    plt.savefig(f"{N}_4_t{time}.png")

    plt.figure(figsize=(9,6))
    plt.plot(S_over_L, (Pi_T_1), label=f"{scale[0]}" + r"$\eta$")
    plt.plot(S_over_L, (Pi_T_2), linestyle="dotted", label=f"{scale[1]}" + r"$\eta$")
    plt.plot(S_over_L, (Pi_T_3), linestyle="dashdot", label=f"{scale[2]}" + r"$\eta$")
    plt.title(f"Normalised Total Energy Transfer Function, N = {N}^{dims}", fontsize=20)
    plt.xlabel("S/L", fontsize=16)
    plt.ylabel(r"$\hat{\Pi}_{\text{Total}}$", fontsize=16)
    plt.legend()
    plt.savefig(f"2D_{N}_Total_t{time}.png")

def EnergySpectrum2D(filename):
    # Import parameters and fields
    params = Parameters2D(filename)
    limit = 2 * np.pi
    N = params.setup()
    print(f"N is {N}")    # Check if single value for N or 3 values for nx, ny, nz
    a_z_sqd, omega_z = params.fields()
    dims = omega_z.ndim
    visc = params.visc
    time = params.time
    x, y, kx, ky = params.space(limit)
    k_magnitude = np.arange(0, N, 1)

    # Zero padding arrays
    zerosReal = np.full((N, N), 0)

    Kx, Ky = np.meshgrid(kx, ky) #removed ij to maintain y,x,z format, otherwise output is rotated

    uxhat, uyhat = Operations().anticurl_zhat(zerosReal, omega_z, Kx, Ky)
    E_Omni = Operations().Energy_Spectrum_Omni_2D(N, uxhat, uyhat, kx, ky)

    plt.figure(figsize=(9, 6))
    plt.plot(k_magnitude, E_Omni)
    plt.title(f"MHD Energy Spectrum, for N = {N}^{dims}", fontsize = 20)
    plt.xlabel(r"$k = |\vec{k}|")
    plt.ylabel(r"$E(\vec{k})$")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(k_magnitude, 2*10e-8 * (k_magnitude)**(-5/3), "--", color="red")
    plt.text(10, 10**(-10), r"$\propto k^{-5/3}$", fontsize=14, color="red")
    plt.plot(k_magnitude, 2*10e-8 * (k_magnitude)**(-3/2), "--", color="green")
    plt.text(10, 10**(-7.5), r"$\propto k^{-3/2}$", fontsize=14, color="green")

def Surface2D():
    # Import parameters and fields
    #params = Parameters2D("mhd2d-n2048/dyn11c-003.h5")
    S_over_L, t1, P_11, P_12, P_13, Pb_11, Pb_12, Pb_13, F_11, F_12, F_13, Fb_11, Fb_12, Fb_13 = Get2DArrays().Arrays("mhd2d-n512/dyn11c-000.h5")
    S_over_L, t2, P_21, P_22, P_23, Pb_21, Pb_22, Pb_23, F_21, F_22, F_23, Fb_21, Fb_22, Fb_23 = Get2DArrays().Arrays("mhd2d-n512/dyn11c-001.h5")
    S_over_L, t3, P_31, P_32, P_33, Pb_31, Pb_32, Pb_33, F_31, F_32, F_33, Fb_31, Fb_32, Fb_33 = Get2DArrays().Arrays("mhd2d-n512/dyn11c-002.h5")
    S_over_L, t4, P_41, P_42, P_43, Pb_41, Pb_42, Pb_43, F_41, F_42, F_43, Fb_41, Fb_42, Fb_43 = Get2DArrays().Arrays("mhd2d-n512/dyn11c-003.h5")

    # Fields
    time_array = np.array([t1, t2, t3, t4])
    field_X, field_Y = np.meshgrid(S_over_L, time_array)
    Get2DArrays().Plotting(2, field_X, field_Y, P_11, P_21, P_31, P_41, -Pb_11, Pb_21, Pb_31, Pb_41, F_11, F_21, F_31, F_41, Fb_11, -Fb_21, Fb_31, -Fb_41)
    Get2DArrays().Plotting(3, field_X, field_Y, P_12, P_22, P_32, P_42, -Pb_12, Pb_22, Pb_32, Pb_42, F_12, F_22, F_32, F_42, Fb_12, -Fb_22, Fb_32, Fb_42)
    Get2DArrays().Plotting(5, field_X, field_Y, P_13, P_23, P_33, P_43, -Pb_13, Pb_23, Pb_33, Pb_43, F_13, F_23, F_33, F_43, -Fb_13, -Fb_23, Fb_33, -Fb_43)

if __name__ == "__main__":
    main3D()
    #main2D()
    #EnergySpectrum2D("mhd2d-n2048/dyn11c-003.h5")
    #Surface2D()
    plt.show()
