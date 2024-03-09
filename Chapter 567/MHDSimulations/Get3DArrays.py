import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from parameters import Parameters
from operations import Operations
from bandpass import BandpassFilter

class Get3DArrays():
    def __init__(self):
        pass

    def Arrays(self, filename1, filename2):
        params = Parameters(filename1, filename2)
        print("Parameters obtained")
        limit = 2 * np.pi
        ux, uy, uz, Bx, By, Bz = params.fields()
        dims = ux.ndim
        N = params.setup()
        print(N)    # Check if single value for N or 3 values for nx, ny, nz
        alpha = params.alpha
        time = params.time
        visc = params.visc
        x, y, z, kx, ky, kz = params.space(limit)
        k_magnitude = np.arange(0, N, 1)
        uxhat, uyhat, uzhat, Bxhat, Byhat, Bzhat = Operations().rfft_field3D(ux, uy, uz, Bx, By, Bz)
        print("Found uhat and Bhat")

        Kx, Ky, Kz = np.meshgrid(kx, ky, kz) #removed ij to maintain y,x,z format, otherwise output is rotated
        K = np.sqrt(Kx**2 + Ky**2 + Kz**2)
        print("Done K")

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

        # Total Energy Arrays
        Pi_T_1 = []
        Pi_T_2 = []
        Pi_T_3 = []

        S_over_L = np.linspace(eta, 1, data_points)   # Avoids division by zero

        scale = [550, 600, 650]
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
                
                # Transer of magnetic energy
                Pi_b = BandpassFilter().Pi_b(ux, uy, uz, B_bL, B_bS, Sij_L, Sij_S, Kx, Ky, Kz)
                if i == 0:
                    Pi_T_1.append(Pi_u + Pi_b)
                if i == 1:
                    Pi_T_2.append(Pi_u + Pi_b)
                if i == 2:
                    Pi_T_3.append(Pi_u + Pi_b)
                
                j += 1
                print(f"{round(j/(len(S_over_L) * len(l_large))*100, 2)}% completed")
        
        print(f"Dimensions are {dims}D")
        Pi_T_1 = Pi_T_1/max(map(abs, Pi_T_1))
        Pi_T_2 = Pi_T_2/max(map(abs, Pi_T_2))
        Pi_T_3 = Pi_T_3/max(map(abs, Pi_T_3))

        return S_over_L, time, Pi_T_1, Pi_T_2, Pi_T_3
    
    def Plotting(self, scale, field_X, field_Y, T_11, T_21):
        T_field = np.array([T_11, T_21])

        # Plotting
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},  figsize=(10,6))
        ax.plot_surface(field_X, field_Y, T_field, cmap = "rainbow")
        plt.title(f"Normalised Total Energy Transfer, {scale}" + r"$\eta$", fontsize=22)
        plt.ylabel("Time (s)", fontsize=18)
        plt.xlabel("S/L", fontsize=18)
        plt.savefig(f"SurfacePlots/3D_Scale{scale}TotalEnergy.png")