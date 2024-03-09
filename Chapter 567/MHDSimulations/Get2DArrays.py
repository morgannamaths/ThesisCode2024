import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from parameters import Parameters
from parameters2D import Parameters2D
from operations import Operations
from bandpass import BandpassFilter

class Get2DArrays():
    def __init__(self):
        pass

    def Arrays(self, filename):
        params = Parameters2D(filename)
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

        scale = [100, 130, 160]
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

        return S_over_L, time, Pi_array_1, Pi_array_2, Pi_array_3, Pi_b_array_1, Pi_b_array_2, Pi_b_array_3, F_array_1, F_array_2, F_array_3, F_b_array_1, F_b_array_2, F_b_array_3, Pi_T_1, Pi_T_2, Pi_T_3
    
    def Plotting(self, scale, field_X, field_Y, P_11, P_21, P_31, P_41, Pb_11, Pb_21, Pb_31, Pb_41, F_11, F_21, F_31, F_41, Fb_11, Fb_21, Fb_31, Fb_41, T_11, T_21, T_31, T_41):
        Pi_u_field = np.array([P_11, P_21, P_31, P_41])
        Pi_b_field = np.array([Pb_11, Pb_21, Pb_31, Pb_41])
        F_u_field = np.array([F_11, F_21, F_31, F_41])
        F_b_field = np.array([Fb_11, Fb_21, Fb_31, Fb_41])
        T_field = np.array([T_11, T_21, T_31, T_41])

        # Plotting
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,6))
        ax.plot_surface(field_X, field_Y, Pi_u_field, cmap = "rainbow")
        plt.title(f"Normalised Kinetic Energy Transfer Function, {scale}" + r"$\eta$", fontsize=22)
        plt.ylabel("Time (s)", fontsize=18)
        plt.xlabel("S/L", fontsize=18)
        plt.savefig(f"SurfacePlots/Scale{scale}KineticEnergy.png")

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,6))
        ax.plot_surface(field_X, field_Y, Pi_b_field, cmap = "rainbow")
        plt.title(f"Normalised Magnetic Energy Transfer Function, {scale}" + r"$\eta$", fontsize=22)
        plt.ylabel("Time (s)", fontsize=18)
        plt.xlabel("S/L", fontsize=18)
        plt.savefig(f"SurfacePlots/Scale{scale}MagneticEnergy.png")

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,6))
        ax.plot_surface(field_X, field_Y, F_u_field, cmap = "rainbow")
        plt.title(f"Normalised Kinetic Enstrophy Transfer Function, {scale}" + r"$\eta$", fontsize=22)
        plt.ylabel("Time (s)", fontsize=18)
        plt.xlabel("S/L", fontsize=18)
        plt.savefig(f"SurfacePlots/Scale{scale}KineticEnstrophy.png")

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},  figsize=(10,6))
        ax.plot_surface(field_X, field_Y, F_b_field, cmap = "rainbow")
        plt.title(f"Normalised Mean Squared Potential Transfer Function, {scale}" + r"$\eta$", fontsize=22)
        plt.ylabel("Time (s)", fontsize=18)
        plt.xlabel("S/L", fontsize=18)
        plt.savefig(f"SurfacePlots/Scale{scale}MagneticEnstrophy.png")

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},  figsize=(10,6))
        ax.plot_surface(field_X, field_Y, T_field, cmap = "rainbow")
        plt.title(f"Normalised Total Energy Transfer, {scale}" + r"$\eta$", fontsize=22)
        plt.ylabel("Time (s)", fontsize=18)
        plt.xlabel("S/L", fontsize=18)
        plt.savefig(f"SurfacePlots/Scale{scale}TotalEnergy.png")