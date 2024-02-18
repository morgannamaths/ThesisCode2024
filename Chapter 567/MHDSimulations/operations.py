import numpy as np
import h5py

class Operations():
    def __init__(self):
        pass

    def curl3D(self, uxhat, uyhat, uzhat, Kx, Ky, Kz):
        duxhat_dy = self.fourier_derivative(uxhat, Ky)
        duxhat_dz = self.fourier_derivative(uxhat, Kz)
        duyhat_dx = self.fourier_derivative(uyhat, Kx)
        duyhat_dz = self.fourier_derivative(uyhat, Kz)
        duzhat_dx = self.fourier_derivative(uzhat, Kx)
        duzhat_dy = self.fourier_derivative(uzhat, Ky)

        dux_dy = self.irfft(duxhat_dy)
        dux_dz = self.irfft(duxhat_dz)
        duy_dx = self.irfft(duyhat_dx)
        duy_dz = self.irfft(duyhat_dz)
        duz_dx = self.irfft(duzhat_dx)
        duz_dy = self.irfft(duzhat_dy)

        curl = np.array([duz_dy - duy_dz, dux_dz - duz_dx, duy_dx - dux_dy])
        return curl

    def curl2D(self, ux, uy, Kx, Ky):
        uxhat = np.fft.rfft2(ux, s=None, norm="forward")
        uyhat = np.fft.rfft2(uy, s=None, norm="forward")
        duxhat_dy = 1j * Ky * uxhat
        duyhat_dx = 1j * Kx * uyhat
        dux_dy = np.fft.irfft2(duxhat_dy, s=None, norm="forward")
        duy_dx = np.fft.irfft2(duyhat_dx, s=None, norm="forward")
        return duy_dx - dux_dy
    
    def anticurl3D(self, fxhat, fyhat, fzhat, Kx, Ky, Kz):
        negative_cf = -1 * self.curl3D(fxhat, fyhat, fzhat, Kx, Ky, Kz)
        print(negative_cf.shape)
        negative_cf_x = negative_cf[0]
        print(negative_cf_x.shape)
        negative_cf_y = negative_cf[1]
        negative_cf_z = negative_cf[2]

        negative_cfhat_x = self.rfft(negative_cf_x)
        print(negative_cfhat_x.shape)
        negative_cfhat_y = self.rfft(negative_cf_y)
        negative_cfhat_z = self.rfft(negative_cf_z)

        anti_cfhat_x = np.full(Kx.shape, 0+0j)
        anti_cfhat_y = np.full(Ky.shape, 0+0j)
        anti_cfhat_z = np.full(Kz.shape, 0+0j)
        
        for y in range(Kx.shape[0]):
            for x in range(Kx.shape[1]):
                for z in range(Kx.shape[2]):
                    if Kx[y][x][z] != 0:
                        anti_cfhat_x[y][x][z] = -1 * negative_cfhat_x[y][x][z]/(Kx[y][x][z]**2)
        for y in range(Ky.shape[0]):
            for x in range(Ky.shape[1]):
                for z in range(Ky.shape[2]):
                    if Ky[y][x][z] != 0:
                        anti_cfhat_y[y][x][z] = -1 * negative_cfhat_y[y][x][z]/(Ky[y][x][z]**2)
        for y in range(Kz.shape[0]):
            for x in range(Kz.shape[1]):
                for z in range(Kz.shape[2]):
                    if Kz[y][x][z] != 0:
                        anti_cfhat_z[y][x][z] = -1 * negative_cfhat_z[y][x][z]/(Kz[y][x][z]**2)
        #test_anti_cfhat_x = -1 * negative_cfhat_x/(Kx**2)
        #print(test_anti_cfhat_x - anti_cfhat_x)
        return anti_cfhat_x, anti_cfhat_y, anti_cfhat_z
    
    def anticurl2D(self, F_z, Kx, Ky):
        negative_cf = -1 * self.curl2D_Single(F_z, Kx, Ky)
        ksqd = Kx**2 + Ky**2
        anti_cf = np.full((Kx.shape[0], Kx.shape[1]), 0+0j)
        print(Kx.shape)
        for y in range(Kx.shape[0]):
            for x in range(Kx.shape[1]):
                if ksqd[y][x] != 0:
                    anti_cf[y][x] = -1 * negative_cf[y][x]/(ksqd[y][x])
        return anti_cf
    
    def anticurl_zhat(self, zeros, F_z, Kx, Ky):
        F_z_hat = np.fft.rfft2(F_z, s=None, norm="forward")

        dFzhat_dx = self.fourier_derivative(F_z_hat, Kx)
        dFzhat_dy = self.fourier_derivative(F_z_hat, Ky)

        dFz_dx = self.irfft(dFzhat_dx)
        dFz_dy = self.irfft(dFzhat_dy)
        print(f"dFzdx has shape {dFz_dx.shape}")

        neg_curl_F = -1 * np.array([dFz_dy, - dFz_dx, zeros])
        print(f"negcurlF has shape {neg_curl_F.shape}")
        neg_curl_F_hat_x = self.rfft(neg_curl_F[1])
        neg_curl_F_hat_y = self.rfft(neg_curl_F[2])

        anticurl_F_hat_x = np.full(Kx.shape, 0+0j)
        anticurl_F_hat_y = np.full(Ky.shape, 0+0j)

        print(f"negcurlFhat_x has shape {neg_curl_F_hat_x.shape}")
        print(f"anticurlFhat_x has shape {anticurl_F_hat_x.shape}")
        print(f"Kx has shape {Kx.shape}")

        for y in range(Kx.shape[0]):
            for x in range(Kx.shape[1]):
                    if Kx[y][x] != 0:
                        anticurl_F_hat_x[y][x] = -1 * neg_curl_F_hat_x[y][x]/(Kx[y][x]**2)
        for y in range(Ky.shape[0]):
            for x in range(Ky.shape[1]):
                    if Ky[y][x] != 0:
                        anticurl_F_hat_y[y][x] = -1 * neg_curl_F_hat_y[y][x]/(Ky[y][x]**2)
        
        return anticurl_F_hat_x, anticurl_F_hat_y
    
    def curl2D_zhat(self, zeros, F_z, Kx, Ky):
        print(f"F_z has shape {F_z.shape}")
        F_z_hat = np.fft.rfft2(F_z, s=None, norm="forward")
        print(f"Fhat has shape {F_z_hat.shape}")

        # Take curl
        dFzhat_dx = self.fourier_derivative(F_z_hat, Kx)
        dFzhat_dy = self.fourier_derivative(F_z_hat, Ky)

        dFz_dx = self.irfft(dFzhat_dx)
        dFz_dy = self.irfft(dFzhat_dy)

        curl_F = -1 * np.array([dFz_dy, - dFz_dx, zeros])
        return curl_F

    def rfft_field3D(self, ux, uy, uz, Bx, By, Bz):
        uxhat = self.rfft(ux)
        uyhat = self.rfft(uy)
        uzhat = self.rfft(uz)
        Bxhat = self.rfft(Bx)
        Byhat = self.rfft(By)
        Bzhat = self.rfft(Bz)
        return uxhat, uyhat, uzhat, Bxhat, Byhat, Bzhat
    
    def rfft(self, f):
        if f.ndim == 3:
            return np.fft.rfftn(f, s=None, axes=(0, 2, 1), norm = "forward")
        elif f.ndim == 2:
            return np.fft.rfftn(f, s=None, axes=(0, 1), norm = "forward")
        else:
            print("f not dimension 2 or 3")
            return

    def irfft(self, fhat):
        if fhat.ndim == 3:
            return np.fft.irfftn(fhat, s=None, axes=(0, 1), norm = "forward")
        elif fhat.ndim == 2:
            return np.fft.irfftn(fhat, s=None, axes=(0, 1), norm = "forward")
        else:
            print("fhat not dimension 2 or 3")
            return

    def fourier_derivative(self, uhat, k):
        return 1j * k * uhat

    def efficient_abs_sqd(self, x):
        return np.real(x)**2 + np.imag(x)**2

    def Energy_Spectrum_Omni(self, N, uxhat, uyhat, uzhat, kx, ky, kz):
        E_Omni = np.zeros(N)
        for m in range(N-1):
            kz_val = kz[m]
            for j in range(N-1):
                ky_val = ky[j]
                for i in range(int(N/2)):
                    kx_val = kx[i]
                    k_sqd = kx_val**2 + ky_val**2 + kz_val**2
                    k_index = int(round(np.sqrt(k_sqd)))
                    E_Omni[k_index] += self.efficient_abs_sqd(uxhat[j][i][m]) + self.efficient_abs_sqd(uyhat[j][i][m]) + self.efficient_abs_sqd(uzhat[j][i][m])
        return E_Omni
    
    def Energy_Spectrum_Omni_2D(self, N, uxhat, uyhat, kx, ky):
        E_Omni = np.zeros(N)
        for j in range(N-1):
            ky_val = ky[j]
            for i in range(int(N/2)):
                kx_val = kx[i]
                k_sqd = kx_val**2 + ky_val**2
                k_index = int(round(np.sqrt(k_sqd)))
                E_Omni[k_index] += self.efficient_abs_sqd(uxhat[j][i]) + self.efficient_abs_sqd(uyhat[j][i])
        return E_Omni

    def l_corr(self, E_Omni, k_magnitude):
        int_E = 0
        for i in range(len(k_magnitude)):
            int_E += E_Omni[i]
        #print(int_E)
        int_kinv_E = 0
        for i in range(len(k_magnitude)):
            if i > 0:       # Avoid dividing by zero
                int_kinv_E += E_Omni[i]/k_magnitude[i]
        #print(int_kinv_E)
        return 3 * np.pi/4 * int_kinv_E/int_E

    def dissipation(self, viscosity, k_magnitude, E_Omni):
        dissipation = 0
        for i in range(len(k_magnitude)):
            dissipation += k_magnitude[i]**2 * E_Omni[i]
            #print(dissipation)
        return 2 * viscosity * dissipation

    def Kolmogorov(self, viscosity, k_magnitude, E_Omni):
        dissipation = self.dissipation(viscosity, k_magnitude, E_Omni)
        print(dissipation)
        print(viscosity)
        return (viscosity**3/dissipation)**0.25
    
    def Sij(self, uxhat_bandpass, uyhat_bandpass, uzhat_bandpass, Kx, Ky, Kz):
        # Find uxdx_bandpass
        uxdx_bandpass = self.irfft(1j * Kx * uxhat_bandpass)

        # Find uxdy_bandpass
        uxdy_bandpass = self.irfft(1j * Ky * uxhat_bandpass)

        # Find uxdz_bandpass
        uxdz_bandpass = self.irfft(1j * Kz * uxhat_bandpass)

        # Find uydx_bandpass
        uydx_bandpass = self.irfft(1j * Kx * uyhat_bandpass)

        # Find uydy_bandpass
        uydy_bandpass = self.irfft(1j * Ky * uyhat_bandpass)

        # Find uydz_bandpass
        uydz_bandpass = self.irfft(1j * Kz * uyhat_bandpass)

        # Find uzdx_bandpass
        uzdx_bandpass = self.irfft(1j * Kx * uzhat_bandpass)

        # Find uzdy_bandpass
        uzdy_bandpass = self.irfft(1j * Ky * uzhat_bandpass)

        # Find uzdz_bandpass
        uzdz_bandpass = self.irfft(1j * Kz * uzhat_bandpass)

        # Compute upper triangle elements of Sij
        S11 = uxdx_bandpass
        S12 = 0.5 * (uxdy_bandpass + uydx_bandpass)
        S13 = 0.5 * (uxdz_bandpass + uzdx_bandpass)
        S22 = uydy_bandpass
        S23 = 0.5 * (uydz_bandpass + uzdy_bandpass)
        S33 = uzdz_bandpass

        # Use symmetry for lower triangle elements of Sij
        S21 = S12
        S31 = S13
        S32 = S23

        # Write out Sij
        Sij = np.array([[S11, S12, S13], [S21, S22, S23], [S31, S32, S33]])

        return Sij
    
    def Sij2D(self, uxhat_bandpass, uyhat_bandpass, Kx, Ky,):
        # Find uxdx_bandpass
        uxdx_bandpass = self.irfft(1j * Kx * uxhat_bandpass)

        # Find uxdy_bandpass
        uxdy_bandpass = self.irfft(1j * Ky * uxhat_bandpass)

        # Find uydx_bandpass
        uydx_bandpass = self.irfft(1j * Kx * uyhat_bandpass)

        # Find uydy_bandpass
        uydy_bandpass = self.irfft(1j * Ky * uyhat_bandpass)

        # Compute upper triangle elements of Sij
        S11 = uxdx_bandpass
        S12 = 0.5 * (uxdy_bandpass + uydx_bandpass)
        S22 = uydy_bandpass

        # Use symmetry for lower triangle elements of Sij
        S21 = S12

        # Write out Sij
        Sij = np.array([[S11, S12,], [S21, S22]])

        return Sij
    
    def Tauij(self, ux_bandpass, uy_bandpass, uz_bandpass):
        # Compute upper triangle elements
        Tau11 = -1 * ux_bandpass * ux_bandpass
        Tau12 = -1 * ux_bandpass * uy_bandpass
        Tau13 = -1 * ux_bandpass * uz_bandpass
        Tau22 = -1 * uy_bandpass * uy_bandpass
        Tau23 = -1 * uy_bandpass * uz_bandpass
        Tau33 = -1 * uz_bandpass * uz_bandpass

        # Use symmetry for lower triangle elements
        Tau21 = Tau12
        Tau31 = Tau13
        Tau32 = Tau23

        # Write out Tauij
        Tauij = np.array([[Tau11, Tau12, Tau13], [Tau21, Tau22, Tau23], [Tau31, Tau32, Tau33]])

        return Tauij
    
    def Tauij2D(self, ux_bandpass, uy_bandpass):
        # Compute upper triangle elements
        Tau11 = -1 * ux_bandpass * ux_bandpass
        Tau12 = -1 * ux_bandpass * uy_bandpass
        Tau22 = -1 * uy_bandpass * uy_bandpass

        # Use symmetry for lower triangle elements
        Tau21 = Tau12

        # Write out Tauij
        Tauij = np.array([[Tau11, Tau12], [Tau21, Tau22]])

        return Tauij
    
    def Pi_u(self, Sij_L, Sij_S, Tauij_L, Tauij_S, B_bL, B_bS, mu0):
        first_term = self.Frobenius3D_2Terms(Tauij_S, Sij_L)
        second_term = self.Frobenius3D_2Terms(Tauij_L, Sij_S)
        
        BBS_LSL = self.Frobenius3D_3Terms(B_bL, B_bS, Sij_L)
        BBS_SLL = self.Frobenius3D_3Terms(B_bS, B_bL, Sij_L)
        BBS_SSL = self.Frobenius3D_3Terms(B_bS, B_bS, Sij_L)

        Pi_u = np.average(first_term - second_term) + (np.average(BBS_LSL) + np.average(BBS_SLL) + np.average(BBS_SSL))/mu0
        return Pi_u
    
    def Pi_u_2D(self, Sij_L, Sij_S, Tauij_L, Tauij_S, B_bL, B_bS, mu0):
        first_term = self.Frobenius2D_2Terms(Tauij_S, Sij_L)
        second_term = self.Frobenius2D_2Terms(Tauij_L, Sij_S)
        
        BBS_LSL = self.Frobenius2D_3Terms(B_bL, B_bS, Sij_L)
        BBS_SLL = self.Frobenius2D_3Terms(B_bS, B_bL, Sij_L)
        BBS_SSL = self.Frobenius2D_3Terms(B_bS, B_bS, Sij_L)

        Pi_u = np.average(first_term - second_term) + (np.average(BBS_LSL) + np.average(BBS_SLL) + np.average(BBS_SSL))/mu0
        return Pi_u
    
    def A_dot_B_dot_gradC(self, A, B, C, Kx, Ky, Kz):
        dCx_dx = self.irfft(1j * Kx * C[0])
        dCx_dy = self.irfft(1j * Ky * C[0])
        dCx_dz = self.irfft(1j * Kz * C[0])

        dCy_dx = self.irfft(1j * Kx * C[1])
        dCy_dy = self.irfft(1j * Ky * C[1])
        dCy_dz = self.irfft(1j * Kz * C[1])

        dCz_dx = self.irfft(1j * Kx * C[2])
        dCz_dy = self.irfft(1j * Ky * C[2])
        dCz_dz = self.irfft(1j * Kz * C[2])

        # B dot grad C
        B_dot_grad_Cx = B[0] * dCx_dx + B[1] * dCx_dy + B[2] * dCx_dz
        B_dot_grad_Cy = B[0] * dCy_dx + B[1] * dCy_dy + B[2] * dCy_dz
        B_dot_grad_Cz = B[0] * dCz_dx + B[1] * dCz_dy + B[2] * dCz_dz

        return A[0] * B_dot_grad_Cx + A[1] * B_dot_grad_Cy + A[2] * B_dot_grad_Cz
    
    def A_dot_B_dot_gradC_2D(self, A, B, C, Kx, Ky):
        dCx_dx = self.irfft(1j * Kx * C[0])
        dCx_dy = self.irfft(1j * Ky * C[0])

        dCy_dx = self.irfft(1j * Kx * C[1])
        dCy_dy = self.irfft(1j * Ky * C[1])

        # B dot grad C
        B_dot_grad_Cx = B[0] * dCx_dx + B[1] * dCx_dy 
        B_dot_grad_Cy = B[0] * dCy_dx + B[1] * dCy_dy

        return A[0] * B_dot_grad_Cx + A[1] * B_dot_grad_Cy
    
    def Frobenius3D_3Terms(self, F1, F2, S):
        return F2[0] * F2[0] * S[0][0] + F1[1] * F2[1] * S[1][1] + F1[2] * F2[2] * S[2][2] + 2 * F1[0] * F2[1] * S[0][1]  + 2 * F1[0] * F2[2] * S[0][2]  + 2 * F1[1] * F2[2] * S[1][2]
    
    def Frobenius3D_2Terms(self, Tau, Sij):
        return Sij[0][0] * Tau[0][0] + Sij[1][1] * Tau[1][1] + Sij[2][2] * Tau[2][2] + 2 * Sij[0][1] * Tau[1][0] + 2 * Sij[0][2] * Tau[2][0] + 2 * Sij[1][2] * Tau[2][1]
    
    def Frobenius2D_3Terms(self, F1, F2, S):
        return F2[0] * F2[0] * S[0][0] + F1[1] * F2[1] * S[1][1] + 2 * F1[0] * F2[1] * S[0][1]
    
    def Frobenius2D_2Terms(self, Tau, Sij):
        return Sij[0][0] * Tau[0][0] + Sij[1][1] * Tau[1][1] + 2 * Sij[0][1] * Tau[1][0]

if __name__ == "__main__":
    print("Test")
    vx = h5py.File("vx.h5", "r")
    ux = vx["vx"][:]
    vy = h5py.File("vy.h5", "r")
    uy = vy["vy"][:]
    om = h5py.File("om.h5", "r")
    omega_dataset = om["om"][:]

    N = len(ux)

    kx = np.fft.rfftfreq(len(ux), 1/len(ux))
    ky = np.fft.fftfreq(len(uy), 1/len(uy))
    ky[int(N/2)] = N/2

    Kx, Ky = np.meshgrid(kx, ky)
    print(Kx.shape)
    print(Ky.shape)

    omega_python_fourier = Operations().curl2D(ux, uy, Kx, Ky)
    print(omega_python_fourier.shape)
    print(omega_python_fourier[0])
    print(omega_dataset[0])
