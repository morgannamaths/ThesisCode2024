import numpy as np
import numdifftools as nd
import h5py

class Operations():
    def __init__(self):
        pass

    def curl2D(self, ux, uy, x, y):
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dux_dx, dux_dy = np.gradient(ux, dx, dy)
        duy_dx, duy_dy = np.gradient(uy, dx, dy)
        return duy_dx - dux_dy

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

    def rfft_ufield(self, ux, uy, uz):
        uxhat = np.fft.rfftn(ux, s=None, axes=(0, 2, 1), norm = "forward")
        uyhat = np.fft.rfftn(uy, s=None, axes=(0, 2, 1), norm = "forward")
        uzhat = np.fft.rfftn(uz, s=None, axes=(0, 2, 1), norm = "forward")
        return uxhat, uyhat, uzhat
    
    def rfft(self, f):
        fhat = np.fft.rfftn(f, s=None, axes=(0, 2, 1), norm = "forward")
        return fhat

    def irfft(self, fhat):
        return np.fft.irfftn(fhat, s=None, axes=(0, 2, 1), norm="forward")

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
        return 2 * viscosity * dissipation

    def Kolmogorov(self, viscosity, k_magnitude, E_Omni):
        dissipation = self.dissipation(viscosity, k_magnitude, E_Omni)
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
    
    def Pi_LtS(self, Sij_L, Sij_S, Tauij_L, Tauij_S):
        first_term = Sij_L[0][0] * Tauij_S[0][0] + Sij_L[1][1] * Tauij_S[1][1] + Sij_L[2][2] * Tauij_S[2][2] + 2 * Sij_L[0][1] * Tauij_S[1][0] + 2 * Sij_L[0][2] * Tauij_S[2][0] + 2 * Sij_L[1][0] * Tauij_S[0][1]
        second_term = Sij_S[0][0] * Tauij_L[0][0] + Sij_S[1][1] * Tauij_L[1][1] + Sij_S[2][2] * Tauij_L[2][2] + 2 * Sij_S[0][1] * Tauij_L[1][0] + 2 * Sij_S[0][2] * Tauij_L[2][0] + 2 * Sij_S[1][0] * Tauij_L[0][1]
        return np.average(first_term - second_term)

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
