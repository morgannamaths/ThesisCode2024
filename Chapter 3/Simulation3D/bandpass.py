import numpy as np
from operations import Operations

class BandpassFilter():
    def __init__(self):
        pass

    def coeffs(self, k_magnitude, l_corr, eta):
        kap_large = k_magnitude * l_corr/2
        T_large = 2 * kap_large**2 * np.exp(-1 * kap_large**2)

        kap_small = k_magnitude * eta/2
        T_small = 2 * kap_small**2 * np.exp(-1 * kap_small**2)

        return T_large, T_small

    def Pi_LtS(self, uxhat, uyhat, uzhat, Kx, Ky, Kz, l_corr, eta, alpha, T_large, T_small):
        # Bandpass Filter for Large scale
        uxhat_bL = alpha/np.sqrt(l_corr) * T_large * uxhat
        uyhat_bL = alpha/np.sqrt(l_corr) * T_large * uyhat
        uzhat_bL = alpha/np.sqrt(l_corr) * T_large * uzhat

        ux_bL = Operations().irfft(uxhat_bL)
        uy_bL = Operations().irfft(uyhat_bL)
        uz_bL = Operations().irfft(uzhat_bL)

        Sij_L = Operations().Sij(uxhat_bL, uyhat_bL, uzhat_bL, Kx, Ky, Kz)
        Tauij_L = Operations().Tauij(ux_bL, uy_bL, uz_bL)

        # Bandpass Filter for Small Scales
        uxhat_bS = alpha/np.sqrt(eta) * T_small * uxhat
        uyhat_bS = alpha/np.sqrt(eta) * T_small * uyhat
        uzhat_bS = alpha/np.sqrt(eta) * T_small * uzhat

        ux_bS = Operations().irfft(uxhat_bS)
        uy_bS = Operations().irfft(uyhat_bS)
        uz_bS = Operations().irfft(uzhat_bS)

        Sij_S = Operations().Sij(uxhat_bS, uyhat_bS, uzhat_bS, Kx, Ky, Kz)
        Tauij_S = Operations().Tauij(ux_bS, uy_bS, uz_bS)

        Pi_LtS = Operations().Pi_LtS(Sij_L, Sij_S, Tauij_L, Tauij_S)
        #print(f"Pi_LtS = {Pi_LtS}")    # Returns scalar, as expected

        return Pi_LtS
    
    def F_LtS(self, ux, uy, uz, omega_x, omega_y, omega_z, Kx, Ky, Kz, l_corr, eta, alpha, T_large, T_small):
        # Need omega_bL and u dot grad omega_bS
        omega_x_hat = Operations().rfft(omega_x)
        omega_y_hat = Operations().rfft(omega_y)
        omega_z_hat = Operations().rfft(omega_z)
        #print(f"omega is shape {omega_x.shape}")
        #print(f"omegahat is shape {omega_x_hat.shape}")    # Should not be the same as omega_x.shape

        # omega_hat_bL
        omega_x_hat_bL = alpha/np.sqrt(l_corr) * T_large * omega_x_hat
        omega_y_hat_bL = alpha/np.sqrt(l_corr) * T_large * omega_y_hat
        omega_z_hat_bL = alpha/np.sqrt(l_corr) * T_large * omega_z_hat

        # omega_hat_bS
        omega_x_hat_bS = alpha/np.sqrt(eta) * T_small * omega_x_hat
        omega_y_hat_bS = alpha/np.sqrt(eta) * T_small * omega_y_hat
        omega_z_hat_bS = alpha/np.sqrt(eta) * T_small * omega_z_hat

        omega_x_bL = Operations().irfft(omega_x_hat_bL)
        omega_y_bL = Operations().irfft(omega_y_hat_bL)
        omega_z_bL = Operations().irfft(omega_z_hat_bL)

        # u dot grad omega_bS = ux * domdx_bS + uy * domdy_bS
        dom_x_dx_bS = Operations().irfft(1j * Kx * omega_x_hat_bS)
        dom_x_dy_bS = Operations().irfft(1j * Ky * omega_x_hat_bS)
        dom_x_dz_bS = Operations().irfft(1j * Kz * omega_x_hat_bS)

        dom_y_dx_bS = Operations().irfft(1j * Kx * omega_y_hat_bS)
        dom_y_dy_bS = Operations().irfft(1j * Ky * omega_y_hat_bS)
        dom_y_dz_bS = Operations().irfft(1j * Kz * omega_y_hat_bS)

        dom_z_dx_bS = Operations().irfft(1j * Kx * omega_z_hat_bS)
        dom_z_dy_bS = Operations().irfft(1j * Ky * omega_z_hat_bS)
        dom_z_dz_bS = Operations().irfft(1j * Kz * omega_z_hat_bS)

        u_dot_grad_om_x_bS = ux * dom_x_dx_bS + uy * dom_x_dy_bS + uz * dom_x_dz_bS
        u_dot_grad_om_y_bS = ux * dom_y_dx_bS + uy * dom_y_dy_bS + uz * dom_y_dz_bS
        u_dot_grad_om_z_bS = ux * dom_z_dx_bS + uy * dom_z_dy_bS + uz * dom_z_dz_bS

        inner_F = omega_x_bL * u_dot_grad_om_x_bS + omega_y_bL * u_dot_grad_om_y_bS + omega_z_bL * u_dot_grad_om_z_bS
        #print(inner_F.shape)   # Confirmed (32, 32, 32)

        # Transfer of enstrophy F_L->S = < omega_bL dot u_dot_grad_om_bS >
        F_LtS = np.average(inner_F)
        #print(F_LtS)   # Returns scalar, as expected

        return F_LtS