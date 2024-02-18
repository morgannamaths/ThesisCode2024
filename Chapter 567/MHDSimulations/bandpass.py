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
    
    def BandpassFilter(self, uxhat, uyhat, uzhat, Bxhat, Byhat, Bzhat, Kx, Ky, Kz, l_corr, eta, alpha, T_large, T_small):
        uxhat_bL = alpha/np.sqrt(l_corr) * T_large * uxhat
        uyhat_bL = alpha/np.sqrt(l_corr) * T_large * uyhat
        uzhat_bL = alpha/np.sqrt(l_corr) * T_large * uzhat

        Bxhat_bL = alpha/np.sqrt(l_corr) * T_large * Bxhat
        Byhat_bL = alpha/np.sqrt(l_corr) * T_large * Byhat
        Bzhat_bL = alpha/np.sqrt(l_corr) * T_large * Bzhat

        ux_bL = Operations().irfft(uxhat_bL)
        uy_bL = Operations().irfft(uyhat_bL)
        uz_bL = Operations().irfft(uzhat_bL)

        Bx_bL = Operations().irfft(Bxhat_bL)
        By_bL = Operations().irfft(Byhat_bL)
        Bz_bL = Operations().irfft(Bzhat_bL)

        Sij_L = Operations().Sij(uxhat_bL, uyhat_bL, uzhat_bL, Kx, Ky, Kz)

        # Bandpass Filter for Small Scales
        uxhat_bS = alpha/np.sqrt(eta) * T_small * uxhat
        uyhat_bS = alpha/np.sqrt(eta) * T_small * uyhat
        uzhat_bS = alpha/np.sqrt(eta) * T_small * uzhat

        Bxhat_bS = alpha/np.sqrt(eta) * T_small * Bxhat
        Byhat_bS = alpha/np.sqrt(eta) * T_small * Byhat
        Bzhat_bS = alpha/np.sqrt(eta) * T_small * Bzhat

        ux_bS = Operations().irfft(uxhat_bS)
        uy_bS = Operations().irfft(uyhat_bS)
        uz_bS = Operations().irfft(uzhat_bS)

        Bx_bS = Operations().irfft(Bxhat_bS)
        By_bS = Operations().irfft(Byhat_bS)
        Bz_bS = Operations().irfft(Bzhat_bS)

        B_bL = np.array([Bx_bL, By_bL, Bz_bL])
        B_bS = np.array([Bx_bS, By_bS, Bz_bS])

        Sij_S = Operations().Sij(uxhat_bS, uyhat_bS, uzhat_bS, Kx, Ky, Kz)

        return ux_bL, uy_bL, uz_bL, ux_bS, uy_bS, uz_bS, B_bL, B_bS, Sij_L, Sij_S
    
    def BandpassFilter_BOAField(self, Bxhat, Byhat, Bzhat, axhat, ayhat, azhat, omega_x_hat, omega_y_hat, omega_z_hat, l_corr, eta, alpha, T_large, T_small):
        # B
        Bxhat_bL = alpha/np.sqrt(l_corr) * T_large * Bxhat
        Byhat_bL = alpha/np.sqrt(l_corr) * T_large * Byhat
        Bzhat_bL = alpha/np.sqrt(l_corr) * T_large * Bzhat

        Bx_bL = Operations().irfft(Bxhat_bL)
        By_bL = Operations().irfft(Byhat_bL)
        Bz_bL = Operations().irfft(Bzhat_bL)

        Bxhat_bS = alpha/np.sqrt(eta) * T_small * Bxhat
        Byhat_bS = alpha/np.sqrt(eta) * T_small * Byhat
        Bzhat_bS = alpha/np.sqrt(eta) * T_small * Bzhat

        Bx_bS = Operations().irfft(Bxhat_bS)
        By_bS = Operations().irfft(Byhat_bS)
        Bz_bS = Operations().irfft(Bzhat_bS)

        B_bL = np.array([Bx_bL, By_bL, Bz_bL])
        B_bS = np.array([Bx_bS, By_bS, Bz_bS])

        # a - large scale only
        axhat_bL = alpha/np.sqrt(l_corr) * T_large * axhat
        ayhat_bL = alpha/np.sqrt(l_corr) * T_large * ayhat
        azhat_bL = alpha/np.sqrt(l_corr) * T_large * azhat

        ax_bL = Operations().irfft(axhat_bL)
        ay_bL = Operations().irfft(ayhat_bL)
        az_bL = Operations().irfft(azhat_bL)

        axhat_bS = alpha/np.sqrt(eta) * T_small * axhat
        ayhat_bS = alpha/np.sqrt(eta) * T_small * ayhat
        azhat_bS = alpha/np.sqrt(eta) * T_small * azhat

        ax_bS = Operations().irfft(axhat_bS)
        ay_bS = Operations().irfft(ayhat_bS)
        az_bS = Operations().irfft(azhat_bS)

        a_bL = np.array([ax_bL, ay_bL, az_bL])
        a_bS = np.array([ax_bS, ay_bS, az_bS])

        # omega - large scale only
        omega_x_hat_bL = alpha/np.sqrt(l_corr) * T_large * omega_x_hat
        omega_y_hat_bL = alpha/np.sqrt(l_corr) * T_large * omega_y_hat
        omega_z_hat_bL = alpha/np.sqrt(l_corr) * T_large * omega_z_hat

        omx_bL = Operations().irfft(omega_x_hat_bL)
        omy_bL = Operations().irfft(omega_y_hat_bL)
        omz_bL = Operations().irfft(omega_z_hat_bL)

        omega_x_hat_bS = alpha/np.sqrt(eta) * T_small * omega_x_hat
        omega_y_hat_bS = alpha/np.sqrt(eta) * T_small * omega_y_hat
        omega_z_hat_bS = alpha/np.sqrt(eta) * T_small * omega_z_hat

        omx_bS = Operations().irfft(omega_x_hat_bS)
        omy_bS = Operations().irfft(omega_y_hat_bS)
        omz_bS = Operations().irfft(omega_z_hat_bS)

        omega_bL = np.array([omx_bL, omy_bL, omz_bL])
        omega_bS = np.array([omx_bS, omy_bS, omz_bS])

        #print(f"B_bL size is {B_bL.shape}")
        #print(f"Bx_bL size is {B_bL[0].shape}")
        #print(f"By_bL size is {B_bL[1].shape}")
        #print(f"Bz_bL size is {B_bL[2].shape}")

        return B_bL, B_bS, a_bL, a_bS, omega_bL, omega_bS
    
    def BandpassFilter2D(self, uxhat, uyhat, Bxhat, Byhat, Kx, Ky, l_corr, eta, alpha, T_large, T_small):
        uxhat_bL = alpha/np.sqrt(l_corr) * T_large * uxhat
        uyhat_bL = alpha/np.sqrt(l_corr) * T_large * uyhat

        Bxhat_bL = alpha/np.sqrt(l_corr) * T_large * Bxhat
        Byhat_bL = alpha/np.sqrt(l_corr) * T_large * Byhat

        ux_bL = Operations().irfft(uxhat_bL)
        uy_bL = Operations().irfft(uyhat_bL)

        Bx_bL = Operations().irfft(Bxhat_bL)
        By_bL = Operations().irfft(Byhat_bL)

        Sij_L = Operations().Sij2D(uxhat_bL, uyhat_bL, Kx, Ky)

        # Bandpass Filter for Small Scales
        uxhat_bS = alpha/np.sqrt(eta) * T_small * uxhat
        uyhat_bS = alpha/np.sqrt(eta) * T_small * uyhat

        Bxhat_bS = alpha/np.sqrt(eta) * T_small * Bxhat
        Byhat_bS = alpha/np.sqrt(eta) * T_small * Byhat

        ux_bS = Operations().irfft(uxhat_bS)
        uy_bS = Operations().irfft(uyhat_bS)

        Bx_bS = Operations().irfft(Bxhat_bS)
        By_bS = Operations().irfft(Byhat_bS)

        B_bL = np.array([Bx_bL, By_bL])
        B_bS = np.array([Bx_bS, By_bS])

        Sij_S = Operations().Sij2D(uxhat_bS, uyhat_bS, Kx, Ky)

        return ux_bL, uy_bL, ux_bS, uy_bS, B_bL, B_bS, Sij_L, Sij_S

    def Pi_u(self, ux_bL, uy_bL, uz_bL, ux_bS, uy_bS, uz_bS, B_bL, B_bS, Sij_L, Sij_S, mu0):
        
        Tauij_L = Operations().Tauij(ux_bL, uy_bL, uz_bL)
        Tauij_S = Operations().Tauij(ux_bS, uy_bS, uz_bS)

        Pi_u = Operations().Pi_u(Sij_L, Sij_S, Tauij_L, Tauij_S, B_bL, B_bS, mu0)
        #print(f"Pi_LtS = {Pi_LtS}")    # Returns scalar, as expected

        return Pi_u
    
    def Pi_u_2D(self, ux_bL, uy_bL, ux_bS, uy_bS, B_bL, B_bS, Sij_L, Sij_S, mu0):
        
        Tauij_L = Operations().Tauij2D(ux_bL, uy_bL)
        Tauij_S = Operations().Tauij2D(ux_bS, uy_bS)

        Pi_u = Operations().Pi_u_2D(Sij_L, Sij_S, Tauij_L, Tauij_S, B_bL, B_bS, mu0)
        #print(f"Pi_u = {Pi_u}")    # Returns scalar, as expected

        return Pi_u
    
    def Pi_b(self, ux, uy, uz, B_bL, B_bS, Sij_L, Sij_S, Kx, Ky, Kz):
        # Need grad B_bL
        Bx_hat_bL = Operations().rfft(B_bL[0])
        By_hat_bL = Operations().rfft(B_bL[1])
        Bz_hat_bL = Operations().rfft(B_bL[2])

        inner_product = Operations().A_dot_B_dot_gradC(B_bS, (ux, uy, uz), (Bx_hat_bL, By_hat_bL, Bz_hat_bL), Kx, Ky, Kz)

        BBS_LLS = Operations().Frobenius3D_3Terms(B_bL, B_bL, Sij_S)
        BBS_LSL = Operations().Frobenius3D_3Terms(B_bL, B_bS, Sij_L)
        BBS_LSS = Operations().Frobenius3D_3Terms(B_bL, B_bS, Sij_S)

        return np.average(inner_product) + np.average(BBS_LLS) + np.average(BBS_LSL) + np.average(BBS_LSS)
    
    def Pi_b_2D(self, ux, uy, B_bL, B_bS, Sij_L, Sij_S, Kx, Ky):
        # Need grad B_bL
        Bx_hat_bL = Operations().rfft(B_bL[0])
        By_hat_bL = Operations().rfft(B_bL[1])

        inner_product = Operations().A_dot_B_dot_gradC_2D(B_bS, (ux, uy), (Bx_hat_bL, By_hat_bL), Kx, Ky)

        BBS_LLS = Operations().Frobenius2D_3Terms(B_bL, B_bL, Sij_S)
        BBS_LSL = Operations().Frobenius2D_3Terms(B_bL, B_bS, Sij_L)
        BBS_LSS = Operations().Frobenius2D_3Terms(B_bL, B_bS, Sij_S)

        return np.average(inner_product) + np.average(BBS_LLS) + np.average(BBS_LSL) + np.average(BBS_LSS)

    
    def F_u(self, ux, uy, uz, omega_x_hat, omega_y_hat, omega_z_hat, Jx, Jy, Jz, B_bL, B_bS, Kx, Ky, Kz, l_corr, eta, alpha, T_large, T_small):
        # omega_hat_bL
        omega_x_hat_bL = alpha/np.sqrt(l_corr) * T_large * omega_x_hat
        omega_y_hat_bL = alpha/np.sqrt(l_corr) * T_large * omega_y_hat
        omega_z_hat_bL = alpha/np.sqrt(l_corr) * T_large * omega_z_hat

        # omega_hat_bS
        omega_x_hat_bS = alpha/np.sqrt(eta) * T_small * omega_x_hat
        omega_y_hat_bS = alpha/np.sqrt(eta) * T_small * omega_y_hat
        omega_z_hat_bS = alpha/np.sqrt(eta) * T_small * omega_z_hat

        Jx_hat = Operations().rfft(Jx)
        Jy_hat = Operations().rfft(Jy)
        Jz_hat = Operations().rfft(Jz)

        # J_hat_bL
        J_x_hat_bL = alpha/np.sqrt(l_corr) * T_large * Jx_hat
        J_y_hat_bL = alpha/np.sqrt(l_corr) * T_large * Jy_hat
        J_z_hat_bL = alpha/np.sqrt(l_corr) * T_large * Jz_hat

        # J_hat_bS
        J_x_hat_bS = alpha/np.sqrt(eta) * T_small * Jx_hat
        J_y_hat_bS = alpha/np.sqrt(eta) * T_small * Jy_hat
        J_z_hat_bS = alpha/np.sqrt(eta) * T_small * Jz_hat

        omega_x_bL = Operations().irfft(omega_x_hat_bL)
        omega_y_bL = Operations().irfft(omega_y_hat_bL)
        omega_z_bL = Operations().irfft(omega_z_hat_bL)

        Jx_bL = Operations().irfft(J_x_hat_bL)
        Jy_bL = Operations().irfft(J_y_hat_bL)
        Jz_bL = Operations().irfft(J_z_hat_bL)

        Jx_bS = Operations().irfft(J_x_hat_bS)
        Jy_bS = Operations().irfft(J_y_hat_bS)
        Jz_bS = Operations().irfft(J_z_hat_bS)

        inner_F = Operations().A_dot_B_dot_gradC((omega_x_bL, omega_y_bL, omega_z_bL), (ux, uy, uz), (omega_x_hat_bS, omega_y_hat_bS, omega_z_hat_bS), Kx, Ky, Kz)

        JBom_SLL = Operations().A_dot_B_dot_gradC((Jx_bS, Jy_bS, Jz_bS), B_bL, (omega_x_hat_bL, omega_y_hat_bL, omega_z_hat_bL), Kx, Ky, Kz)
        JBom_LSL = Operations().A_dot_B_dot_gradC((Jx_bL, Jy_bL, Jz_bL), B_bS, (omega_x_hat_bL, omega_y_hat_bL, omega_z_hat_bL), Kx, Ky, Kz)
        JBom_SSL = Operations().A_dot_B_dot_gradC((Jx_bS, Jy_bS, Jz_bS), B_bS, (omega_x_hat_bL, omega_y_hat_bL, omega_z_hat_bL), Kx, Ky, Kz)
        BJom_SLL = Operations().A_dot_B_dot_gradC(B_bS, (Jx_bL, Jy_bL, Jz_bL), (omega_x_hat_bL, omega_y_hat_bL, omega_z_hat_bL), Kx, Ky, Kz)
        BJom_LSL = Operations().A_dot_B_dot_gradC(B_bL, (Jx_bS, Jy_bS, Jz_bS), (omega_x_hat_bL, omega_y_hat_bL, omega_z_hat_bL), Kx, Ky, Kz)
        BJom_SSL = Operations().A_dot_B_dot_gradC(B_bS, (Jx_bS, Jy_bS, Jz_bS), (omega_x_hat_bL, omega_y_hat_bL, omega_z_hat_bL), Kx, Ky, Kz)

        return np.average(inner_F) + np.average(JBom_SLL) + np.average(JBom_LSL) + np.average(JBom_SSL) - np.average(BJom_SLL) - np.average(BJom_LSL) - np.average(BJom_SSL)
    
    def F_u_2D(self, ux, uy, zerosReal, zerosComplex, omega_z_hat, Jz, B_bL, B_bS, Kx, Ky, l_corr, eta, alpha, T_large, T_small):
        # omega_hat_bL
        omega_x_hat_bL = zerosComplex
        omega_y_hat_bL = zerosComplex
        omega_z_hat_bL = alpha/np.sqrt(l_corr) * T_large * omega_z_hat

        # omega_hat_bS
        omega_x_hat_bS = zerosComplex
        omega_y_hat_bS = zerosComplex
        omega_z_hat_bS = alpha/np.sqrt(eta) * T_small * omega_z_hat

        Jx_hat = zerosComplex
        Jy_hat = zerosComplex
        Jz_hat = Operations().rfft(Jz)

        # J_hat_bL
        J_x_hat_bL = alpha/np.sqrt(l_corr) * T_large * Jx_hat
        J_y_hat_bL = alpha/np.sqrt(l_corr) * T_large * Jy_hat
        J_z_hat_bL = alpha/np.sqrt(l_corr) * T_large * Jz_hat

        # J_hat_bS
        J_x_hat_bS = alpha/np.sqrt(eta) * T_small * Jx_hat
        J_y_hat_bS = alpha/np.sqrt(eta) * T_small * Jy_hat
        J_z_hat_bS = alpha/np.sqrt(eta) * T_small * Jz_hat

        omega_x_bL = Operations().irfft(omega_x_hat_bL)
        omega_y_bL = Operations().irfft(omega_y_hat_bL)
        omega_z_bL = Operations().irfft(omega_z_hat_bL)

        Jx_bL = Operations().irfft(J_x_hat_bL)
        Jy_bL = Operations().irfft(J_y_hat_bL)
        Jz_bL = Operations().irfft(J_z_hat_bL)

        Jx_bS = Operations().irfft(J_x_hat_bS)
        Jy_bS = Operations().irfft(J_y_hat_bS)
        Jz_bS = Operations().irfft(J_z_hat_bS)

        B_bL3 = np.array([B_bL[0], B_bL[1], zerosReal])
        B_bS3 = np.array([B_bS[0], B_bS[1], zerosReal])

        inner_F = Operations().A_dot_B_dot_gradC((omega_x_bL, omega_y_bL, omega_z_bL), (ux, uy, zerosReal), (omega_x_hat_bS, omega_y_hat_bS, omega_z_hat_bS), Kx, Ky, 0)

        JBom_SLL = Operations().A_dot_B_dot_gradC((Jx_bS, Jy_bS, Jz_bS), B_bL3, (omega_x_hat_bL, omega_y_hat_bL, omega_z_hat_bL), Kx, Ky, 0)
        JBom_LSL = Operations().A_dot_B_dot_gradC((Jx_bL, Jy_bL, Jz_bL), B_bS3, (omega_x_hat_bL, omega_y_hat_bL, omega_z_hat_bL), Kx, Ky, 0)
        JBom_SSL = Operations().A_dot_B_dot_gradC((Jx_bS, Jy_bS, Jz_bS), B_bS3, (omega_x_hat_bL, omega_y_hat_bL, omega_z_hat_bL), Kx, Ky, 0)

        return -np.average(inner_F) - np.average(JBom_SLL) - np.average(JBom_LSL) - np.average(JBom_SSL)
    
    def F_b(self, ux, uy, uz, a_x_hat, a_y_hat, a_z_hat, Kx, Ky, Kz, l_corr, eta, alpha, T_large, T_small):
        # a_hat_bL
        a_x_hat_bL = alpha/np.sqrt(l_corr) * T_large * a_x_hat
        a_y_hat_bL = alpha/np.sqrt(l_corr) * T_large * a_y_hat
        a_z_hat_bL = alpha/np.sqrt(l_corr) * T_large * a_z_hat

        # a_hat_bS
        a_x_hat_bS = alpha/np.sqrt(eta) * T_small * a_x_hat
        a_y_hat_bS = alpha/np.sqrt(eta) * T_small * a_y_hat
        a_z_hat_bS = alpha/np.sqrt(eta) * T_small * a_z_hat

        # a_bS
        a_x_bS = Operations().irfft(a_x_hat_bS)
        a_y_bS = Operations().irfft(a_y_hat_bS)
        a_z_bS = Operations().irfft(a_z_hat_bS)

        #a_bS dot u dot grada_bL
        inner_F = Operations().A_dot_B_dot_gradC((a_x_bS, a_y_bS, a_z_bS), (ux, uy, uz), (a_x_hat_bL, a_y_hat_bL, a_z_hat_bL), Kx, Ky, Kz)

        return -np.average(inner_F)