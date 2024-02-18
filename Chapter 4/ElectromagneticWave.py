import numpy as np
import vpython as vp

def GenerateCanvas():
    vp.canvas(width=800, height=600, background=vp.color.white)

def EMWave(lam, freq, x_min, x_max, dx, t, Emax, Bmax, scale):
    E_arrows = []
    B_arrows = []
    x = np.arange(x_min, x_max, 0.1)

    for element in x:
        E = Emax * np.cos(2 * np.pi * (freq * t - element/lam))
        B = Bmax * np.cos(2 * np.pi * (freq * t - element/lam))
        E_arrows.append([vp.arrow(pos=vp.vector(element, 0, 0), axis=vp.vector(0, scale * E, 0), color=vp.color.red, shaftwidth=0.02)])
        B_arrows.append([vp.arrow(pos=vp.vector(element, 0, 0), axis=vp.vector(0, 0, scale * B), color=vp.color.cyan, shaftwidth=0.02)])

if __name__ == "__main__":
    GenerateCanvas()
    EMWave(1, 1, 0, 10, 0.1, 0, 0.02, 0.02, 10)