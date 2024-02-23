import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

plot_every = 16

def FluidFlow():
    mask, W, H = Mask()
    visc = 0.54
    time = 2**(17)

    # Lattice speeds and weights
    Lattice_quanity = 9
    vx = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    vy = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # Initial conditions
    F = np.ones((H, W, Lattice_quanity)) + 0.1 * np.random.randn(H, W, Lattice_quanity)
    # Move fluid to the right (corresponds to the 3rd node in the lattice)
    F[:, :, 3] = 2.3

    # main loop
    j = 1
    for t in range(time):
        print(t)

        # Zhu Hae boundary conditions (prevent waves bouncing over bounding walls)
        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        for i, cx, cy in zip(range(Lattice_quanity), vx, vy):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)
        
        boundary = F[mask, :]
        boundary = boundary[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * vx, 2)/rho
        uy = np.sum(F * vy, 2)/rho

        # Apply boundary
        F[mask, :] = boundary
        ux[mask] = 0
        uy[mask] = 0

        # Collision
        F_equilibrium = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(Lattice_quanity), vx, vy, weights):
            F_equilibrium[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy)**2 /2 -3 * (ux**2 + uy**2)/2)
        
        F = F -(1/visc) * (F - F_equilibrium)

        if (t%plot_every == 0): # Only plot every nth point for saving computation
            # Vorticity
            #vorticity = Curl(ux, uy, W, H)
            plt.title(f"Von Karmen Vortex Streets, viscosity = {visc}")
            plt.imshow(np.sqrt(ux**2 + uy**2), cmap="plasma")
            #plt.imshow(np.abs(vorticity), cmap="coolwarm")
            #plt.savefig(f"Python Animation/FluidSimOutput/fig{str(j).zfill(5)}")
            #plt.savefig(f"Python Animation/VorticityOutput/fig{str(j).zfill(5)}")
            plt.pause(0.01)
            plt.cla()
            j += 1

def Curl(ux, uy, W, H):
    kx = np.fft.rfftfreq(W, 1/W)
    ky = np.fft.fftfreq(H, 1/H)
    Kx, Ky = np.meshgrid(kx, ky)

    uxhat = np.fft.rfftn(ux, norm="forward", axes=(0, 1))
    uyhat = np.fft.rfftn(uy, norm="forward", axes=(0, 1))
    
    duxdy = np.fft.irfftn(1j * Ky * uyhat, norm="forward", axes=(0, 1))
    duydx = np.fft.irfftn(1j * Kx * uxhat, norm="forward", axes=(0, 1))

    return duydx - duxdy

def Mask():
    #img = np.asarray(Image.open("CarTest100x400.png"))
    img = np.asarray(Image.open("Cylinder.png"))
    W, H = img.shape[1], img.shape[0]
    #plt.imshow(img)
    #plt.pause(3)
    mask = np.full((H, W), False)
    print(img.shape)
    for y in range(H):
        for x in range(W):
            if img[y][x].all() == np.array([0, 0, 0, 255]).all():
                mask[y][x] = True
    return mask, W, H


if __name__ == "__main__":
    FluidFlow()
    #Mask()