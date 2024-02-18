import numpy as np
import h5py

class Parameters():

    ux = 0
    uy = 0
    uz = 0

    B0z = 0
    visc = 0
    resist = 0

    alpha = np.sqrt(2)

    nx = 0
    ny = 0
    nz = 0

    def __init__(self, filename):
        f = h5py.File(filename, "r")
        #print(list(f.keys())) gives "mhd3d-mpi"
        data = f["mhd3d-mpi"]
        #print(list(data.keys())) gives "vx", "vy", "vz"
        vx = data["vx"][:]
        vy = data["vy"][:]
        vz = data["vz"][:]

        # Make data (32, 32, 32)
        self.ux = np.delete(vx, [32, 33], 2)
        self.uy = np.delete(vy, [32, 33], 2)
        self.uz = np.delete(vz, [32, 33], 2)

        # Access attributes
        params = data.attrs["params"][:]
        #params_order = data.attrs["params-order"][:]
        self.B0z = params[0]
        self.visc = params[1]
        self.resist = params[2]

        # Setup nx, ny, nz
        self.nx = len(self.ux)
        self.ny = len(self.uy)
        self.nz = len(self.uz)

    def fields(self):
        return self.ux, self.uy, self.uz

    def setup(self):
        if (self.nx == self.ny and self.nx == self.nz):
            return self.nx
        else:
            return self.nx, self.ny, self.nz

    def space(self, limit):
        if (self.nx == self.ny and self.nx == self.nz):
            x = y = z = np.arange(0, limit, limit/self.nx)
            # Setup k-space with half of kx redundant
            kx = np.fft.rfftfreq(self.nx, 1/self.nx)
            ky = kz = np.fft.fftfreq(self.nx, 1/self.nx)
            # Make ky and kz go 0 to 16 then -15 to -1 so it is consitent with kx
            ky[int(self.nx/2)] = kz[int(self.nx/2)] = self.nx/2
        else:
            x = np.arange(0, limit, limit/self.nx)
            y = np.arange(0, limit, limit/self.ny)
            z = np.arange(0, limit, limit/self.nz)
            # Setup k-space with half of kx redundant
            kx = np.fft.rfftfreq(self.nx, 1/self.nx)
            ky = np.fft.fftfreq(self.ny, 1/self.ny)
            kz = np.fft.fftfreq(self.nz, 1/self.nz)
            # Make ky and kz go 0 to 16 then -15 to -1 so it is consitent with kx
            ky[int(self.ny/2)] = self.ny/2
            kz[int(self.nz/2)] = self.nz/2
        return x, y, z, kx, ky, kz

    def Taylor(self, omega):
        v_sqd = self.ux**2 + self.uy**2 + self.uz**2
        v_av = np.average(v_sqd)
        omega_av = np.average(omega**2)
        return np.sqrt(5 * v_av/omega_av)

    # omega
    # Meshes?
    # ell_corr
    # dissipation length




if __name__ == "__main__":
    print(Parameters("v32cubed.h5").visc)
