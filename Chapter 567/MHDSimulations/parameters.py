import numpy as np
import h5py

class Parameters():

    ux = 0
    uy = 0
    uz = 0

    Bx = 0
    By = 0
    Bz = 0

    B0z = 0
    visc = 0
    resist = 0

    alpha = np.sqrt(2)
    time = 0

    nx = 0
    ny = 0
    nz = 0

    def __init__(self, filenameU, filenameB):
        fU = h5py.File(filenameU, "r")
        fB = h5py.File(filenameB, "r")
        #print(list(fU.keys())) gives "mhd3d-mpi"
        #print(list(fB.keys())) gives "mhd3d-mpi"

        dataU = fU["mhd3d-mpi"]
        dataB = fB["mhd3d-mpi"]
        #print(list(dataU.keys())) gives "vx", "vy", "vz"
        #print(list(dataB.keys())) gives "bx", "by", "bz"

        vx = dataU["vx"][:]
        vy = dataU["vy"][:]
        vz = dataU["vz"][:]
        bx = dataB["bx"][:]
        by = dataB["by"][:]
        bz = dataB["bz"][:]

        # Data is (n, n, n+2)
        # Make data (n, n, n)
        self.ux = np.delete(vx, [vx.shape[2]-2, vx.shape[2]-1], 2)
        self.uy = np.delete(vy, [vy.shape[2]-2, vy.shape[2]-1], 2)
        self.uz = np.delete(vz, [vz.shape[2]-2, vz.shape[2]-1], 2)
        self.Bx = np.delete(bx, [bx.shape[2]-2, bx.shape[2]-1], 2)
        self.By = np.delete(by, [by.shape[2]-2, by.shape[2]-1], 2)
        self.Bz = np.delete(bz, [bz.shape[2]-2, bz.shape[2]-1], 2)

        # Access attributes
        paramsU = dataU.attrs["params"][:]
        self.time = dataU.attrs["time"]
        print(f"Snapshot as time = {self.time}")
        #params_order = data.attrs["params-order"][:]
        self.B0z = paramsU[0]
        self.visc = paramsU[1]
        self.resist = paramsU[2]

        # Setup nx, ny, nz
        self.nx = len(self.ux)
        self.ny = len(self.uy)
        self.nz = len(self.uz)

    def fields(self):
        return self.ux, self.uy, self.uz, self.Bx, self.By, self.Bz

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

if __name__ == "__main__":
    Parameters("mhd3d-32/r1-mhd3-00-uno-Xsp-v.h5", "mhd3d-32/r1-mhd3-00-uno-Xsp-b.h5")
