import numpy as np
import h5py

class Parameters2D():
    
    a_z = 0
    omega_z = 0

    B0z = 0
    visc = 0
    resist = 0

    alpha = np.sqrt(2)

    time = 0

    nx = 0
    ny = 0

    def __init__(self, filename):
        f = h5py.File(filename, "r")
        #print(list(f.keys())) gives "aa" and "om

        aa = f["aa"]
        om = f["om"]
        #print(aa.shape) #gives (512, 514)
        #print(om.shape) #gives (512, 514)

        # Data is (n, n+2)
        # Make data (n, n)
        self.a_z = np.delete(aa, [aa.shape[1]-2, aa.shape[1]-1], 1)
        self.omega_z = np.delete(om, [om.shape[1]-2, om.shape[1]-1], 1)

        # Access attributes
        paramsU = f.attrs["params"][:]
        self.time = f.attrs["time"]
        print(f"Snapshot at time = {self.time}")
        #params_order = f.attrs["params-order"][:]
        self.B0z = paramsU[1]
        self.visc = paramsU[2]
        print(self.visc)
        self.resist = paramsU[3]

        # Setup nx, ny
        self.nx = self.a_z.shape[0]
        self.ny = self.a_z.shape[1]

    def fields(self):
        return self.a_z, self.omega_z

    def setup(self):
        if (self.nx == self.ny):
            return self.nx
        else:
            return self.nx, self.ny

    def space(self, limit):
        if (self.nx == self.ny):
            x = y = np.arange(0, limit, limit/self.nx)
            # Setup k-space with half of kx redundant
            kx = np.fft.rfftfreq(self.nx, 1/self.nx)
            ky = np.fft.fftfreq(self.nx, 1/self.nx)
            # Make ky go 0 to 16 then -15 to -1 so it is consitent with kx
            ky[int(self.nx/2)] = self.nx/2
        else:
            x = np.arange(0, limit, limit/self.nx)
            y = np.arange(0, limit, limit/self.ny)
            # Setup k-space with half of kx redundant
            kx = np.fft.rfftfreq(self.nx, 1/self.nx)
            ky = np.fft.fftfreq(self.ny, 1/self.ny)
            # Make ky go 0 to 16 then -15 to -1 so it is consitent with kx
            ky[int(self.ny/2)] = self.ny/2
        return x, y, kx, ky

    def Taylor2D(self, ux, uy):
        v_sqd = ux**2 + uy**2 
        v_av = np.average(v_sqd)
        omega_av = np.average(self.omega_z**2)
        return np.sqrt(5 * v_av/omega_av)

if __name__ == "__main__":
    Parameters2D("mhd2d-n512/dyn11c-000.h5")
