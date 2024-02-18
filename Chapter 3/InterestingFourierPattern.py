import numpy as np
import matplotlib.pyplot as plt

def FourierPattern(N):
    omega_N = np.exp((- 2j * np.pi)/N)
    M, K = np.meshgrid(np.arange(N), np.arange(N))
    F_N = omega_N**(M*K)
    return np.real(F_N)

def Graphing(N1, N2):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(10, 4)
    axs[0].set_title(r"Interesting Pattern in $F_N$, for $N =$" + f"{N1}", fontsize=16)
    axs[0].imshow(FourierPattern(N1))
    axs[1].set_title(r"Interesting Pattern in $F_N$, for $N =$" + f"{N2}", fontsize=16)
    axs[1].imshow(FourierPattern(N2))

    plt.show()

if __name__ == "__main__":
    Graphing(32, 128)