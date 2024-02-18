import numpy as np
import matplotlib.pyplot as plt

def FourierExamples(N, upper_limit):
    x = np.arange(0, upper_limit, upper_limit/N)
    y = 3 * np.sin(2*x) + 2 * np.cos(4*x)

    plt.figure(figsize=(5.5, 4))
    plt.plot(x, y)
    plt.title(r"Function in $x$-space", fontsize=14)
    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$f(x)$", fontsize=12)

    yhat = np.fft.fftn(y, norm="forward")
    k = np.fft.fftfreq(N, 1/N)

    plt.figure(figsize=(5.5, 4))
    plt.plot(k, np.abs(yhat))
    plt.title(r"Function in $k$-space", fontsize=14)
    plt.xlabel(r"$k$", fontsize=12)
    plt.ylabel(r"$\hat{f}(k)$", fontsize=12)

    yhat_re = np.fft.rfftn(y, norm="forward")
    k_re = np.fft.rfftfreq(N, 1/N)

    plt.figure(figsize=(5.5, 4))
    plt.plot(k_re, np.abs(yhat_re))
    plt.title(r"Real function in $k$-space", fontsize=14)
    plt.xlabel(r"$k$", fontsize=12)
    plt.ylabel(r"$\hat{f}(k)$", fontsize=12)

def DrawAxes(boundary):
    plt.plot([-boundary, boundary], [0, 0], color="black")
    plt.plot([0, 0], [-boundary, boundary], color="black")
    for i in range(boundary):
        plt.plot([-0.1, 0.1], [i, i], color="black")
        plt.plot([i, i], [-0.1, 0.1], color="black")
        plt.plot([-0.1, 0.1], [-i, -i], color="black")
        plt.plot([-i, -i], [-0.1, 0.1], color="black")

def RedundantNodes2D(steps, boundary):
    k = np.linspace(-boundary, boundary, steps)

    plt.figure(figsize=(8, 6))
    DrawAxes(boundary)
    plt.plot(k, -25*k**3)
    plt.ylim((-boundary, boundary))
    plt.title(r"2D $k$-space", fontsize=20)
    plt.xlabel(r"$k_x$", fontsize=16)
    plt.ylabel(r"$k_y$", fontsize=16)

if __name__ == "__main__":
    #FourierExamples(32, 2 * np.pi)
    RedundantNodes2D(1024, 8)
    plt.show()