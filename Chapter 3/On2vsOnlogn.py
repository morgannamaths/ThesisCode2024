import numpy as np
import matplotlib.pyplot as plt

n = 1024
N = np.arange(0.1, 100, 100/n)

On2 = N**2
Onlogn = N * np.log(N)

plt.figure(figsize=(8, 6))
plt.plot(N, On2, label=r"$\mathcal{O}(N^2)$")
plt.plot(N, Onlogn, label=r"$\mathcal{O}(n\log(n))$")
plt.title(r"Comparison of $\mathcal{O}(N^2)$ and $\mathcal{O}(n\log(n))$ complexity", fontsize = 18)
plt.xlabel(r"Value of $N$", fontsize=14)
plt.ylabel("Complexity", fontsize=14)
plt.legend()
plt.show()