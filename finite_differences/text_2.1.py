import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.sin(5*x)

def uxx_exact(x):
    return -25*np.sin(5*x)

x0 = 1.5
exact = uxx_exact(x0)

hs = np.logspace(-4, 0, 200)
err_std = []
err_two = []

u0 = u(x0)

for h in hs:
    up  = u(x0 + h)
    um  = u(x0 - h)
    up2 = u(x0 + 2*h)
    um2 = u(x0 - 2*h)

    approx_std = (up - 2*u0 + um) / (h*h)
    approx_two = (up2 - 2*u0 + um2) / (4*h*h)

    err_std.append(abs(approx_std - exact))
    err_two.append(abs(approx_two - exact))

err_std = np.array(err_std)
err_two = np.array(err_two)

plt.figure()
plt.loglog(hs, err_std, "o-", markersize=3, label="standard")
plt.loglog(hs, err_two, "o-", markersize=3, label="two-app")

mid = len(hs)//2
C = err_std[mid] / (hs[mid]**2)
plt.loglog(hs, C*hs**2, "--", label="~h^2")

plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("absolute error")
plt.legend()
plt.tight_layout()
plt.savefig("exercise1d_errors.png", dpi=300)
plt.show()