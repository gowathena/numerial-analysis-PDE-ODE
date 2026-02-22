import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, np.pi, 600)  # theta = k*Delta

kcd2_D = np.sin(theta)                          # (k')Delta for CD2
kpade_D = 3*np.sin(theta) / (2 + np.cos(theta)) # (k')Delta for Pade4
exact = theta                                   # exact: k' = k

plt.figure()
plt.plot(theta, exact, label="exact: k'Δ = kΔ")
plt.plot(theta, kcd2_D, label="CD2: k'Δ = sin(kΔ)")
plt.plot(theta, kpade_D, label="Pade4: k'Δ = 3 sin(kΔ)/(2+cos(kΔ))")

plt.xlabel(r"$k\Delta$")
plt.ylabel(r"$k'\Delta$")
plt.legend()
plt.tight_layout()
plt.savefig("modified_wavenumber.png", dpi=300)
plt.show()