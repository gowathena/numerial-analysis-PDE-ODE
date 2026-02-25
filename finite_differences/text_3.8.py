import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 100/np.sqrt(x + 0.01) + 1/((x - 0.3)**2 + 0.001) - np.pi

def trap(fx, h):
    return h * (0.5*fx[0] + fx[1:-1].sum() + 0.5*fx[-1])

def simpson(fx, h):
    n = len(fx) - 1
    if n % 2:
        raise ValueError("Simpson needs even n.")
    w = np.ones(n+1)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    return (h/3) * (w @ fx)

def end_corr_trap(fx, h):
    IT = trap(fx, h)

    f0, f1, f2, f3, f4 = fx[:5]
    fn, fn1, fn2, fn3, fn4 = fx[-1], fx[-2], fx[-3], fx[-4], fx[-5]

    fp0 = (-25*f0 + 48*f1 - 36*f2 + 16*f3 - 3*f4) / (12*h)
    fp1 = ( 25*fn - 48*fn1 + 36*fn2 - 16*fn3 + 3*fn4) / (12*h)

    return IT - (h**2/12) * (fp1 - fp0)

def compute_rules(n):
    x = np.linspace(0.0, 1.0, n+1)
    fx = f(x)
    h = 1.0/n
    return (
        trap(fx, h),
        simpson(fx, h),
        end_corr_trap(fx, h),
    )

# reference integral (fine Simpson)
n_ref = 2**16
I_ref = compute_rules(n_ref)[1]

ns = 2**np.arange(3, 11)  # 8..1024
errT, errS, errTc = [], [], []

for n in ns:
    IT, IS, ITc = compute_rules(int(n))
    errT.append(abs(IT  - I_ref)/abs(I_ref))
    errS.append(abs(IS  - I_ref)/abs(I_ref))
    errTc.append(abs(ITc - I_ref)/abs(I_ref))

plt.loglog(ns, errT,  "o-", label="Trap")
plt.loglog(ns, errS,  "s-", label="Simpson")
plt.loglog(ns, errTc, "d-", label="Trap + end")
plt.xlabel("n (panels)")
plt.ylabel("relative error")
plt.grid(True, which="both")
plt.legend()
plt.show()