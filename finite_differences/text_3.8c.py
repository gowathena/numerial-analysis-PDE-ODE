import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 100/np.sqrt(x + 0.01) + 1/((x - 0.3)**2 + 0.001) - np.pi

def S(a, b, fa, fm, fb):
    return (b - a) * (fa + 4*fm + fb) / 6.0

def adaptive_simpson(a=0.0, b=1.0, tol=1e-8, max_depth=30):
    xs = [a, 0.5*(a+b), b]  

    def rec(a, b, fa, fm, fb, Sab, tol, depth):
        m = 0.5*(a + b)
        l = 0.5*(a + m)
        r = 0.5*(m + b)

        fl, fr = f(l), f(r)
        xs.extend([l, r])

        Sleft  = S(a, m, fa, fl, fm)
        Sright = S(m, b, fm, fr, fb)

        err = Sleft + Sright - Sab

        if depth >= max_depth or abs(err) < 15*tol:
            return Sleft + Sright + err/15.0

        return (rec(a, m, fa, fl, fm, Sleft,  tol/2, depth+1) +
                rec(m, b, fm, fr, fb, Sright, tol/2, depth+1))

    fa, fb = f(a), f(b)
    m = 0.5*(a + b)
    fm = f(m)

    I0 = S(a, b, fa, fm, fb)
    I = rec(a, b, fa, fm, fb, I0, tol, 0)

    return I, np.unique(xs)


I, xs = adaptive_simpson(a=0.0, b=1.0, tol=1e-8)

print("Adaptive I =", I)

x_plot = np.linspace(0, 1, 2000)

plt.plot(x_plot, f(x_plot), label="f(x)")
plt.plot(xs, f(xs), 'o', markersize=3, label="evaluation points")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()