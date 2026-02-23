import numpy as np
import matplotlib.pyplot as plt

def simpson_weights(a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even")
    x = np.linspace(a, b, n+1)
    h = (b-a)/n
    w = np.ones(n+1)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    w *= h/3
    return x, w

def solve_fredholm(n=200):
    a, b = 0, np.pi
    t, w = simpson_weights(a, b, n)

    f = lambda x: np.pi * x**2
    K = lambda x, s: 3*(0.5*np.sin(3*x) - s*x**2)

    N = len(t)
    A = np.eye(N)
    for i in range(N):
        A[i,:] -= w * K(t[i], t)

    phi = np.linalg.solve(A, f(t))
    exact = np.sin(3*t)

    err = np.max(np.abs(phi - exact))
    print("max error =", err)

    plt.plot(t, phi, 'o', markersize=3, label="numerical")
    plt.plot(t, exact, '-', label="exact")
    plt.legend()
    plt.grid()
    plt.show()

solve_fredholm(200)