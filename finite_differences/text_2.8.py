import numpy as np
import matplotlib.pyplot as plt

def exact_y(x):
    C2 = 6.0
    C1 = (6*np.cos(1.0) - 3.0) / np.sin(1.0)
    return C1*np.cos(x) + C2*np.sin(x) + x**3 - 6*x

def solve_bvp(N=24):
    h = 1.0 / N
    x = (np.arange(1, N+1) - 0.5) * h
    x3 = x**3

    A = np.zeros((N, N))
    B = np.zeros((N, N))

    # Left boundary: (10/12)y1'' + (1/12)y2'' = ...
    A[0, 0] = 10/12
    A[0, 1] = 1/12
    B[0, 0] = -29/24
    B[0, 1] = 37/24
    B[0, 2] = -11/24
    B[0, 3] = 1/8

    # Interior rows: Eq (2.19)
    for i in range(1, N-1):
        A[i, i-1] = 1/12
        A[i, i]   = 10/12
        A[i, i+1] = 1/12
        B[i, i-1] = 1.0
        B[i, i]   = -2.0
        B[i, i+1] = 1.0

    # Right boundary: (1/12)y_{N-1}'' + (10/12)y_N'' = ...
    A[N-1, N-2] = 1/12
    A[N-1, N-1] = 10/12
    B[N-1, N-1] = -29/24
    B[N-1, N-2] = 37/24
    B[N-1, N-3] = -11/24
    B[N-1, N-4] = 1/8

    # Solve (A + B/h^2) y = A x^3
    M = A + (1/h**2)*B
    rhs = A @ x3

    y = np.linalg.solve(M, rhs)
    return x, y

N = 24
x, y = solve_bvp(N)
yex = exact_y(x)

print(f"Max error: {np.max(np.abs(y - yex)):.2e}")

plt.figure()
plt.plot(x, yex, '-', label="exact")
plt.plot(x, y, "o--", label="numerical (Padé)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("bvp_N24.png", dpi=300)
plt.show()