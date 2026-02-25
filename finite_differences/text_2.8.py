import numpy as np
import matplotlib.pyplot as plt

def exact_y(x):
    C2 = 6.0
    C1 = (6*np.cos(1.0) - 3.0) / np.sin(1.0)
    return C1*np.cos(x) + C2*np.sin(x) + x**3 - 6*x

def solve_bvp(N=24, b2=0.0):
    h = 1.0 / N
    x = (np.arange(1, N+1) - 0.5) * h
    x3 = x**3

    # boundary coefficients (3rd-order family)
    a1 = (335/264)*b2 - 23/24
    a2 = 7/8 - (223/88)*b2
    a3 = 1/8 + (119/88)*b2
    a4 = -(23/264)*b2 - 1/24
    b1 = b2/11 - 1.0 

    A = np.zeros((N, N))
    B = np.zeros((N, N))
    g = np.zeros(N)

    # Left boundary row (i=1)
    A[0, 0] = 1.0
    if N > 1:
        A[0, 1] = b2
    B[0, 0] = a1
    if N > 1: B[0, 1] = a2
    if N > 2: B[0, 2] = a3
    if N > 3: B[0, 3] = a4

    # Interior rows (i=2..N-1): Eq (2.19)
    for i in range(1, N-1):
        A[i, i-1] = 1/12
        A[i, i]   = 10/12
        A[i, i+1] = 1/12

        B[i, i-1] = 1.0
        B[i, i]   = -2.0
        B[i, i+1] = 1.0

    # Right boundary row (i=N)
    A[N-1, N-1] = 1.0
    if N > 1:
        A[N-1, N-2] = b2
    B[N-1, N-1] = a1
    if N > 1: B[N-1, N-2] = a2
    if N > 2: B[N-1, N-3] = a3
    if N > 3: B[N-1, N-4] = a4

    M = A + (1/h**2)*B
    rhs = A @ x3 - g

    y = np.linalg.solve(M, rhs)
    return x, y

def main():
    N = 24
    x, y = solve_bvp(N=N, b2=0.0)
    yex = exact_y(x)

    plt.figure()
    plt.plot(x, yex, label="exact")
    plt.plot(x, y, "o--", label="numerical (Pade)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bvp_N24.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(x, np.abs(y - yex), "o-")
    plt.xlabel("x")
    plt.ylabel("|error|")
    plt.tight_layout()
    plt.savefig("bvp_error_N24.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()