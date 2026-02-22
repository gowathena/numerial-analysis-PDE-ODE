import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(2*x) * np.cos(20*x) + np.exp(np.sin(2*x))

def fp(x):
    return (2*np.cos(2*x)*np.cos(20*x)
            - 20*np.sin(2*x)*np.sin(20*x)
            + 2*np.cos(2*x)*np.exp(np.sin(2*x)))

def grid(N):
    L = 2*np.pi
    dx = L / N
    x = np.arange(N) * dx
    return x, dx

def d_fwd1(F, dx):
    return (np.roll(F, -1) - F) / dx

def d_c2(F, dx):
    return (np.roll(F, -1) - np.roll(F, 1)) / (2*dx)

def d_c4(F, dx):
    return (-np.roll(F, -2) + 8*np.roll(F, -1) - 8*np.roll(F, 1) + np.roll(F, 2)) / (12*dx)

def thomas(a, b, c, d):
    n = len(b)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / den if i < n-1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i-1]) / den
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

def d_pade4_periodic(F, dx):
    N = len(F)
    a = 1/4
    A = np.zeros((N, N))
    np.fill_diagonal(A, 1.0)
    for i in range(N):
        A[i, (i-1) % N] = a
        A[i, (i+1) % N] = a
    rhs = (3/(4*dx)) * (np.roll(F, -1) - np.roll(F, 1))
    return np.linalg.solve(A, rhs)

def d_pade4_onesidedBC(F, dx):
    N = len(F)
    a = 1/4
    D = np.zeros(N)

    D[0] = (-11*F[0] + 18*F[1] - 9*F[2] + 2*F[3]) / (6*dx)
    D[-1] = (11*F[-1] - 18*F[-2] + 9*F[-3] - 2*F[-4]) / (6*dx)

    n = N - 2
    sub = np.zeros(n)
    diag = np.ones(n)
    sup = np.zeros(n)
    rhs = np.zeros(n)

    for k in range(n):
        i = k + 1
        sub[k] = a if k > 0 else 0.0
        sup[k] = a if k < n-1 else 0.0
        rhs[k] = (3/(4*dx)) * (F[i+1] - F[i-1])

    rhs[0] -= a * D[0]
    rhs[-1] -= a * D[-1]
    D[1:-1] = thomas(sub, diag, sup, rhs)
    return D

def linf(err):
    return np.max(np.abs(err))

def main():
    Ns = [256, 512, 1024, 2048]
    dxs = []
    e1 = []; e2 = []; e4 = []
    ep_per = []; ep_os = []

    for N in Ns:
        x, dx = grid(N)
        F = f(x)
        exact = fp(x)

        dxs.append(dx)
        e1.append(linf(d_fwd1(F, dx) - exact))
        e2.append(linf(d_c2(F, dx) - exact))
        e4.append(linf(d_c4(F, dx) - exact))
        ep_per.append(linf(d_pade4_periodic(F, dx) - exact))
        ep_os.append(linf(d_pade4_onesidedBC(F, dx) - exact))

    dxs = np.array(dxs)
    e1 = np.array(e1); e2 = np.array(e2); e4 = np.array(e4)
    ep_per = np.array(ep_per); ep_os = np.array(ep_os)

    Nshow = 256
    x, dx = grid(Nshow)
    F = f(x); exact = fp(x)

    plt.figure()
    plt.plot(x, exact, label="exact")
    plt.plot(x, d_fwd1(F, dx), "--", label="fwd1")
    plt.plot(x, d_c2(F, dx), "--", label="c2")
    plt.plot(x, d_c4(F, dx), "--", label="c4")
    plt.plot(x, d_pade4_periodic(F, dx), "--", label="pade4 periodic")
    plt.plot(x, d_pade4_onesidedBC(F, dx), "--", label="pade4 one-sided")
    plt.legend()
    plt.xlabel("x"); plt.ylabel("f'(x)")
    plt.tight_layout()

    plt.figure()
    plt.loglog(dxs, e1, "o-", label="fwd1")
    plt.loglog(dxs, e2, "o-", label="c2")
    plt.loglog(dxs, e4, "o-", label="c4")
    plt.loglog(dxs, ep_os, "o-", label="pade4 + one-sided")
    plt.loglog(dxs, ep_per, "o-", label="pade4 periodic")

    C1 = e1[0]/dxs[0]
    C2 = e2[0]/dxs[0]**2
    C3 = e2[0]/dxs[0]**3
    C4 = e4[0]/dxs[0]**4
    plt.loglog(dxs, C1*dxs, "--", label="~dx")
    plt.loglog(dxs, C2*dxs**2, "--", label="~dx^2")
    plt.loglog(dxs, C3*dxs**3, "--", label="~dx^3")
    plt.loglog(dxs, C4*dxs**4, "--", label="~dx^4")

    plt.gca().invert_xaxis()
    plt.xlabel("dx"); plt.ylabel("L_inf error")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()