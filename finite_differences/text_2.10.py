import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 - x**8

def fp_exact(x):
    return -8*x**7

def d1_nonuniform(x, F):
    N = len(x) - 1
    d = np.zeros_like(F)

    d[1:-1] = (F[2:] - F[:-2]) / (x[2:] - x[:-2])

    h0 = x[1] - x[0]
    h1 = x[2] - x[1]
    d[0] = (-(2*h0+h1)/(h0*(h0+h1))*F[0]
            + (h0+h1)/(h0*h1)*F[1]
            - h0/(h1*(h0+h1))*F[2])

    hN1 = x[N] - x[N-1]
    hN2 = x[N-1] - x[N-2]
    d[N] = ((2*hN1+hN2)/(hN1*(hN1+hN2))*F[N]
            - (hN1+hN2)/(hN1*hN2)*F[N-1]
            + hN1/(hN2*(hN1+hN2))*F[N-2])

    return d

def d1_transform_tanh(N, a):
    xi = -1 + 2*np.arange(N+1)/N
    s = xi*np.arctanh(a)
    x = (1/a)*np.tanh(s)

    F = f(x)
    dxi = 2/N

    dF_dxi = np.zeros_like(F)
    dF_dxi[1:-1] = (F[2:] - F[:-2])/(2*dxi)
    dF_dxi[0] = (-3*F[0] + 4*F[1] - F[2])/(2*dxi)
    dF_dxi[-1] = (3*F[-1] - 4*F[-2] + F[-3])/(2*dxi)

    dx_dxi = (np.arctanh(a)/a) * (1/np.cosh(s)**2)
    fp = dF_dxi / dx_dxi

    return x, fp

def d1_transform_cos(N):
    xi = np.pi*np.arange(N+1)/N
    x = np.cos(xi)
    F = f(x)

    dxi = np.pi/N
    dF_dxi = np.zeros_like(F)
    dF_dxi[1:-1] = (F[2:] - F[:-2])/(2*dxi)
    dF_dxi[0] = (-3*F[0] + 4*F[1] - F[2])/(2*dxi)
    dF_dxi[-1] = (3*F[-1] - 4*F[-2] + F[-3])/(2*dxi)

    dx_dxi = -np.sin(xi)
    fp = np.zeros_like(F)
    fp[1:-1] = dF_dxi[1:-1] / dx_dxi[1:-1]
    fp[0] = fp_exact(x[0])
    fp[-1] = fp_exact(x[-1])

    return x, fp

def uniform_search(E_target):
    M = 16
    while True:
        x = -1 + 2*np.arange(M+1)/M
        dx = 2/M
        F = f(x)

        fp = np.zeros_like(F)
        fp[1:-1] = (F[2:] - F[:-2])/(2*dx)
        fp[0] = (-3*F[0] + 4*F[1] - F[2])/(2*dx)
        fp[-1] = (3*F[-1] - 4*F[-2] + F[-3])/(2*dx)

        E = np.max(np.abs(fp - fp_exact(x)))
        if E <= E_target:
            return M, E
        M *= 2

def main():
    N = 32

    for a in [0.98, 0.9]:
        xi = -1 + 2*np.arange(N+1)/N
        x = (1/a)*np.tanh(xi*np.arctanh(a))
        F = f(x)

        fp1 = d1_nonuniform(x, F)
        x2, fp2 = d1_transform_tanh(N, a)

        plt.figure()
        plt.plot(x, fp_exact(x), label="exact")
        plt.plot(x, fp1, "o--", label="Eq(2.20)")
        plt.plot(x2, fp2, "s--", label="transform")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("f'(x)")
        plt.tight_layout()
        plt.savefig(f"p210a_a{a}.png", dpi=300)
        plt.show()

    x3, fp3 = d1_transform_cos(N)
    plt.figure()
    plt.plot(x3, fp_exact(x3), label="exact")
    plt.plot(x3, fp3, "o--", label="cos transform")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f'(x)")
    plt.tight_layout()
    plt.savefig("p210b_cos.png", dpi=300)
    plt.show()

    x98, fp98 = d1_transform_tanh(N, 0.98)
    E_trans = np.max(np.abs(fp98 - fp_exact(x98)))
    Mmin, Euni = uniform_search(E_trans)

    print("E_trans =", E_trans)
    print("Smallest uniform M =", Mmin)
    print("Uniform error =", Euni)

if __name__ == "__main__":
    main()