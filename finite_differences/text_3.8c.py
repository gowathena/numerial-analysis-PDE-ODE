import numpy as np

def f(x):
    return 100/(np.sqrt(x)+0.01) + 1/((x-0.3)**2+0.001) - np.pi

def S(a,b,fa,fm,fb):
    return (b-a)*(fa + 4*fm + fb)/6

def adaptive_simpson(a=0.0, b=1.0, tol=1e-8, max_depth=30):
    xs = []

    def rec(a,b,fa,fm,fb,Sab,tol,depth):
        m = 0.5*(a+b)
        l = 0.5*(a+m)
        r = 0.5*(m+b)

        fl, fr = f(l), f(r)
        xs.extend([l, r])

        Sleft  = S(a,m,fa,fl,fm)
        Sright = S(m,b,fm,fr,fb)

        if depth >= max_depth or abs(Sleft + Sright - Sab) < 15*tol:
            return Sleft + Sright + (Sleft + Sright - Sab)/15.0

        return rec(a,m,fa,fl,fm,Sleft,tol/2,depth+1) + rec(m,b,fm,fr,fb,Sright,tol/2,depth+1)

    fa, fb = f(a), f(b)
    m = 0.5*(a+b)
    fm = f(m)
    xs.extend([a, m, b])

    I0 = S(a,b,fa,fm,fb)
    I = rec(a,b,fa,fm,fb,I0,tol,0)

    return I, np.unique(xs)

'''
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 100/(np.sqrt(x)+0.01) + 1/((x-0.3)**2+0.001) - np.pi

def S(a,b,fa,fm,fb):
    return (b-a)*(fa+4*fm+fb)/6

def adaptive(a,b,tol):
    xs = []
    def rec(a,b,fa,fm,fb,Sab,tol):
        c = (a+b)/2
        d = (a+c)/2
        e = (c+b)/2
        fd,fe = f(d),f(e)
        xs.extend([d,e])
        Sleft  = S(a,c,fa,fd,fm)
        Sright = S(c,b,fm,fe,fb)
        if abs(Sleft+Sright-Sab) < 15*tol:
            return Sleft+Sright
        return rec(a,c,fa,fd,fm,Sleft,tol/2) + rec(c,b,fm,fe,fb,Sright,tol/2)

    fa, fb = f(a), f(b)
    m = (a+b)/2
    fm = f(m)
    xs.extend([a,m,b])
    I = rec(a,b,fa,fm,fb,S(a,b,fa,fm,fb),tol)
    return I, np.unique(xs)

I, xs = adaptive(0,1,1e-8)
print("Adaptive I =", I)

x_plot = np.linspace(0,1,2000)
plt.plot(x_plot, f(x_plot))
plt.plot(xs, f(xs), 'o', markersize=3)
plt.grid()
plt.show()

'''
