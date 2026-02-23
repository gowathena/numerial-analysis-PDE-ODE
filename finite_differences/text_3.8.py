import numpy as np
import matplotlib.pyplot as plt

f = lambda x: 100/(np.sqrt(x)+0.01) + 1/((x-0.3)**2+0.001) - np.pi

def trap(x, fx):
    h = x[1]-x[0]
    return h*(0.5*fx[0] + np.sum(fx[1:-1]) + 0.5*fx[-1])

def simpson(x, fx):
    n = len(x)-1
    h = x[1]-x[0]
    w = np.ones(n+1)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    return (h/3)*np.dot(w, fx)

def end_corrected(x, fx):
    h = x[1]-x[0]
    IT = trap(x, fx)
    f0,f1,f2,f3,f4 = fx[:5]
    fn,fn1,fn2,fn3,fn4 = fx[-1],fx[-2],fx[-3],fx[-4],fx[-5]
    fp0 = (-25*f0+48*f1-36*f2+16*f3-3*f4)/(12*h)
    fp1 = (25*fn-48*fn1+36*fn2-16*fn3+3*fn4)/(12*h)
    return IT - (h**2/12)*(fp1-fp0)

# reference (fine Simpson)
n_ref = 2**16
x_ref = np.linspace(0,1,n_ref+1)
I_ref = simpson(x_ref, f(x_ref))

ns = [2**k for k in range(3,11)]
errT, errS, errTc = [], [], []

for n in ns:
    x = np.linspace(0,1,n+1)
    fx = f(x)
    errT.append(abs(trap(x,fx)-I_ref)/abs(I_ref))
    errS.append(abs(simpson(x,fx)-I_ref)/abs(I_ref))
    errTc.append(abs(end_corrected(x,fx)-I_ref)/abs(I_ref))

plt.loglog(ns, errT, 'o-', label="Trap")
plt.loglog(ns, errS, 's-', label="Simpson")
plt.loglog(ns, errTc, 'd-', label="Trap+end")
plt.legend()
plt.grid(True)
plt.show()