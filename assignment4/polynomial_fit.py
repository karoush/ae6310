import numpy as np
import matplotlib.pylab as plt

def f1(x):
    return np.exp(5*x/4) + 2*np.cos(3*np.pi*x)

def eval_poly(w, x):
    fhat = 0.0
    for wi in w[::-1]:
        fhat = wi + x*fhat
    return fhat

# From the given sample set, construct the model of the function
def construct_poly_surrogate(m, xi, func):
    N = len(xi)
    f = np.zeros(N)

    # Form the matrix phi
    Phi = np.zeros((N, m))
    Phi[:,0] = 1.0

    # Evaluate the function
    for i, x in enumerate(xi):
        f[i] = func(x)

        for j in range(1,m):
            Phi[i,j] = x**j

    return np.linalg.solve(np.dot(Phi.T, Phi), np.dot(Phi.T, f))

xlow = 0
xhigh = 2

# Set the samples
x = np.linspace(xlow, xhigh, 100)

for m in range(4,12,2):
    print('m = ', m)
    for N in [2, 4, 5, 6, 10, 15, 20]:
        if N < m:
            continue

        # xi = np.linspace(xlow, xhigh, N)
        y = 0.5*(1.0 - np.cos(np.linspace(0, np.pi, N)))
        xi = xlow + (xhigh - xlow)*y

        w = construct_poly_surrogate(m, xi, f1)

        fhat = np.zeros(x.shape)
        for i, xval in enumerate(x):
            fhat[i] = eval_poly(w, xval)

        plt.figure()
        plt.title('m = %d N = %d'%(m, N))
        plt.plot(x, f1(x), label='exact')
        plt.plot(xi, f1(xi), 'ko', label='samples')
        plt.plot(x, fhat, label='surroage')
        plt.legend()
        plt.savefig('polynomial_fit_example_m=%d_N=%d.pdf'%(m, N))