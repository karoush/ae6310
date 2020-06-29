import numpy as np
import matplotlib.pylab as plt

def rosen(x):
    """
    The black-box function: The Rosenbrock function

    Args:
        x: Array of the design point

    Returns:
        The Rosenbrock function value
    """
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def phi(r):
    """
    Evaluate the radial basis function phi(r) given the
    distance (radius) between the evaluation point and the
    basis point.

    Args:
        r: The distance (radius) to the evaluation point

    Returns:
        The radial basis function value
    """
    return np.sqrt(1.0 + r**2)

def construct_surrogate(xi, func):
    """
    Construct the radial basis surrogate model with the given
    set of sample points and the given function pointer.

    Args:
        xi: Two-dimensional array of the sample points
        func: Function pointer to the black-box function

    Returns:
        The surrogate model weights
    """
    N = xi.shape[0]

    f = np.zeros(N)
    Phi = np.zeros((N, N))

    # Set the values into f and Phi
    for i in range(N):
        f[i] = func(xi[i,:])

        # Place the basis function values in row i of
        # the Phi matrix
        for j in range(N):
            # Evaluate the j-th basis function at the point xi[i,:]
            r = np.sqrt(np.dot(xi[i,:] - xi[j,:], xi[i,:] - xi[j,:]))
            Phi[i,j] = phi(r)

    # Solve Phi^{T}*Phi*w = Phi^{T}*f, but because Phi is
    # square, we can solve Phi*w = f instead

    return np.linalg.solve(Phi, f)

def eval_surroage(x, xi, w):
    """
    Evaluate the surrogate model at the specified design point.

    Args:
        x: The design point at which to evaluate the surrogate
        xi: The sample points
        w: The surrogate model weights

    Returns:
        The radial basis surrogate function value
    """
    # m = N in this case, since we are using an interpolating model
    N = len(w)

    fhat = 0.0
    for i in range(N):
        # r = ||x - x[i]||_{2}
        r = np.sqrt(np.dot(x - xi[i,:], x - xi[i,:]))
        fhat += w[i]*phi(r)

    return fhat

# Evaluate the sample points
N = 50

# Generate random sample points between [-1, 1]^{2}
xi = -1.0 + 2.0*np.random.uniform(size=(N, 2))

# Find the weights and
w = construct_surrogate(xi, rosen)

# Plot the true function and the black box function
npts = 250
X = np.linspace(-1, 1, npts)
X, Y = np.meshgrid(X, X)
F = np.zeros((npts, npts))
Fhat = np.zeros((npts, npts))

for j in range(npts):
    for i in range(npts):
        xpt = np.array([X[i,j], Y[i,j]])
        F[i,j] = rosen(xpt)
        Fhat[i,j] = eval_surroage(xpt, xi, w)

# Evaluate the R2 value (coefficient of determination)
SSE = np.sum((F - Fhat)**2)
SST = np.sum((F - np.average(F))**2)

R2 = 1.0 - SSE/SST
print('R2 = ', R2)

plt.figure()
plt.contour(X, Y, F, levels=50)
plt.title('True black-box function')

plt.figure()
plt.contour(X, Y, Fhat, levels=50)
plt.plot(xi[:,0], xi[:,1], 'ob')
plt.title('Surrogate function')

plt.figure()
plt.contour(X, Y, F - Fhat, levels=50)
plt.plot(xi[:,0], xi[:,1], 'ob')
plt.title('Surrogate error')

plt.show()
