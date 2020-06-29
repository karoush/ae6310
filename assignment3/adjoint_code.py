import numpy as np

def evaluate_governing_eqns(x, E=1.0, L=1.0):
    """
    This function provides the K(x) matrix and F vector that are
    needed to solve the governing equations.

    Args:
        x (np.ndarray) Vector of length 3. The design variables

    Return:
        The matrix K(x) and the vector F
    """

    K = np.zeros((2, 2), dtype=x.dtype)
    F = np.zeros(2, dtype=x.dtype)

    s = E/(2.0*np.sqrt(2)*L)
    K[0,0] = s*(2.0*np.sqrt(2)*x[0] + x[1])
    K[0,1] = s*x[1]
    K[1,0] = s*x[1]
    K[1,1] = s*(x[1] + 2.0*np.sqrt(2)*x[2])

    F[0] = 1.0
    F[1] = 1.0

    return K, F

def evaluate(x, E=1.0, L=1.0):
    """
    Evaluate the function of interest: The sum square of the state
    variables or f = dot(u, u)

    Args:
        x (np.ndarray) Vector of length 3. The design variables
        E (float) The elastic modulus
        L (float) The length of the bar

    Return:
        The value of the function of interest
    """

    # Solve the governing equations to obtain the state variables
    # R(x,u) = K(x)*u - F = 0
    K, F = evaluate_governing_eqns(x, E=E, L=L)

    # Solve the governing equations to obtain u
    u = np.linalg.solve(K, F)

    # Evaluate the function of interest
    f = u[0]**2 + u[1]**2

    return f

def adjoint_total_derivative(x, E=1.0, L=1.0):
    """
    Use the adjoint method to evaluate the derivative of the function
    of interest with respect to the design variables.

    Args:
        x (np.ndarray) Vector of length 3. The design variables
        E (float) The elastic modulus
        L (float) The length of the bar

    Return:
        dfdx (np.ndarray) Vector of length 3. The total derivative
    """

    # Solve the governing equations to obtain the state variables
    # R(x,u) = K(x)*u - F = 0
    K, F = evaluate_governing_eqns(x, E=E, L=L)

    # Solve the governing equations to obtain u
    u = np.linalg.solve(K, F)

    # dR/du = K(x)
    dfdu = np.zeros((2), dtype=x.dtype)
    dfdu[0] = 2.0*u[0]
    dfdu[1] = 2.0*u[1]

    # Solve for the adjoint variables
    psi = -np.linalg.solve(K.T, dfdu)

    # Multiply the adjoint variables by dR/dx
    dRdx = np.zeros((2, 3), dtype=x.dtype)

    s = E/(2.0*np.sqrt(2)*L)
    dRdx[0,0] = s*2.0*np.sqrt(2.0)*u[0]
    dRdx[1,0] = 0.0

    dRdx[0,1] = s*(u[0] + u[1])
    dRdx[1,1] = s*(u[0] + u[1])

    dRdx[0,2] = 0.0
    dRdx[1,2] = s*2.0*np.sqrt(2.0)*u[1]

    return np.dot(psi.T, dRdx)

def direct_total_derivative(x, E=1.0, L=1.0):
    """
    Use the direct method to evaluate the derivative of the function
    of interest with respect to the design variables.

    Args:
        x (np.ndarray) Vector of length 3. The design variables
        E (float) The elastic modulus
        L (float) The length of the bar

    Return:
        dfdx (np.ndarray) Vector of length 3. The total derivative
    """

    # Solve the governing equations to obtain the state variables
    # R(x,u) = K(x)*u - F = 0
    K, F = evaluate_governing_eqns(x, E=E, L=L)

    # Solve the governing equations to obtain u
    u = np.linalg.solve(K, F)

    # dR/du*phi = - dR/dx ===> K*phi = - dR/dx
    # Compute dR/dx
    dRdx = np.zeros((2, 3), dtype=x.dtype)
    s = E/(2.0*np.sqrt(2)*L)
    dRdx[0,0] = s*2.0*np.sqrt(2.0)*u[0]
    dRdx[1,0] = 0.0

    dRdx[0,1] = s*(u[0] + u[1])
    dRdx[1,1] = s*(u[0] + u[1])

    dRdx[0,2] = 0.0
    dRdx[1,2] = s*2.0*np.sqrt(2.0)*u[1]

    phi = -np.linalg.solve(K, dRdx)

    # Compute df/du
    dfdu = np.zeros((2), dtype=x.dtype)
    dfdu[0] = 2.0*u[0]
    dfdu[1] = 2.0*u[1]

    return np.dot(dfdu, phi)

# Set the perturbation vector. We perturb the design variables
# along this vector in the design space
# pert = np.array([1.0, -1.0, 0.5])
pert = np.random.uniform(size=3)

# Compute the function of interest at the point x0
x0 = np.array([1.0, 2.0, 3.0])
f0 = evaluate(x0)
print('Design x0 point:', x0)
print('Function value: ', f0)

# Evaluate the total derivative using the adjoint method
dfdx = adjoint_total_derivative(x0)
adj = np.dot(pert, dfdx)
print('Adjoint-based derivative:        ', adj)

# Evaluate the total derivative using the direct method
dfdx = direct_total_derivative(x0)
direct = np.dot(pert, dfdx)
print('Direct-based derivative:         ', direct)

# Approximate the derivative using the forward difference method
h = 1e-6
x1 = x0 + h*pert
f1 = evaluate(x1)
fd = (f1 - f0)/h
print('Finite-difference approximation: ', fd)

# Approximate the derivative using the complex-step method
h = 1e-30
x2 = x0 + h*1j*pert
f2 = evaluate(x2)
cs = f2.imag/h
print('Complex-step approximation:      ', cs)

# Compare the derivatives between adjoint, direct, complex-step and
# forward difference.
print('Relative error between CS and Adjoint: ', (adj - cs)/cs)
print('Relative error between CS and Direct:  ', (direct - cs)/cs)
print('Relative error between CS and FD:      ', (fd - cs)/cs)
print('Relative error between Adjoint and FD: ', (fd - adj)/adj)
