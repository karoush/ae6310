# -*- coding: utf-8 -*-
"""
This file is all functions referenced in assign1
"""

import numpy as np
import matplotlib.pylab as plt

#from course provided notebook: "Minimization of Quadratic Functions"
def quadratic_decomposition(A, b, tol=1e-6):
    # Perform the eigenvalue decomposition
    lam, Q = np.linalg.eigh(A)
    
    # Compute bar{b} = Q^{T}*b
    bbar = np.dot(Q.T, b)
    
    if lam[0] > tol:
        print('Positive definite; Unique minimizer')
        xstar = -np.linalg.solve(A, b)
    elif lam[0] > -tol:
        # The problem is semi-definite 
        
        # Keep track of whether this problem is unbounded from below or not
        unbounded = False
        
        # Copy the values of A to a new matrix
        Abar = np.array(A, dtype=np.float)
        
        for index, eig in enumerate(lam):
            if np.fabs(eig) < tol:
                # Add the outer product of the two vectors to make the matrix
                # Abar non-singular
                Abar += np.outer(Q[:, index], Q[:, index])
                
                # Keep track of whether bbar is zero or not
                if np.fabs(bbar[index]) > tol:
                    unbounded = True
                    
        if not unbounded:
            print('Semi-definite; Infinite number of minimizers')
            xstar = -np.linalg.solve(Abar, b)
        else:
            print('Semi-definite; Quadratic unbounded')
            xstar = None
    else:
        # The smallest eigenvalue is negative
        xstar = None
        
    if xstar is not None:
        print('xstar = ', xstar)

    if b.shape[0] == 2:
        m = 50
        x1 = np.linspace(-2, 2, m)
        x2 = np.linspace(-2, 2, m)
        X1, X2 = np.meshgrid(x1, x2)

        Fx = np.zeros(X1.shape)
        Fxi = np.zeros(X1.shape)

        for j in range(m):
            for i in range(m):
                x = np.array([X1[i,j], X2[i,j]])
                Fx[i, j] = 0.5*np.dot(x, np.dot(A, x)) + np.dot(b, x)
                Fxi[i, j] = 0.5*np.sum(lam*x**2) + np.dot(bbar, x)

        fig, ax = plt.subplots(1, 2)
        ax[0].contour(X1, X2, Fxi)
        ax[1].contour(X1, X2, Fx)
        if xstar is not None:
            ax[1].plot([xstar[0]], [xstar[1]], 'bo')
        ax[0].set_aspect('equal', 'box')
        ax[0].set_title('Eig. space')

        ax[1].set_aspect('equal', 'box')
        ax[1].set_title('Design space')
        fig.tight_layout()
        plt.show()
        
def approx_gradient(x, fobj, h=1e-6):
    '''Approximate the gradient using central difference'''
    x = np.array(x)
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    g = np.array([0.5*(fobj(x + h*e1) - fobj(x - h*e1))/h,
                  0.5*(fobj(x + h*e2) - fobj(x - h*e2))/h])
    return g 

def approx_hessian(x, fobj, h=1e-6):
    '''Approximate the Hessian'''
    x = np.array(x)
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    H = np.array([[(fobj(x + h*e1) - 2*fobj(x) + fobj(x - h*e1))/h**2,
                   0.25*(fobj(x + h*(e1 + e2)) -
                         fobj(x + h*(e1 - e2)) -
                         fobj(x + h*(e2 - e1)) + 
                         fobj(x + h*(e1 + e2)))/h**2],
                  [0, (fobj(x + h*e2) - 2*fobj(x) + fobj(x - h*e2))/h**2]])
    H[1,0] = H[0,1]
    return H

def cplot(fobj,line):
    n = 50
    x1 = np.linspace(-2, 2, n)
    x2 = np.linspace(-2, 2, n)
    X1, X2 = np.meshgrid(x1, x2)
    f = np.zeros((n, n))

    if line:
        for j in range(n):
            for i in range(n):
                x = np.array([X1[i,j], X2[i,j]])
                f[i, j] = fobj([X1[i, j], X2[i, j]])
        fig, ax = plt.subplots(1, 1)
        ax.contour(X1, X2, f)

    else:
    # Query the function at the specified locations
        for i in range(n):
            for j in range(n):
                f[i, j] = fobj([X1[i, j], X2[i, j]])
    
        fig, ax = plt.subplots(1, 1)
        ax.contourf(X1, X2, f)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.xlabel('x1')
    plt.ylabel('x2')    
    plt.title(str(fobj)[1:13])