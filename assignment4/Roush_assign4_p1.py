# -*- coding: utf-8 -*-
"""
This is for AE6310 Assignment #4, Problem 2
Created on Fri Apr 17 11:18:38 2020

@author: kroush7
"""

import numpy as np
import matplotlib.pylab as plt
import time
import math


def box_func(x):
    '''This is the black box function'''
    #col1= x1, col2=x2
    return x[0]**2 -x[1]**2 -math.cos(0.5*math.pi*x[0])*math.cos(0.5*math.pi*x[1])
  
def create_DoE_points(M):
    '''create the sample points'''
    DoE_points= [] #num points=M**2; col1= x1, col2=x2
    j=0
    while j<M:
        for i in range(M):
            x1= -1 + ((2*i)/(M-1))
            x2= -1 + ((2*j)/(M-1))
            DoE_points.append([x1,x2])
            # print('[%d, %d]'%(x1,x2))
            # print('i= %d, j= %d'%(i,j))
        j+=1
    DoE_points= np.array(DoE_points)
    return DoE_points

def phi_quadratic(xi):
    '''defines the basis functions'''
    #xi are the sample points
    #xi[0]= x1, xi[1]= x2
    phi0= 1
    phi1= xi[0]
    phi2= xi[1]
    phi3= xi[0]**2
    phi4= xi[0]*xi[1]
    phi5= xi[1]**2
    quad_basis= [phi0,phi1,phi2,phi3,phi4,phi5]
    return quad_basis

def surrogate_quad(xi, func):
    '''constructs the surrogate function from the function value at sample points
    and the basis function values at those points'''
    #xi are the sample points
    #xi[0]= x1, xi[1]= x2
    N= xi.shape[0] #number of sample points
    
    f = np.zeros(N) #black box function evaluated at sample points
    Phi = np.zeros((N, 6)) #6 quad basis funs w/ N rows (evaluated at N pts)

    for i in range(N): #calc values of black box func @sample pts
        f[i] = func(xi[i,:])
        for j in range(6): #6 quad basis
            # Evaluate the j-th basis function at the point xi[i,:]
            #technically calcs values of all basis funcs @ each pt
            basis_evals= phi_quadratic(xi[i,:]) 
            Phi[i,j] = basis_evals[j]
            
    #now we want to solve the unconstrained opt problem
    weights= np.linalg.solve(np.dot(Phi.T, Phi), np.dot(Phi.T, f))
    return weights

def eval_surrogate_quad(x, w):
    '''evualates the surrogate function at a desired point'''
    unweight_vals= phi_quadratic(x)
    fhat= np.dot(w,unweight_vals)
    
    return fhat
    
def phi_radialBasis(r):
    '''returns the radial basis function value, given r'''
    #e^(-r**2/(2*r0**2)); r0=1
    return math.exp(-0.5*(r**2))

def surrogate_radial(xi, func):
    '''creates radial basis surrogate function given the sample points and
    black box function'''
    #adapted from provided radial_basis_function.py
    N = xi.shape[0] #number of points

    f = np.zeros(N) #true function values at sample points
    Phi = np.zeros((N, N)) #basis function values at sample points

    # Set the values into f and Phi
    for i in range(N):
        f[i] = func(xi[i,:])

        # Place the basis function values in row i of
        # the Phi matrix
        for j in range(N):
            # Evaluate the j-th basis function at the point xi[i,:]
            r = np.sqrt(np.dot(xi[i,:] - xi[j,:], xi[i,:] - xi[j,:]))
            Phi[i,j] = phi_radialBasis(r)
    # Solve Phi^{T}*Phi*w = Phi^{T}*f, but because Phi is
    # square, we can solve Phi*w = f instead
    weights= np.linalg.solve(Phi, f)
    return weights
 
def eval_surrogate_radial(x, xi, w):
    """
    Evaluate the surrogate model at the specified design point.

    Args:
        x: The design point at which to evaluate the surrogate
        xi: The sample points
        w: The surrogate model weights

    Returns:
        The radial basis surrogate function value
    """
    #adapted from provided radial_basis_function.py
    # m = N in this case, since we are using an interpolating model
    N = len(w)
    fhat = 0.0
    for i in range(N):
        # r = ||x - x[i]||_{2}
        r = np.sqrt(np.dot(x - xi[i,:], x - xi[i,:]))
        fhat += w[i]*phi_radialBasis(r)

    return fhat

def check100_pts(w,func):
    '''validates the model with 100 random points'''
    npts= 100
    N=100
    
    #adapted from provided radial_basis_function.py
    X = -1.0 + 2.0*np.random.uniform(size=(N, 1)) #between -1,1
    Y = -1.0 + 2.0*np.random.uniform(size=(N, 1))
    X, Y = np.meshgrid(X, Y)
    F = np.zeros((npts, npts))
    Fhat = np.zeros((npts, npts))
    
    for j in range(npts):
        for i in range(npts):
            xpt = np.array([X[i,j], Y[i,j]])
            F[i,j] = box_func(xpt)
            Fhat[i,j] = func(xpt, w)
    
    # Evaluate the R2 value (coefficient of determination)
    SSE = np.sum((F - Fhat)**2)
    SST = np.sum((F - np.average(F))**2)
    
    R2 = 1.0 - SSE/SST
    # R2= round(R2,5)
    print('R2, 100 random = ', R2)
        
def error_contours(xi,w,M,func):
    '''plots the contours of the function, surrogate model, and error'''
    #adapted from provided radial_basis_function.p
    npts = 250
    X = np.linspace(-1, 1, npts)
    X, Y = np.meshgrid(X, X)
    F = np.zeros((npts, npts))
    Fhat = np.zeros((npts, npts))
    
    for j in range(npts):
        for i in range(npts):
            xpt = np.array([X[i,j], Y[i,j]])
            F[i,j] = box_func(xpt)
            Fhat[i,j] = func(xpt, w)
    
    # Evaluate the R2 value (coefficient of determination)
    SSE = np.sum((F - Fhat)**2)
    SST = np.sum((F - np.average(F))**2)
    
    R2 = 1.0 - SSE/SST
    # R2= round(R2,5)
    print('R2, sample pts = ', R2)
    
    plt.figure()
    plt.contour(X, Y, F, levels=50)
    plt.title('True black-box function')
    
    plt.figure()
    plt.contour(X, Y, Fhat, levels=50)
    plt.plot(xi[:,0], xi[:,1], 'ob')
    plt.title('Surrogate function, M=%d'%(M))
    
    plt.figure()
    plt.contour(X, Y, F - Fhat, levels=50)
    plt.plot(xi[:,0], xi[:,1], 'ob')
    plt.title('Surrogate error, M=%d'%(M))
    
    plt.show()
    
def radial_graphCalc(xi,w,M):
    # Plot the true function and the black box function
    npts = 250
    X = np.linspace(-1, 1, npts)
    X, Y = np.meshgrid(X, X)
    F = np.zeros((npts, npts))
    Fhat = np.zeros((npts, npts))
    
    for j in range(npts):
        for i in range(npts):
            xpt = np.array([X[i,j], Y[i,j]])
            F[i,j] = box_func(xpt)
            Fhat[i,j] = eval_surrogate_radial(xpt, xi, w)
    
    print(F-Fhat)
    # Evaluate the R2 value (coefficient of determination)
    SSE = np.sum((F - Fhat)**2)
    SST = np.sum((F - np.average(F))**2)
    
    R2 = 1.0 - SSE/SST
    # R2= round(R2,5)
    print('R2 sample pts = ', R2)
    
    plt.figure()
    plt.contour(X, Y, F, levels=50)
    plt.title('True black-box function')
    
    plt.figure()
    plt.contour(X, Y, Fhat, levels=50)
    plt.plot(xi[:,0], xi[:,1], 'ob')
    plt.title('Surrogate function, M=%d'%(M))
    
    plt.figure()
    plt.contour(X, Y, F - Fhat, levels=50)
    plt.plot(xi[:,0], xi[:,1], 'ob')
    plt.title('Surrogate error, M=%d'%(M))

    plt.show()
    
def check100_pts_radial(w,func,xi):
    '''validates the model with 100 random points'''
    npts= 100
    N=100
    
    #adapted from provided radial_basis_function.py
    X = -1.0 + 2.0*np.random.uniform(size=(N, 1)) #between -1,1
    Y = -1.0 + 2.0*np.random.uniform(size=(N, 1))
    X, Y = np.meshgrid(X, Y)
    F = np.zeros((npts, npts))
    Fhat = np.zeros((npts, npts))
    
    for j in range(npts):
        for i in range(npts):
            xpt = np.array([X[i,j], Y[i,j]])
            F[i,j] = box_func(xpt)
            Fhat[i,j] = func(xpt, xi, w)
    
    # Evaluate the R2 value (coefficient of determination)
    SSE = np.sum((F - Fhat)**2)
    SST = np.sum((F - np.average(F))**2)
    
    R2 = 1.0 - SSE/SST
    # R2= round(R2,5)
    print('R2, 100 random = ', R2)
    
def quad_basis_answers(M):
    '''calculates answers to Problem 1, part 1'''
    xi= create_DoE_points(M)
    w= surrogate_quad(xi, box_func)
    print('M= %d weights: '%(M), w)

    error_contours(xi,w,M,eval_surrogate_quad)
    check100_pts(w,eval_surrogate_quad)
    
def radial_basis_answers(M):
    '''calculates answers to Problem 1, part 2'''
    xi= create_DoE_points(M)
    w= surrogate_radial(xi, box_func)
    print('M= %d weights: '%(M), w)

    radial_graphCalc(xi,w,M)
    check100_pts_radial(w,eval_surrogate_radial,xi)
#%%
start_time = time.time()


## Problem 1, Part A
quad_basis_answers(3) #M=3
print('\n')
quad_basis_answers(5) #M=5

## Problem 1, Part B
radial_basis_answers(3) #M=3
print('\n')
radial_basis_answers(5) #M=5


print("\n--- %s seconds ---" % (time.time() - start_time))  