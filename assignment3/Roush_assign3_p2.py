# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:24:29 2020

@author: kroush7
"""

import numpy as np
import time
import matplotlib.pylab as plt

def evaluate(x):
    """
    Evaluate the function of interest
    Args: x (np.ndarray) Vector of length 3. The design variables
    Return: The value of the function of interest
    """
    # Fill in the values of the governing equation
    # R(x,u) = K(x)*u - F = 0
    K, F = evaluate_governing_eqns(x)
    # Solve the governing equations to obtain u
    u = np.linalg.solve(K, F)
    
    # Evaluate the function of interest
    f = x[0]*x[1] +np.sqrt(u[0]*u[1])
    return f

def evaluate_governing_eqns(x):
    #0= K(x)*u -F
    K = np.zeros((2, 2), dtype=x.dtype)
    F = np.zeros(2, dtype=x.dtype)

    #x= [x1, x2, x3]
    K[0,0] = x[0] +x[1] #x1+x2
    K[0,1] = x[2] -x[1] #x3-x2
    K[1,0] = x[2] -x[1] #x3-x2
    K[1,1] = x[0] +x[1] -x[2] #x1+x2-x3

    F[0] = 1.0
    F[1] = 1.0
    return K, F

def adjoint_total_derivative(x):
    """ Use the adjoint method to evaluate the derivative of the function
    of interest with respect to the design variables.
    Args: x (np.ndarray) Vector of length 3. The design variables
    Return: dfdx (np.ndarray) Vector of length 3. The total derivative
    """
    # Fill in the values of the governing equation
    # R(x,u) = K(x)*u - F = 0
    K, F = evaluate_governing_eqns(x)
    # Solve the governing equations to obtain u
    u = np.linalg.solve(K, F)

    # define dR/du = K
    # define df/du 
    dfdu = np.zeros((2), dtype=x.dtype)
    dfdu[0] = u[1]/(2*np.sqrt(u[0]*u[1]))
    dfdu[1] = u[0]/(2*np.sqrt(u[0]*u[1]))
    # Solve for the adjoint variables
    psi = -np.linalg.solve(K.T, dfdu.T)

    # Define dR/dx
    dRdx = np.zeros((2, 3), dtype=x.dtype)
    dRdx[0,0] = u[0] #first column
    dRdx[1,0] = u[1]
    dRdx[0,1] = u[0]- u[1] #column 2
    dRdx[1,1] = u[1]- u[0] 
    dRdx[0,2] = u[1] #column 3
    dRdx[1,2] = u[0]- u[1]

    #define dF/dx
    dFdx = np.zeros((1, 3), dtype=x.dtype)
    dFdx[0,0] = x[1]
    dFdx[0,1] = x[0]
    dFdx[0,2] = 0.0

    return dFdx+ np.dot(psi.T, dRdx)

def ptA_compare(h):
    # Set the perturbation vector. We perturb the design variables
    pert = np.array([1,1,1])
    
    # Compute the function of interest at the point x0
    x0 = np.array([1.1, 1.2, 1.3])
    f0 = evaluate(x0)
    print('Design x0 point:', x0)
    print('Function value: ', f0)
    
    # forward difference approximation
    print ('\nStep size: ',h)
    pert_x1= np.array([1,0,0])
    pert_x2= np.array([0,1,0])
    pert_x3= np.array([0,0,1])
    
    x1_d= x0+ h*pert_x1
    f1_d= evaluate(x1_d)
    f1d = (f1_d - f0)/h
    
    x2_d= x0+ h*pert_x2
    f2_d= evaluate(x2_d)
    f2d = (f2_d - f0)/h
    
    x3_d= x0+ h*pert_x3
    f3_d= evaluate(x3_d)
    f3d = (f3_d - f0)/h
    fd= np.array([f1d, f2d, f3d])
    print('Forward-difference approximation: ', fd)
    fd= np.linalg.norm(fd)
    
    # complex step approximation
    x1_c = x0 + h*1j*pert_x1
    f1_c = evaluate(x1_c)
    cs_1 = f1_c.imag/h
    
    x2_c = x0 + h*1j*pert_x2
    f2_c = evaluate(x2_c)
    cs_2 = f2_c.imag/h
    
    x3_c = x0 + h*1j*pert_x3
    f3_c = evaluate(x3_c)
    cs_3 = f3_c.imag/h
    
    cs= np.array([cs_1,cs_2,cs_3])
    print('Complex-step approximation:       ', cs) 
    cs= np.linalg.norm(cs)
    
    total_der = adjoint_total_derivative(x0)[0]
    print('Adjoint based total derivative:   ', total_der)
    total_der= np.linalg.norm(total_der)
    
    error_fd= abs(fd- total_der)/abs(total_der)
    error_cs= abs(cs- total_der)/abs(total_der)
    print('Magnitude of relative FD error:   ', error_fd)
    print('Magnitude of relative CS error:   ', error_cs,'\n')

def ptA_graph():
    pert = np.array([1,1,1])
    x0 = np.array([1.1, 1.2, 1.3])
    f0 = evaluate(x0)
    n= 1000
    exp= np.linspace(1,30,n)
    h_list= 1*10**(-exp)
    fd_error=[]
    cs_error= []

    pert_x1= np.array([1,0,0])
    pert_x2= np.array([0,1,0])
    pert_x3= np.array([0,0,1])
    
    for h in h_list:
        #forward step approx
        x1_d= x0+ h*pert_x1
        f1_d= evaluate(x1_d)
        f1d = (f1_d - f0)/h
        x2_d= x0+ h*pert_x2
        f2_d= evaluate(x2_d)
        f2d = (f2_d - f0)/h
        x3_d= x0+ h*pert_x3
        f3_d= evaluate(x3_d)
        f3d = (f3_d - f0)/h
        
        fd= np.array([f1d, f2d, f3d])
        fd= np.linalg.norm(fd)
    
        #complex step approximation
        x1_c = x0 + h*1j*pert_x1
        f1_c = evaluate(x1_c)
        cs_1 = f1_c.imag/h
        x2_c = x0 + h*1j*pert_x2
        f2_c = evaluate(x2_c)
        cs_2 = f2_c.imag/h
        x3_c = x0 + h*1j*pert_x3
        f3_c = evaluate(x3_c)
        cs_3 = f3_c.imag/h
    
        cs= np.array([cs_1,cs_2,cs_3])
        cs= np.linalg.norm(cs)
    
        total_der = adjoint_total_derivative(x0)[0]
        total_der= np.linalg.norm(total_der)
    
        fd_error.append(abs(fd- total_der)/abs(total_der))
        cs_error.append(abs(cs- total_der)/abs(total_der))
        
    fig, ax = plt.subplots()
    ax.plot(h_list,fd_error,'b.')
    ax.plot(h_list,cs_error,'g.')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.legend(['FD','CS'],loc='best')
    plt.xlabel('Step Size')
    plt.ylabel('Relative Error') 

def ptB():
    x0 = np.array([1.1, 1.2, 1.3])

    total_der = adjoint_total_derivative(x0)[0]
    print('Adjoint-based derivative @x0:    ', total_der)

    h= 1e-30
    pert_x1= np.array([1,0,0])
    x1_c = x0 + h*1j*pert_x1
    f1_c = evaluate(x1_c)
    cs_1 = f1_c.imag/h
    
    pert_x2= np.array([0,1,0])
    x2_c = x0 + h*1j*pert_x2
    f2_c = evaluate(x2_c)
    cs_2 = f2_c.imag/h
    
    pert_x3= np.array([0,0,1])
    x3_c = x0 + h*1j*pert_x3
    f3_c = evaluate(x3_c)
    cs_3 = f3_c.imag/h
    
    cs= np.array([cs_1,cs_2,cs_3])
    print('Complex-step approximation:      ', cs)
    diff= abs(cs- total_der)/abs(total_der)
    print('Rel error between CS & Adjoint:  ', diff)
    
#%%
start_time = time.time()

h= 1e-30
ptA_compare(h)
h=1e-6
ptA_compare(h)

# ptA_graph()
# ptB()

print("\n--- %s seconds ---" % (time.time() - start_time))  
