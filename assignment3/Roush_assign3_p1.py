# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:15:13 2020

@author: kroush7
"""

import numpy as np
import scipy.optimize as sp
import matplotlib.pylab as plt
import time

def func(x):
    #x=[x1,x2]
    return (x[0]+2)**2 +10*(x[1]+3)**2

def quad_penalty(x, *args):
    #defined everywhere
    if len(args) !=0:
        rho= args[0]
    else:
        rho= 5
    f_val= func(x) #function value
    c_val= con(x)*-1 #constraint value, have to *-1 to undo scipy
    return f_val+ (0.5*rho)*(max(c_val,0)**2)
  
def log_barrier(x, *args):
    #only defined in feasible space
    if len(args) !=0:
        mu= args[0]
    else:
        mu= 0.1
    f_val= func(x)
    c_val= con(x)*-1
    
    if c_val < 0:
        return f_val -(mu)*np.log(-c_val)
    else:
        return float('NaN')
    
def func_deriv(x):
    res= np.zeros((1, 2)) 
    dfdx1= 2*x[0]+4
    dfdx2= 20*x[1]+60 
    return np.array([dfdx1,dfdx2])

def func_jac(x):
    df2dx12= 2
    df2dx1x2= 0
    df2dx2x1= 0
    df2dx22= 20
    return np.array([[df2dx12,df2dx1x2],[df2dx2x1,df2dx22]])

def con(x):
    #x1^2 +x2^2 <= 2 
    # scipy wants the form of c(x) >= 0
    #-x1^2 -x2^2 +2 >= 0
    return -x[0]**2 -x[1]**2 +2

def con_deriv(x):
    res= np.zeros((1, 2)) 
    dcdx1= -2*x[0]
    dcdx2= -2*x[1]
    return np.array([dcdx1,dcdx2])

def lagrange_mult(x_star,delta_fstar):
    A_x= con_deriv(x_star)
    l_mult= np.dot(A_x,delta_fstar)/(-np.dot(A_x,np.transpose(A_x)))*-1
    return l_mult

def cplot(fobj,line,*args):
    n = 100
    x1 = np.linspace(-3.5, 2, n)
    x2 = np.linspace(-3.5, 1.5, n)
    X1, X2 = np.meshgrid(x1, x2)
    f = np.zeros((n, n))
    
    if line:
        for j in range(n):
            for i in range(n):
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
    
    #this is the constraint
    circ = plt.Circle((0, 0), 2**0.5, color='r',fill=False,linestyle='-')
    ax.add_artist(circ)
    
    if len(args) !=0:
        uncon_min=args[0]
        x_star=args[1]
        ax.plot(uncon_min[0],uncon_min[1],'ro')
        ax.plot(x_star[0],x_star[1],'ko')
    
    # ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.xlabel('x1')
    plt.ylabel('x2')  

def p1_ptA():
    # Plot the function with constraint and unconstrained min
    
    # find unconstrained min
    res = sp.minimize(func, [0,0], method='BFGS', options={'disp':False})
    uncon_min= res.x
    
    # find unconstrained min
    x0=[0,0]
    cons= ({'type':'ineq',
            'fun': con, 
            'jac': con_deriv})
    res = sp.minimize(func, x0, jac=func_deriv, constraints=cons, 
                      method='SLSQP', options={'disp':False})
    x_star=res.x
    delta_fStar= res.jac
    l_mult=lagrange_mult(x_star,delta_fStar)
    cplot(func,False,uncon_min,x_star)
    
    print('Constrained min= ',x_star)
    print('Langrange multiplier= ',l_mult)
    
def var_rho(): #use python
    n= 500
    x1 = np.linspace(-3.5, 2, n)
    x2 = np.linspace(-3.5, 2, n)
    rho= np.linspace(1,1000,n)
    f_evals= []
    lambda_est= []
    x0= [0,0]
    for current_rho in rho:
        res = sp.minimize(quad_penalty, x0, args=current_rho, method='BFGS', options={'disp':False})
        f_evals.append(res.nfev)
        current_L_est= current_rho*con(res.x)*-1
        lambda_est.append(current_L_est)
        
    fig, ax = plt.subplots(1, 1)  
    ax.set_xscale('log')
    ax.plot(rho,f_evals,'r+')
    fig.tight_layout()
    plt.xlabel('rho')
    plt.ylabel('Function evals')   
    
    fig, ax = plt.subplots(1, 1)    
    ax.plot(rho,lambda_est,'b+')
    ax.plot(rho, 11.3536*np.ones([1,n])[0],'k--')
    ax.set_xscale('log')
    fig.tight_layout()
    plt.xlabel('rho')
    plt.ylabel('Multiplier estimates')
    
def var_mu(): #use matlab
    n= 500
    x1 = np.linspace(-3.5, 2, n)
    x2 = np.linspace(-3.5, 2, n)
    mu= np.linspace(1,0.001,n)
    f_evals= []
    lambda_est= []
    x0= [0,0]
    for current_mu in mu:
        res = sp.minimize(quad_penalty, x0, args=current_mu, method='BFGS', options={'disp':False})
        f_evals.append(res.nfev)
        current_L_est= -current_mu/(con(res.x)*-1)
        lambda_est.append(current_L_est)
        
    fig, ax = plt.subplots(1, 1)    
    ax.plot(1/mu,f_evals,'r+')
    fig.tight_layout()
    plt.xlabel('1/mu')
    plt.ylabel('Function evals')   
    
    fig, ax = plt.subplots(1, 1)    
    ax.plot(1/mu,lambda_est,'b+')
    fig.tight_layout()
    plt.xlabel('1/mu')
    plt.ylabel('Multiplier estimates')

    
    
#%%
start_time = time.time()

p1_ptA()
# cplot(quad_penalty,False) #part 1B
# cplot(log_barrier,False) #part 1C

# # check to see what the new constrained min is
# res = sp.minimize(quad_penalty, [0,0], method='BFGS', options={'disp':False})
# x_star=res.x
# delta_fStar= res.jac
# l_mult=lagrange_mult(x_star,delta_fStar)
# print('Constrained min= ',x_star)
# print('Langrange multiplier= ',l_mult)
# cplot(log_barrier,False,[-2,-3],res.x ) #part 1C
    
#part 1D
# var_rho()
# var_mu()

print("\n--- %s seconds ---" % (time.time() - start_time))  
