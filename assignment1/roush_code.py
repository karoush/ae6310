# -*- coding: utf-8 -*-
"""
This has all the code I have written for the assigment

@author: kroush7
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib import rc
from ref import *

rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{sfmath}')

def f2a(x):
    return x[0]**4 +x[1]**4 +1 -x[0]**2 +x[1]**2
def f2b(x):
    return x[0]**2 +x[1]**2 +2*x[0]*x[1]
def f2c(x):
    return 4*x[0]**2 +x[1]**2 +x[0]*x[1] +x[0]
def f2d(x):
    return x[0]**4 +x[1]**2 +2*x[0]*x[1] -x[0] -x[1]

def problem1():
    A = np.array([[2, 1], [1, 2]])
    b = np.array([-1,-1])
    print('P1a:')
    quadratic_decomposition(A, b) #P1a
    A = np.array([[2, 2], [2, 2]])
    b = np.array([-1,-1])
    print('P1b:')
    quadratic_decomposition(A, b) #P1b
    A = np.array([[6, 2], [2, 6]])
    b = np.array([-1,-1])
    print('P1c:')
    quadratic_decomposition(A, b) #P1c
    A = np.array([[8, 2], [2, 4]])
    b = np.array([1,1])
    print('P1d:')
    quadratic_decomposition(A, b) #P1d

def problem2_graph():
    funcs = [f2a, f2b, f2c, f2d]
    for fobj in funcs:
        cplot(fobj, True)
    plt.show()

def problem2_cp():
    #part A
    gradf_x1=[4,0,-2,0] #4*x1**2-2*x1
    gradf_x2=[4,0,2,0] #4*x1**2+2*x1
    x1_cp=np.roots(gradf_x1)
    x2_cp=np.roots(gradf_x2)
    print('P2A: X1_cp=', x1_cp)
    print('P2A: X2_cp=', x2_cp)
    for i in x1_cp:
        for j in x2_cp:
            x=[i,j]
            H=[[12*x[0]**2-2, 0],[0, 12*x[1]**2+2]]
            eig, Q = np.linalg.eigh(H)
            print(eig)
            print('X1_cp, X2_cp=', x)
            print(H)
            print('\n')
    #part B
    A = np.array([[2, 2], [2, 2]])
    b = np.array([0,0])
    quadratic_decomposition(A, b)
    #Part C
    A = np.array([[8, 1], [1, 2]])
    b = np.array([0,0])
    quadratic_decomposition(A, b) 
    
    #part D
    print('\nPart D')
    x1_cp= [-0.5, 0, 0.5] #4*x1**3 +2*x2 -1
    x2_cp= [0.75, 0.5, 0.25] #2*x2 +x1 -1
    i=0
    while i<len(x1_cp):
        x=[x1_cp[i], x2_cp[i]]
        H=[[12*x[0]**2,2],[1,2]]
        eig, Q = np.linalg.eigh(H)
        print(eig)
        print('X1_cp, X2_cp=', x)
        print(H)
        print('\n')
        i+=1
     
def problem3():
    #phi(alpha)=alpha*(1-alpha)**2 *(alpha-3)
    #phi'(alpha)= 4*(alpha)**3 -15*(alpha)**2 +14*(alpha) -3
    alpha=np.linspace(-1, 4, 1000)
    merit_func=alpha*(1-alpha)**2 *(alpha-3)
    plt.plot(alpha, merit_func, "-k", label="$\phi (alpha)$")
    
    #PART A
    #phi(0)=0, phi'(0)= -3
    c=[0.01, 0.1, 0.5]
    suff_dec_1= 0 +c[0]*alpha*(-3)
    suff_dec_2= 0 +c[1]*alpha*(-3)
    suff_dec_3= 0 +c[2]*alpha*(-3)
    
    plt.plot(alpha, suff_dec_1, ":b", label="c1=0.01")
    plt.plot(alpha, suff_dec_2, ":g", label="c1=0.1")
    plt.plot(alpha, suff_dec_3, ":r", label="c1=0.5")
    plt.legend(loc="upper left")
    plt.title('Armijo Condition')
    plt.xlabel("alpha")
    plt.ylabel('value')
    plt.show()
    
    
    #PART B
    merit_func_der= (alpha-1)* (4*alpha**2 -11*alpha +3)
    yval= np.ones(len(alpha))
    plt.plot(alpha, merit_func_der, "-k", label="$\phi'(alpha)$")
    
    c2=[0.9,0.5,0.1]
    wolfe2_1= c2[0]*(-3)
    wolfe2_2= c2[1]*(-3)
    wolfe2_3= c2[2]*(-3)
    ywolfe2_1= yval*wolfe2_1
    ywolfe2_2= yval*wolfe2_2
    ywolfe2_3= yval*wolfe2_3
    
    plt.plot(alpha, ywolfe2_1, ":b", label="c2=0.9")
    plt.plot(alpha, ywolfe2_2, ":g", label="c2=0.5")
    plt.plot(alpha, ywolfe2_3, ":r", label="c2=0.1")
    plt.legend(loc="upper left")
    plt.title('Second Wolfe Condition')
    plt.xlabel("alpha")
    plt.ylabel('value')
    plt.show()
    
    #PART C
    #c1=0.01; c2=0.9
    alpha_good_1=[]
    phi_good_1=[]
    i=0
    while i<len(alpha):
        if (merit_func[i]<= suff_dec_1[i]) and (abs(merit_func_der[i])<=abs(wolfe2_1)):
            #print(alpha[i],"satisfies both conditions")
            alpha_good_1.append(alpha[i])
            phi_good_1.append(merit_func[i])
        else:
            pass
        i+=1
    #c1=0.1; c2=0.5
    alpha_good_2=[]
    phi_good_2=[]
    i=0
    while i<len(alpha):
        if (merit_func[i]<= suff_dec_2[i]) and (abs(merit_func_der[i])<=abs(wolfe2_2)):
            #print(alpha[i],"satisfies both conditions")
            alpha_good_2.append(alpha[i])
            phi_good_2.append(merit_func[i])
        else:
            pass
        i+=1
    plt.scatter(alpha_good_1,phi_good_1, marker=".", facecolors='none', 
                edgecolors='r', label="c1=0.01, c2=0.9")
    plt.scatter(alpha_good_2,phi_good_2, marker=".", facecolors='none', 
                edgecolors='b', label="c1=0.1, c2=0.5")
    plt.title('Intervals that satisfy Strong Wolfe Conditions')
    plt.legend(loc="lower left")
    plt.xlabel("alpha")
    plt.ylabel('$\phi (alpha)$')
    plt.show()
    
# =============================================================================
#     plt.plot(alpha, merit_func, "-k", label="$\phi (alpha)$")
#     plt.plot(alpha, suff_dec_1, ":b", label="c1=0.01")
#     plt.plot(alpha, suff_dec_2, ":g", label="c1=0.1")
#     plt.legend(loc="upper left")
#     plt.title('First Wolfe Condition')
#     plt.xlim((-0.5,3.5))
#     plt.ylim((-1,2))
#     plt.xlabel("alpha")
#     plt.ylabel('value')
#     plt.show()   
#     plt.plot(alpha, abs(merit_func_der), "-k", label="$|\phi'(alpha)|$")
#     plt.plot(alpha, abs(ywolfe2_1), ":b", label="c2=0.9")
#     plt.plot(alpha, abs(ywolfe2_2), ":g", label="c2=0.5")
#     plt.legend(loc="upper left")
#     plt.title('Second Strong Wolfe Condition')
#     plt.xlim((-0.5,3))
#     plt.ylim((-1,5))
#     plt.xlabel("alpha")
#     plt.ylabel('value')
#     plt.show()
# =============================================================================