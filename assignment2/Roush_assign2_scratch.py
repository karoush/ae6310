# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:57:39 2020

@author: kroush7
"""
import numpy as np
import sympy as smp
import matplotlib.pylab as plt
import time

def cauchy_step(g,B,delta):
    if np.transpose(g).dot(B.dot(g)) <= 0:
        tau= 1
    else:
        temp= (np.linalg.norm(g,ord=2)**3)/(delta*(np.transpose(g)).dot(B.dot(g)))
        tau= min(1, temp)
    p= -tau*(delta/np.linalg.norm(g,ord=2))*g
    return p

def trust_region_step(g,B,delta):
    onBound= False
    lam, Q = np.linalg.eigh(B) #find eigenvalues and eigenvectors of B
    VV= np.diag([lam[0],lam[1]])
    p= -1* (np.linalg.inv(B)).dot(g) 
    if lam[0] >0 and np.linalg.norm(p,ord=2) <= delta: #smallest eigenvalue >0 = postive definite B
        print('B is postive definite, and ||p||2 <=delta')
        lagrange_mult= 0
        p_star= p
    else:
        print('B is not postive definite, or ||p||2 >delta')
        #Q[:,1] is second eigenvector
        q1= Q[:,0]
        q2= Q[:,1]
# =============================================================================
#         c1= q1.dot(g)**(2)+q2.dot(g)**(2)
#         c2= 2*q1.dot(g)**(2)*lam[0] +2*q2.dot(g)**(2)*lam[1]
#         c3= q1.dot(g)**(2)*lam[0]**2 + q2.dot(g)**(2)*lam[1]**2 -(1/delta**2)
#         coefs= [c1,c2,c3]
#         # r = np.roots(coefs)
#         # print(r)
# =============================================================================
        
        L= smp.symbols('L',real=True)
        coeff1= (np.dot(q1,g)**2)*(L**2 +2*L*lam[0]+lam[0]**2)**(-1)
        coeff2= (np.dot(q2,g)**2)*(L**2 +2*L*lam[0]+lam[1]**2)**(-1)
        solution= smp.solve(coeff1+coeff2-delta**2, L)
        lams=[]
        lagrange_mult= []
        for i in solution:
            lams.append(i[0])
        # print(lams)
        for i in lams:
            if i>0 and i>-(lam[0]):
                lagrange_mult.append(i)
        lagrange_mult= lagrange_mult[0]
        
        A= np.array(VV+ lagrange_mult*np.identity(2))
        A= smp.Matrix(A).inv()
        A= np.array(A)
        p_star= np.dot(-Q, np.dot(A, np.dot(np.transpose(Q),g)))
        p_star_mag= (p_star[0,0]**2 + p_star[1,0]**2)**0.5
        
        if p_star_mag >= delta: #if exact step is outside trust region
            p_star[0,0]=(p_star[0,0]*delta)/p_star_mag
            p_star[1,0]=(p_star[1,0]*delta)/p_star_mag
            onBound= True
# =============================================================================
#         p= smp.Matrix(B+L*smp.eye(2))**-1 *-smp.Matrix(g)
#         p_mag= p.row(0)**2 + p.row(1)**2
#         res= smp.solveset((p_mag[0])-delta**2,L)
#         lams=[]
#         lagrange_mult= []
#         for x in res.args:
#             # x=str(x)
#             # x=x[0:12]
#             # x=float(x)
#             lams.append(x)
#         for i in lams:
#             if i>0 and i>-(lam[0]):
#                 lagrange_mult.append(i)
#         sub= lagrange_mult[0]
#         p_star= smp.Matrix(B+sub*smp.eye(2))**-1 *-smp.Matrix(g)
#         p_star_mag= p_star.row(0)**2 + p_star.row(1)**2 #its actually ||p||2**2
#         p_star_mag= np.array(p_star_mag)
#         p_star=np.array(p_star)
#         if p_star_mag==delta**2:
#             onBound= True
# =============================================================================
    return p_star, lagrange_mult, onBound

def cplot(g,b,r,title,line,c,ex):
    n = 50
    x1 = np.linspace(-2, 2, n)
    x2 = np.linspace(-2, 2, n)
    X1, X2 = np.meshgrid(x1, x2)
    f = np.zeros((n, n))
     
    if line:
        for j in range(n):
            for i in range(n):
                p= np.array([[X1[i,j], X2[i,j]]])
                #print(p)
                f[i, j] = p1_model(p,g,b)
                #print(f[i,j])
        fig, ax = plt.subplots(1, 1)
        ax.contour(X1, X2, f)
        
    x_c= c[0][0]
    y_c= c[1][0]
    x_ex= ex[0][0]
    y_ex= ex[1][0]
    plt.plot(x_c, y_c, 'rx') #cauchy step= red
    plt.plot(x_ex, y_ex, 'g+') #exact step= green
    
    ax.set_aspect('equal', 'box')
    circle1 = plt.Circle((0, 0), r, color='r', fill=False)
    ax.add_artist(circle1)
    fig.tight_layout()
    plt.xlabel('p1')
    plt.ylabel('p2') 
    plt.title(title)
    
def p1_model(p,g,B):
    #p=[p1, p2]
    return p.dot(np.transpose(g))+ 0.5* p.dot(B.dot(np.transpose(p)))

def problem1_graphs():
    #Problem 1, case1
    g1= (1/(2**0.5))* np.array([[-1,0]])
    b1= np.array([[3,1],[1,3]])
    r= 1;
    c= np.array([[0.23570226],[0]])
    ex= np.array([[0.26516504],[-0.08838835]])
    cplot(g1,b1,1,'First case',True,c,ex)
    
    #Problem 1, second one
    g2= (1/(2**0.5))* np.array([[-1,0]])
    b2= np.array([[3,1],[1,3]])
    r=0.5
    c= np.array([[0.23570226],[0]])
    ex= np.array([[0.26516504],[-0.08838835]])
    cplot(g1,b1,r,'Second case',True,c,ex)
    
    #Problem 1, third one
    g3= np.array([[-1,0]])
    b3= np.array([[1,0],[0,-2]])
    r=1
    c= np.array([[0.70710678],[0.70710678]])
    ex= np.array([[0.212746894185773],[0.977107342626340]])
    cplot(g3,b3,r,'Third case',True,c,ex)
    
def problem1_points():
    g1= (1/(2**0.5))* np.array([[-1],[0]])
    b1= np.array([[3,1],[1,3]])
    r1= 1;
    p=cauchy_step(g1,b1,r1)
    print('Cauchy step, case 1=\n', p)
    p_star, lam_multiplier, onBound= trust_region_step(g1,b1,r1)
    print('Exact step, case 1=\n', p_star)
    print('Lagrange multiplier, case1= ', lam_multiplier)
    print('onBound= ', onBound)
    print('\n')
    
    g2= (1/(2**0.5))* np.array([[-1],[0]])
    b2= np.array([[3,1],[1,3]])
    r2= 0.5;
    p2=cauchy_step(g2,b2,r2)
    print('Cauchy step, case 2=\n', p2)
    p_star2, lam_multiplier2, onBound2= trust_region_step(g2,b2,r2)
    print('Exact step, case 2=\n', p_star2)
    print('Lagrange multiplier, case2= ', lam_multiplier2)
    print('onBound= ', onBound2)
    print('\n')
    
    g3= np.array([[-1],[-1]])
    b3= np.array([[1,0],[0,-2]])
    r3= 1;
    p3=cauchy_step(g3,b3,r3)
    print('Cauchy step, case 3=\n', p3)
    p_star3, lam_multiplier3, onBound3= trust_region_step(g3,b3,r3)
    print('Exact step, case 3=\n', p_star3)
    print('Lagrange multiplier, case3= ', lam_multiplier3)
    print('onBound= ', onBound3)
    print('\n')
        
def case1(x):
    #x0= [x1,x2]
    #minimizer inside trust region
    x1= x[0][0]
    x2= x[0][1]
    fobj= 3*x1**2 +2*x1*x2 +3*x2**2 -x1 -x2
    g= np.array([[6*x1+2*x2 -1],[2*x1+6*x2-1]])
    b= np.array([[6,2],[2,6]])
    return fobj, g, b

def case2(x):
    #positive definite, minimizer outside trust region
    x1= x[0][0]
    x2= x[0][1]
    fobj= x1**2+x1*x2+x2**2-x1-x2
    g= np.array([[2*x1+x2-1],[x1+2*x2-1]])
    b= np.array([[2,1],[1,2]])
    return fobj, g, b

def case3(x):
    #indefinite hessian
    x1= x[0][0]
    x2= x[0][1]
    fobj= 4*x1**2 +6*x1*x2 +2*x2**2 +x1+x2
    g= np.array([[8*x1+6*x2+1],[6*x1+4*x2+1]])
    b= np.array([[8,6],[6,4]])
    return fobj, g, b

def cplot_2(fobj,i_point,r,Cauchy_x, exact_x, title,line):
    n = 100
    x1 = np.linspace(-1, 1, n)
    x2 = np.linspace(-1, 1, n)
    X1, X2 = np.meshgrid(x1, x2)
    f = np.zeros((n, n))
     
    if line:
        for j in range(n):
            for i in range(n):
                x= np.array([[X1[i,j], X2[i,j]]])
                fout, g, b = fobj(x)
                f[i, j] = fout
        fig, ax = plt.subplots(1, 1)
        ax.contour(X1, X2, f)
    
    x_i= i_point[0][0]
    y_i= i_point[0][1]
    x_c= Cauchy_x[0][0]
    y_c= Cauchy_x[1][0]
    x_ex= exact_x[0][0]
    y_ex= exact_x[1][0]
    plt.plot(x_c, y_c, 'rx') #cauchy step= red
    plt.plot(x_ex, y_ex, 'g+') #exact step= green
    
    circle1 = plt.Circle((x_i, y_i), r, color='r', fill=False)
    ax.add_artist(circle1)
    fig.tight_layout()
    plt.xlabel('x1')
    plt.ylabel('x2') 
    plt.title(title)
        # plt.title(title)

def check_alg():
    x0= np.array([[0,0]])
    
    fobj, g, b= case1(x0)
    delta= 0.2
    p_cauchy= cauchy_step(g,b,delta)
    p_star, lagrange_mult, onBound= trust_region_step(g,b,delta)
    cplot_2(case1, x0, delta, p_cauchy, p_star, 'Case 1: Cauchy step= red, exact step= green',True)

    fobj, g, b= case2(x0)
    delta= 0.2
    p_cauchy= cauchy_step(g,b,delta)
    p_star, lagrange_mult, onBound= trust_region_step(g,b,delta)
    cplot_2(case2, x0, delta, p_cauchy, p_star, 'Case 2: Cauchy step= red, exact step= green',True)
    
    fobj, g, b= case3(x0)
    delta= 0.2
    p_cauchy= cauchy_step(g,b,delta)
    p_star, lagrange_mult, onBound= trust_region_step(g,b,delta)
    cplot_2(case3, x0, delta, p_cauchy, p_star, 'Case 3: Cauchy step= red, exact step= green',True)

start_time = time.time()
problem1_graphs()
problem1_points()
# check_alg()

print("\n--- %s seconds ---" % (time.time() - start_time))   