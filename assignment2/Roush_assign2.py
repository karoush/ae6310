# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:57:39 2020

@author: kroush7
"""
import numpy as np
import sympy as smp
import matplotlib.pylab as plt
import math
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
    #convert from sympy to numeric values
    a2= np.float64(B[0][0])
    b2= np.float64(B[0][1])
    c2= np.float64(B[1][0])
    d2= np.float64(B[1][1])
    B=np.array([[a2,b2],[c2,d2]])
    
    lam, Q = np.linalg.eigh(B) #find eigenvalues and eigenvectors of B
    VV= np.diag([lam[0],lam[1]])
    p= -1* (np.linalg.inv(B)).dot(g) 
    p_mag= (p[0,0]**2 + p[1,0]**2)**0.5 #linalg norm doesnt work due to sympy/numpy conflict
    if lam[0] >0 and p_mag <= delta: #smallest eigenvalue >0 = postive definite B
        #print('B is postive definite, and ||p||2 <=delta')
        lagrange_mult= 0
        p_star= p
    else:
        #print('B is not postive definite, or ||p||2 >delta')
        #Q[:,1] is second eigenvector
        q1= Q[:,0]
        q2= Q[:,1]
        
        L= smp.symbols('L',real=True)
        coeff1= (np.dot(q1,g)**2)*(L**2 +2*L*lam[0]+lam[0]**2)**(-1)
        coeff2= (np.dot(q2,g)**2)*(L**2 +2*L*lam[0]+lam[1]**2)**(-1)
        solution= smp.solve(coeff1+coeff2-delta**2, L)
        lams=[]
        # print(lam)
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
        # print(p_star)
        if p_star_mag >= delta: #if exact step is outside trust region
            p_star[0,0]=(p_star[0,0]*delta)/p_star_mag
            p_star[1,0]=(p_star[1,0]*delta)/p_star_mag
            onBound= True
    return p_star, lagrange_mult, onBound

def cplot(g,b,r,title,line,c,ex, p1):
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
    if p1:
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
    cplot(g1,b1,1,'P1, Case1: Cauchy step= red, exact step= green',True,c,ex,True)
    
    #Problem 1, second one
    g2= (1/(2**0.5))* np.array([[-1,0]])
    b2= np.array([[3,1],[1,3]])
    r=0.5
    c= np.array([[0.23570226],[0]])
    ex= np.array([[0.26516504],[-0.08838835]])
    cplot(g2,b2,r,'P1, Case2: Cauchy step= red, exact step= green',True,c,ex,True)
    
    #Problem 1, third one
    g3= np.array([[-1,0]])
    b3= np.array([[1,0],[0,-2]])
    r=1
    c= np.array([[0.70710678],[0.70710678]])
    ex= np.array([[0.212746894185773],[0.977107342626340]])
    cplot(g3,b3,r,'P1, Case3: Cauchy step= red, exact step= green',True,c,ex,True)
    
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

def problem2():
    x0= np.array([[0,0]])
    
    print('Case 1: Minimizer in the interior of the trust region')
    fobj, g, b= case1(x0)
    delta= 0.2
    p_cauchy= cauchy_step(g,b,delta)
    p_star, lagrange_mult, onBound= trust_region_step(g,b,delta)
    cplot_2(case1, x0, delta, p_cauchy, p_star, 'Test Case 1: Cauchy step= red, exact step= green',True)

    print('\nCase 2: Positive deﬁnite model, but minimizer is constrained')
    fobj, g, b= case2(x0)
    delta= 0.15
    p_cauchy= cauchy_step(g,b,delta)
    p_star, lagrange_mult, onBound= trust_region_step(g,b,delta)
    cplot_2(case2, x0, delta, p_cauchy, p_star, 'Test Case 2: Cauchy step= red, exact step= green',True)
    
    print('\nCase 3: Indeﬁnite/negative deﬁnite Hessian')
    fobj, g, b= case3(x0)
    delta= 0.2
    p_cauchy= cauchy_step(g,b,delta)
    p_star, lagrange_mult, onBound= trust_region_step(g,b,delta)
    cplot_2(case3, x0, delta, p_cauchy, p_star, 'Test Case 3: Cauchy step= red, exact step= green',True)

def p3_f1(x):
    #x0= [x1;x2]
    x1= x[0][0]
    x2= x[1][0]
    fobj= -10*(x1**2) +10*(x2**2) +4*math.sin(x1*x2) -2*x1+ x1**4
    g= np.array([[-20*x1+4*x2*math.cos(x1*x2)-2+4*x1**3],[20*x2+4*x1*math.cos(x1*x2)]])
    return fobj, g

def p3_f2(x):
    #x0= [x1;x2]
    x1= x[0][0]
    x2= x[1][0]
    fobj= 100*(x2-x1**2)**2 +(1-x1)**2
    g= np.array([[-200*x1**3+2*x1*(200*x2 +1)-2],[200*x1**2 -200*x2]])
    return fobj, g 

def problem3_graphs(func,title):
    n = 500
    x1 = np.linspace(-5, 5, n)
    x2 = np.linspace(-5, 5, n)
    X1, X2 = np.meshgrid(x1, x2)
    f1 = np.zeros((n, n))
     
    for j in range(n):
        for i in range(n):
            x= np.array([[X1[i,j], X2[i,j]]])
            x=np.transpose(x)
            if func == 'p3_f1':
                fobj, g= p3_f1(x)
                f1[i, j] = fobj
            elif func == 'p3_f2':
                fobj, g= p3_f2(x)
                f1[i, j] = fobj
            else:
                print('specify a function name')
    fig, ax= plt.subplots(1, 1)
    ax.contour(X1, X2, f1)
    # fig.tight_layout()
    plt.xlabel('x1')
    plt.ylabel('x2') 
    plt.title(title)
    
def problem3_fig():
    funcs= ['p3_f1','p3_f2']
    titles= ['P3, Function 1','P3, Function 2']
    i= 0
    while i <len(funcs):
        problem3_graphs(funcs[i],titles[i])
        i+=1
        
def trust_algo(desc,fobj, x0,cauchy, tol, delta, delta_max, eta, breakout):
    # x0= np.array([[0,0]])
    f_x, g_x = fobj(x0)
    f_xk= None
    g_xk= None
    b= np.identity(2)
    k=1
    norm2_gk= (g_x[0][0]**2 + g_x[1][0]**2)**0.5
    xk= x0 #if you start at a point closer to the min than the tolerance
    # while k<3:
    while norm2_gk > tol:
        #find the amount to move to
        if cauchy:
            pk= cauchy_step(g_x,b,delta) 
        else:
            pk, lagrange_mult, onBound = trust_region_step(g_x,b,delta)
        xk= x0+pk #move to new point
        # f_x, g_x = fobj(x0) #function value and grad at OLD point
        f_xk, g_xk = fobj(xk) #function value and grad at NEW point
        m_x= p3_model(x0,g_x,b) #model value at OLD point
        m_xk= p3_model(xk,g_xk,b) #model value at NEW point
        
        rho_k= (f_x - f_xk)/((m_x-m_xk)[0][0]) #convergance parameter
        # print(rho_k)
        if breakout:
            if rho_k <0:
                break
        if str(rho_k)=='zoo' or isinstance(rho_k, complex) or math.isnan(rho_k):
            break
        
        yk= g_xk - g_x #yk= grad@newPoint - grad@oldPoint
        sk= pk
        #update B using SR-1 formula
        b= b+ (np.dot((yk- np.dot(b,sk)), np.transpose(yk-np.dot(b,sk)))/ (np.dot(np.transpose(sk),(yk- np.dot(b,sk)))))
        x0= set_new_point(rho_k,eta, x0, xk) #set the new x0
        delta= set_new_radius(rho_k, delta, pk, delta_max)
        
        f_x= f_xk
        g_x=g_xk
        norm2_gk= (g_x[0][0]**2 + g_x[1][0]**2)**0.5
        k+=1 #update iterations
        
    print(desc)
    print('Iterations=',k)
    print('Min at:', xk,'\n')
    
def p3_model(p,g,b):
    #p=[p1;p2]
    return np.dot(np.transpose(p),g) +0.5*(np.dot(np.transpose(p),np.dot(b,p)))

def set_new_point(rho_k,eta, x0, xk):
    if rho_k >= eta:
        x0=xk
    else:
        x0=x0
    return x0

def set_new_radius(rho_k, delta, pk, delta_max):
    pk_norm= (pk[0][0]**2 + pk[1][0]**2)**0.5 #that damn numpy/sympy conversion; use lambdify
    if rho_k <0.25:
        delta= 0.25*delta
    # elif (rho_k > 0.75) and (np.linalg.norm(pk,2)== delta):
    elif (rho_k > 0.75) and (pk_norm== delta):
        delta= min(2*delta, delta_max)
    else:
        delta= delta
    return delta

def problem3(x0, tol, delta, delta_max, eta):
    breakout= True #change this to modify breakout behavior
    print('Starting point=', x0)
    print('Tolerance=',tol ,'\n')
    trust_algo('f1, Cauchy Step',p3_f1,x0,True,tol, delta, delta_max, eta, breakout) #Function 1, cauchy; works 
    trust_algo('f1, Exact Step', p3_f1,x0,False,tol, delta, delta_max, eta, breakout) #Function 1, exact step 
    trust_algo('f2, Cauchy Step',p3_f2,x0,True,tol, delta, delta_max, eta, breakout) #Function 2, cauchy 
    trust_algo('f2, Exact Step', p3_f2,x0,False,tol, delta, delta_max, eta, breakout) #Function 2, exact step

#%%
start_time = time.time()

problem1_graphs()
problem1_points()
problem2()
problem3_fig() #graphs

x0= np.array([[0],[0]])
tol= 1e-9
delta= 0.25
delta_max= 2
eta= 0.05
problem3(x0, tol ,delta, delta_max, eta)

print("\n--- %s seconds ---" % (time.time() - start_time))  
#%% 