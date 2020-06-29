# -*- coding: utf-8 -*-
"""
This is for AE6310 Assignment #4, Problem 2
Created on Fri Apr 17 18:57:30 2020

@author: kroush7
"""

import numpy as np
import scipy.optimize as sp
import time
import math

def obj_func(x):
    '''returns the value of the main function'''
    #x[0]= x1, x[1]= x2
    return abs(x[0]*x[1]) +0.1*(x[0]**2 +x[1]**2)

def checkTolerance(tolerance,x0):
    print('\nTolerance= ', tolerance)
    print('Gradient-based method (BFGS): ')
    res_grad= sp.minimize(obj_func, x0, method= 'BFGS', tol=tolerance, options={'disp':True})
    print('Gradient-free method (Nelder-Mead): ')
    res_gradFree= sp.minimize(obj_func, x0, method= 'Nelder-Mead', tol= tolerance, options={'disp':True})
    print('Gradient-free method (Powell): ')
    res_gradFree= sp.minimize(obj_func, x0, method= 'Powell',tol=tolerance,options={'disp':True})


#%%
start_time = time.time()

x0= [-1,1]
## Return minimizer
print('-----PART A, find the minimizer-----')
print('Gradient-based method (BFGS): ')
res_grad= sp.minimize(obj_func, x0, method= 'BFGS', options={'disp':True})
print('Min at: ', res_grad.x)
print('\nGradient-free method (Nelder-Mead): ')
res_gradFree= sp.minimize(obj_func, x0, method= 'Nelder-Mead', options={'disp':True})
print('Min at: ', res_gradFree.x)
print('\nGradient-free method (Powell): ')
res_gradFree= sp.minimize(obj_func, x0, method= 'Powell',options={'disp':True})
print('Min at: ', res_gradFree.x)

## Check if tighter tolerances modify behavior
print('\n-----PART B, check tolerancing behavior-----')
tolerance= 1e-30
checkTolerance(tolerance,x0)
tolerance= 1e-20
checkTolerance(tolerance,x0)
tolerance= 1e-15
checkTolerance(tolerance,x0)
tolerance= 1e-10
checkTolerance(tolerance,x0)


print("\n--- %s seconds ---" % (time.time() - start_time))  