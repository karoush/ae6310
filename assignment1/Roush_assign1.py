# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:04:22 2020

@author: kroush7
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib import rc
from ref import *
from searches import *
from roush_code import * #this is problems 1-3
import time

def problem4():
    ########
    # Steepest descent, strong wolfe 
    contour_plot(f1)
    x0 = [-1.0, -1.0]
    xstar = steepest_descent(x0, f1, f1_grad, c2=0.1, max_iters=100, line_search_type='strong Wolfe')
    print(xstar)
    plt.xlim(-2, 2)
    plt.ylim(-1.5, 1.5)
    plt.title('Steepest descent, strong Wolfe')
    plt.show()
    ########
    # Conjugate gradient
    contour_plot(f1)
    x0 = [-1.0, -1.0]
    xstar = conjugate_gradient(x0, f1, f1_grad, c2=0.1, max_iters=100, line_search_type='strong Wolfe')
    print(xstar)
    plt.xlim(-2, 2)
    plt.ylim(-1.5, 1.5)
    plt.title('Conjugate gradient, strong Wolfe')
    plt.show()

    ########
    # Steepest descent, strong wolfe 
    contour_plot(rosen)
    x0 = [1.5, 1.0]
    xstar = steepest_descent(x0, rosen, rosen_grad, c2=0.1, max_iters=100, line_search_type='strong Wolfe')
    print(xstar)
    plt.xlim(-2, 2)
    plt.ylim(-1.5, 1.5)
    plt.title('Steepest descent, strong Wolfe')
    plt.show()
    ########
    # Conjugate gradient
    contour_plot(rosen)
    x0 = [1.0, -1.0]
    xstar = conjugate_gradient(x0, rosen, rosen_grad, c2=0.1, max_iters=100, line_search_type='strong Wolfe')
    print(xstar)
    plt.xlim(-2, 2)
    plt.ylim(-1.5, 1.5)
    plt.title('Conjugate gradient, strong Wolfe')
    plt.show()

start_time = time.time()
problem1()
problem2_graph()
problem2_cp()
problem3()
problem4()
print("\n--- %s seconds ---" % (time.time() - start_time))