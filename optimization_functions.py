# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:32:32 2022

@author: Oguzhan Savas
"""
# Optimization functions
import numpy as np

# Himmelblau Function
# Search space: [-5 : 5]
# Global optimas at:    f(3.0, 2.0) = 0.0
#                       f(-2.805118, 3.131312) = 0.0
#                       f(-3.779310, -3.283186) = 0.0
#                       f(3.584428, -1.848126) = 0.0
def himmelblau(x,y):
    return (((x**2+y-11)**2) + (((x+y**2-7)**2)))


# Ackley Function
# Search space: [-5 : 5]
# Global optima at;     f(0, 0) = 0.0
def ackley(x, y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


# Rastrigin Function
# Search space: [-5.12 : 5.12] 
# Global optima at:     f(0, 0) = 0.0
def rastrigin(x, y):
    return (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y)) + 20


# Tree-Hump-Camel Function
# Search space: [-5 : 5] 
# Global optima at:     f(0, 0) = 0.0
def three_hump_camel(x, y):
    return 2*(x**2) - 1.05*(x**4) + ((x**6)/6) + (x*y) + (y**2)


# SphereFunction
# Search space: [-inf : inf] 
# Global optima at:     f(0, 0) = 0.0
def sphere(x, y):
    return x**2 + y**2
