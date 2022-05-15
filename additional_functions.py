# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:29:43 2022

@author: Oguzhan Savas
"""
# Additional functions
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from optimization_functions import *
from sklearn.model_selection import train_test_split


def user_defined_input():
    upper_bound = st.sidebar.slider("Upper Boundary", min_value=0, max_value=10, step=1)
    lower_bound = st.sidebar.slider("Lower Boundary", min_value=-10, max_value=0, step=1)
    grid_division = st.sidebar.slider("Grid Division", min_value=10, max_value=100, step=10)
    data = {"upper_bound" : upper_bound,
            "lower_bound" : lower_bound,
            "grid_division" : grid_division}
    inputs = pd.DataFrame(data, index=[0])
    return inputs


#def get_search_space(func):
#    if func == himmelblau:
#        lower_bound, upper_bound = -5, 5
#        
#    elif func == ackley:
#        lower_bound, upper_bound = -5, 5
#        
#    elif func == three_hump_camel:
#        lower_bound, upper_bound = -2, 2
#        
#    elif func == sphere:
#        lower_bound, upper_bound = -5, 5
#        
#    elif func == rastrigin:
#        lower_bound, upper_bound = -5.12, 5.12
#        
#    return lower_bound, upper_bound
        

def create_grid(lower_bound, upper_bound, grid_division):
    X = np.linspace(lower_bound, upper_bound, grid_division)
    Y = np.linspace(lower_bound, upper_bound, grid_division)
    x_grid, y_grid = np.meshgrid(X, Y)
    return X, Y, x_grid, y_grid


def compute_real_result(x_grid, y_grid, opt_func):
    F_real = opt_func(x_grid, y_grid)
    return F_real


def plot_real_result(X, Y, Z):
    fig = plt.figure(figsize=(4,4))
    ax = plt.axes(projection='3d')
    ax.contour3D(X,Y,Z,450)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F_real')
    #ax.set_title('Real Results of the Optimization Function')
    ax.view_init(40,40)
    return fig
    
    
def data_prep(x_grid, y_grid, func):
    # data to train the ml model
    X_ml = x_grid.reshape((np.size(x_grid)))
    Y_ml = y_grid.reshape((np.size(y_grid)))
    F = func(X_ml, Y_ml)
    
    ml_data_dict = {"X_ml"      : X_ml,
                    "Y_ml"      : Y_ml,
                    "F_real"    : F}
    
    ml_df = pd.DataFrame(ml_data_dict)
    
    # Independent variables
    ind_var = ml_df.drop(["F_real"], axis=1)
    # Target variable
    target_var = ml_df["F_real"]
    # Create holdout, train, and test data
    X_interm, X_holdout, y_interm, y_holdout = train_test_split(ind_var, target_var, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_interm, y_interm, test_size=0.3, random_state=42)
    
    return ml_df, X_holdout, y_holdout, X_train, X_test, y_train, y_test


def plot_ml_result(X_test, preds):
    # coordinates for ml results
    x_axis_test = X_test["X_ml"]
    y_axis_test = X_test["Y_ml"]

    my_cmap = plt.get_cmap('jet')

    fig = plt.figure(figsize =(4, 4))
    ax = plt.axes(projection ='3d')
     
    trisurf = ax.plot_trisurf(x_axis_test, y_axis_test, preds,
                             cmap = my_cmap,
                             linewidth = 0.2,
                             antialiased = True,
                             edgecolor = 'grey')
    ax.set_xlabel('X_coord')
    ax.set_ylabel('Y_coord')
    ax.set_zlabel('ML_predictions')
    #ax.set_title('ML Results of the Optimization Function')
    ax.view_init(40,40)
    return fig
