# -*- coding: utf-8 -*-
"""
Created on Wed May  4 23:16:13 2022

@author: Oguzhan Savas
"""
import sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import ensemble

from optimization_functions import *
from additional_functions import *

st.write("""
         # Predicting Optimization Functions with Machine Learning
         
         This app uses a **gradient boost regressor** to predict the outcomes of **optimization functions**.
         """)

st.sidebar.header("User defined inputs")

func = st.sidebar.selectbox(
    "Choose optimization function:",
    ("Himmelblau", "Ackley", "Rastrigin", "Three Hump Camel", "Sphere"))

functions = {"Himmelblau" : himmelblau,
             "Ackley" : ackley,
             "Rastrigin" : rastrigin,
             "Three Hump Camel" : three_hump_camel,
             "Sphere" : sphere}

opt_func = functions.get(func)

inputs = user_defined_input()

st.subheader("User input parameters")
st.write('*Selected function:*', func)
st.write(inputs)

#opt_func = inputs["opt_func"].to_string()
lower_bound = int(inputs["lower_bound"])
upper_bound = int(inputs["upper_bound"])
grid_division = int(inputs["grid_division"])

st.subheader("Real Results of the Optimization Function")

X, Y, x_grid, y_grid = create_grid(lower_bound, upper_bound, grid_division)
F_real = compute_real_result(x_grid, y_grid, opt_func)

fig_0 = plot_real_result(X, Y, F_real)
st.pyplot(fig_0)


# data prep
ml_df, X_holdout, y_holdout, X_train, X_test, y_train, y_test = data_prep(x_grid, y_grid, opt_func)

# Machine learning model -- make these user defined!!
params = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 10,
    "learning_rate": 0.05}

# define the gradient boost model
model = ensemble.GradientBoostingRegressor(**params)

model.fit(X_train, y_train)
preds = model.predict(X_test)

preds_holdout = model.predict(X_holdout)

st.subheader("Machine Learning Results")
st.write("**Plotting Predictions**")
fig_1 = plot_ml_result(X_test, preds)
st.pyplot(fig_1)

st.write("**Plotting Holdout Data**")
fig_2 = plot_ml_result(X_holdout, preds_holdout)
st.pyplot(fig_2)

# error inspection
st.subheader("Error Inspection")

rmse = np.sqrt(MSE(y_test, preds))
st.write("The *root mean squared error* of the test set is:", rmse)


rmse_holdout = np.sqrt(MSE(y_holdout, preds_holdout))
st.write("The *root mean squared error* of the holdout set is:", rmse_holdout)

