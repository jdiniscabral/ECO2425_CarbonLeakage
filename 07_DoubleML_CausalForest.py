####################
# References:

#Code for regression results adapted from replication package of:
# Aichele, R. and G. Felbermayr (2015): “Kyoto and Carbon Leakage: An Empirical Analysis of the Carbon Content of
# Bilateral Trade,” The Review of Economics and Statistics, 97, 104–115.
# Obtained from AEJ

#Code for LASSO and Tree-based methods based on Jupyter codes from:
# James, G. M., D. Witten, T. Hastie, R. Tibshirani, and J. Taylor (2023): 
# An Introduction to Statistical Learning: with Applications in Python, Springer.

#Code for DAG, Refutation Tests, Cummulative Gain plots based on codes from:
# Molak, A. (2023): Causal Inference and Discovery in Python: Unlock the secrets of modern
# causal machine learning with DoWhy, EconML, PyTorch and more, Packt Publishing Ltd

#Code for Matching, IPW, Meta-Learners, Linear DR learner, Linear Double ML based codes from:
# Facure Alves, M. (2022): Causal Inference for the Brave and True. Available:
# https://matheusfacure.github.io/python-causality-handbook/landing-page.html

#Code for all other DR learners, Double ML based on:
# Battocchi, K., Dillon, E., Hei, M., Lewis, G., Oka, P., Oprescu, M., Syrgkanis, V. 
# “EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation,”.
# Codes Available under “Notebooks” at: https://github.com/py-why/EconML

import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsRegressor
import graphviz as gr
import seaborn as sns
from causalinference import CausalModel
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_predict
import stargazer
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier


#change directory
os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\ECO2425\Term_Project")

#import cleaned dataset
AF_data = pd.read_csv("AF_data_clean.csv")

# Define a list of the multilateral resistance variables
mrterms = ["mrdis", "mrcon", "mrcom", "mrwto", "mrfta", "mreu"]

#Replicate Table 8, Figure 8

#############
# Double ML #
#############

# Using the Causal Inference for the Brave and True implementation


style.use("ggplot")
from toolz import curry

@curry
def elast(data, y, t):
    return (np.sum((data[t] - data[t].mean())*(data[y] - data[y].mean())) /
            np.sum((data[t] - data[t].mean())**2))
def cumulative_gain(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast(ordered_df.head(rows), y, t) * (rows/size) for rows in n_rows])

style.use("ggplot")

y = 'lbeim'
X = ['lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
     'mrcon', 'mrcom', 'mrwto', 'mrfta', 'mreu', 'mrdis'] + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995'] 
T_t = 'dk_ctax'
T_nt = 'dk_notax'
T = 'dk'

debias_m = LGBMRegressor(max_depth=5, random_state=2425)
denoise_m = LGBMRegressor(max_depth=5, random_state=2425)

# get the residuals for predictions of DK, DKT and DKNT
AF_data = AF_data.assign(T_resid=AF_data[T] -
                               cross_val_predict(
                                   debias_m, AF_data[X], AF_data[T], cv=4)
                               + AF_data[T].mean())

AF_data = AF_data.assign(T_t_resid=AF_data[T_t] -
                               cross_val_predict(
                                   debias_m, AF_data[X], AF_data[T_t], cv=4)
                               + AF_data[T_t].mean())

AF_data = AF_data.assign(T_nt_resid=AF_data[T_nt] -
                               cross_val_predict(
                                   debias_m, AF_data[X], AF_data[T_nt], cv=4)
                               + AF_data[T_nt].mean())


#Final Model (denoise for lbeim)
AF_data = AF_data.assign(lbeim_resid=AF_data[y] -
                               cross_val_predict(
                                   denoise_m, AF_data[X], AF_data[y], cv=4)
                               + AF_data[y].mean())

final_model = smf.ols(
    formula='lbeim_resid ~ T_resid', data=AF_data).fit()
final_model_i = smf.ols(
    formula='lbeim_resid ~ T_t_resid', data=AF_data).fit()
final_model_x = smf.ols(
    formula='lbeim_resid ~ T_nt_resid', data=AF_data).fit()


print(final_model_i.summary())
print(final_model_x.summary())
print(final_model.summary())


#########################
# EconML Implementation #
#########################
#EconML notebook implementation


#More flexible approach
###NOT USED IN THE MAIN TEXT. PERFORMANCE IS BAD

from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML
import numpy as np
from itertools import product
from sklearn.linear_model import (Lasso, MultiTaskElasticNetCV, ElasticNetCV)
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

y = AF_data['lbeim']
X = AF_data[['lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
     'mrcon', 'mrcom', 'mrwto', 'mrfta', 'mreu', 'mrdis'] + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]
T_i = AF_data['dk_ctax']
T_x = AF_data['dk_notax']
T = AF_data['dk']

est = DML(model_y=RandomForestRegressor(max_depth=3),
          model_t=RandomForestRegressor(max_depth=3),
          model_final=RandomForestRegressor(max_depth=3, n_estimators=100, min_samples_leaf=30),
          random_state=2425)


est.fit(y, T, X=X)
te_pred = est.effect(X)
np.mean(te_pred)
ate_sum = est.effect_inference(X).population_summary()
print(ate_sum)


est.fit(y, T_i, X=X)
te_pred = est.effect(X)
np.mean(te_pred)
ate_sum = est.effect_inference(X).population_summary()
print(ate_sum)


est.fit(y, T_x, X=X)
te_pred = est.effect(X)
np.mean(te_pred)
ate_sum = est.effect_inference(X).population_summary()
print(ate_sum)



#################
# Causal Forest #
#################
#EconML notebook implementation

from econml.grf import CausalForest, CausalIVForest, RegressionForest
from econml.dml import CausalForestDML
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

y = AF_data['lbeim']
X = AF_data[['lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
     'mrcon', 'mrcom', 'mrwto', 'mrfta', 'mreu', 'mrdis'] + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]
T_multiple = AF_data[['dk_ctax','dk_notax']]
T = AF_data['dk']

n_treatments = 1

#Causal forest from EconML
est = CausalForest(criterion='het', n_estimators=400, min_samples_leaf=5, max_depth=None,
                   min_var_fraction_leaf=None, min_var_leaf_on_val=True,
                   min_impurity_decrease = 0.0, max_samples=0.45, min_balancedness_tol=.45,
                   warm_start=False, inference=True, fit_intercept=True, subforest_size=4,
                   honest=True, verbose=0, n_jobs=-1, random_state=2425)
est.fit(X, T, y)

point, lb, ub = est.predict(X, interval=True, alpha=0.1)
print("ATE:")
print(np.mean(point[:,0]))
print(np.mean(lb[:,0]))
print(np.mean(ub[:,0]))

#Create a SHAP plot

#set variable names
X.columns = ['GDP Imp', 'GDP Exp', 'Distance', 'Contiguous', 'Common Language', 'Colony', 'Transition',  
     'MR Contiguous', 'MR Common Language', 'MR WTO', 'MR FTA', 'MR EU', 'MR Distance', 
     '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007']

# SHAP Plot FIGURE 8
import shap
explainer = shap.Explainer(est, shap.maskers.Independent(X, max_samples=100))
shap_values = explainer(X[:200])
shap.plots.beeswarm(shap_values)



AF_data['Forest_HTE'] = point

#same causal forest plot, multiple treatments (DKT and DKNT)
n_treatments = 2
est = CausalForest(criterion='het', n_estimators=400, min_samples_leaf=5, max_depth=None,
                   min_var_fraction_leaf=None, min_var_leaf_on_val=True,
                   min_impurity_decrease = 0.0, max_samples=0.45, min_balancedness_tol=.45,
                   warm_start=False, inference=True, fit_intercept=True, subforest_size=4,
                   honest=True, verbose=0, n_jobs=-1, random_state=2425)
est.fit(X, T_multiple, y)

point, lb, ub = est.predict(X, interval=True, alpha=0.01)

print("ATE TAX:")
print(np.mean(point[:,0]))
print("ATE NO TAX:")
print(np.mean(point[:,1]))


#########################################
# Country-Specific HTE (not used in the text)

Kyoto_countries = ['AUS', 'AUT', 'BEL', 'CAN', 'CHE', 'CZE', 'DEU', 'ESP',
                   'FRA', 'GBR', 'GRC', 'HUN', 'IRL', 'ITA', 'JPN', 'NLD',
                   'PRT', 'RUS', 'SVK', 'ROU', 'NZL', 'FIN', 'POL', 'NOR', 
                   'SWE', 'DNK', 'SVN', 'EST']

#get the country indicator and print the HTE for that country
#Note: For each country (AUS), this is the HTE conditional on AUS being either the importer OR exporter
# Restrict observations to 2002 when countries signed Kyoto
#actual heatmap is made in Excel
for country in Kyoto_countries:
    AF_data[country + "_ind"] = ((AF_data['ccode'] == country) | (AF_data['pcode'] == country)).astype(int) 
    print(country)
    print((AF_data["Forest_HTE"][(AF_data[country + "_ind"] == 1) & (AF_data['year'] >= 2003)]).mean())
