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


#change directory
os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\ECO2425\Term_Project")

#import cleaned dataset
AF_data = pd.read_csv("AF_data_clean.csv")

##########################################
########## Run LASSO and Ridge ###########
##########################################

#Replicate Figure 4 and shrinkage model tables from the appendix

#Based on ISLP codes

import sklearn.linear_model as skl
import sklearn.model_selection as skm
from matplotlib.pyplot import subplots
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


Y = AF_data[['lbeim']]
X = AF_data[['contig', 'comlang_ethno', 'colony', 'trans', 'fta', 'wto', 'eu']]



#first, add all of the interactions between variables to the dataset
#note: excluded joint EU status since it predicts non-treatment by definition
poly = PolynomialFeatures(interaction_only=True,include_bias = False)
X_df = poly.fit_transform(X)
feature_names_inter = poly.get_feature_names_out(input_features = ['contig', 'comlang_ethno', 'colony', 'trans', 'fta', 'wto', 'eu'])

#after gettung interaction,s add the variables we are not interacting
X_df = np.column_stack([X_df, np.array(AF_data[['lcy', 'lpy', 'ldist', 'mrcon', 'mrcom', 'mrfta', 'mrwto']])])

#set up the variables
#exclude the treatment variable, add in additional variables that we had not considered previously

labels_inter = feature_names_inter.tolist() + ['GDP M', 'GDP X', 'Distance', 'mrcon', 'mrcom', 'mrfta', 'mrwto'] 

###########################################################################

#make into numpy arrays
X = np.array(X_df)
Y = np.array(Y)

#standardize the coefficients
Xs = X - X.mean(0)[None,:]
X_scale = X.std(0)
Xs = Xs / X_scale[None,:]
lambdas = 10**np.linspace(0, -2, 200) / Y.std()

#set up the K parameter for k-fold cross-validation
#set K = 4 due to constraints on computing power
K = 4
kfold = skm.KFold(K, random_state=0, shuffle=True)

scaler = StandardScaler(with_mean=True,  with_std=True)
########################
### RIDGE REGRESSION ###
########################

#####RIDGE COEFFICIENTS (reproduces the Ridge plot from the appendix)
#run estimates for different estimates
#set L1 ratio = 0, meaning we use ridge regression
soln_array = skl.ElasticNet.path(Xs, Y, l1_ratio=0., alphas=lambdas)[1]
soln_array.shape

#reduce dimension
soln_array = np.squeeze(soln_array)

#set up the solution set
soln_path = pd.DataFrame(soln_array.T,
                         #columns=X_df.columns,
                         index=-np.log(lambdas))
soln_path.index.name = 'negative log(lambda)'

#Coefficients Plot
path_fig, ax = subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(labels_inter, loc='lower right');



#####RIDGE CV
ridge = skl.ElasticNet(alpha=lambdas[59], l1_ratio=0)
pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
pipe.fit(X, Y)

#cross validate for lambda
validation = skm.ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

#fit the model with the best lambda value
param_grid = {'ridge__alpha': lambdas}
grid = skm.GridSearchCV(pipe, param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid.fit(X, Y)
grid.best_params_['ridge__alpha']
grid.best_estimator_

#Cross-Validation plot (Not used in the text)
ridge_fig, ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(lambdas),
            -grid.cv_results_['mean_test_score'],
            yerr=grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20);

#############
### LASSO ###
#############
#Reproduces the LASSO figures from the main text

#####LASSO COEFFICIENTS
lambdas, soln_array = skl.Lasso.path(Xs, Y, l1_ratio=1, n_alphas=100)[:2]

#reduce dimension
soln_array = np.squeeze(soln_array)

#set up the solution set
soln_path = pd.DataFrame(soln_array.T, 
                         columns=labels_inter, 
                         index=-np.log(lambdas))
soln_path.index.name = 'negative log(lambda)'
soln_path

#coefficients figure
path_fig, ax = subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(labels_inter, loc='lower left');

#####LASSO CV
lassoCV = skl.ElasticNetCV(n_alphas=100,
                           l1_ratio=1,
                           cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler),
                         ('lasso', lassoCV)])
pipeCV.fit(Xs, Y)
tuned_lasso = pipeCV.named_steps['lasso']
tuned_lasso.alpha_

#Cross-Validation plot
lassoCV_fig, ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(tuned_lasso.alphas_),
            tuned_lasso.mse_path_.mean(1),
            yerr=tuned_lasso.mse_path_.std(1) / np.sqrt(K))
ax.axvline(-np.log(tuned_lasso.alpha_), c='k', ls='--')
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20);

# Get the coefficients from the tuned Lasso model
lasso_coefficients = tuned_lasso.coef_

# Create a DataFrame for easy visualization
coef_df = pd.DataFrame({'Feature': labels_inter, 'Coefficient': lasso_coefficients})

# Filter to show only non-zero coefficients
selected_features = coef_df
print(selected_features)

############################
print(-np.log(0.018))
#Fit a LASSO Model with a specific lambda value 
#doing this to see what shrinks when we increase lambda a bit. Optimal lambda doesn't shrink anything
LASSO_Varselect = skl.Lasso(alpha = 0.018)
LASSO_Varselect.fit(Xs, Y)

#display the coefficients
Var_coefs = LASSO_Varselect.coef_
coef_varselect = pd.DataFrame({'Feature': labels_inter, 'Coefficient': Var_coefs})
alpha_corres = pd.DataFrame({'negative log': -np.log(tuned_lasso.alphas_), 'alpha': tuned_lasso.alphas_})
print(coef_varselect)
