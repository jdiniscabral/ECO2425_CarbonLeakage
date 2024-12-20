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

#############################################
########## Run Tree-Based Methods ###########
#############################################

#Based on ISLP Codes

#Replicate Figure 5, Table 3 and tree-based methods form appendix

from matplotlib.pyplot import subplots
import sklearn.model_selection as skm
import math
from sklearn.tree import (DecisionTreeRegressor as DTR, plot_tree)
from sklearn.ensemble import (RandomForestRegressor as RF, GradientBoostingRegressor as GBR)

#to produce regression trees from the appendix, make the following modifications to Y_df:
#limp for imports
#lint for intensity

#set up the X and Y matrices with the full set of variables we consider for variables selection
Y_df = AF_data[['lbeim']]
X_df = AF_data[['lcy', 'lpy', 'contig', 'comlang_ethno', 'colony', 'trans', 'fta', 'wto', 
                 'mrcon', 'mrcom', 'mrwto', 'mrfta']]
feature_names = ['GDP M', 'GDP X', 'Contiguous', 'Language', 'Colony', 'Transition', 'FTA', 'WTO',     
             'MR Contiguous', 'MR Language', 'MR WTO', 'MR FTA'] 

#get X and Y as arrays
X = np.array(X_df, dtype=np.float32)
Y = np.array(Y_df, dtype=np.float32)

#####
###Simple Regression Tree
#####
#Run with lbeim as dependent variable for Regression Tree in the Main Text
# Run with lint (trade) or lint (intensity) to produce the appendix figures

#split the data into test and train 20% test data
(X_train, X_test, y_train, y_test) = skm.train_test_split(X, Y, test_size=0.2, random_state=0)

#plot the regression tree estimated using the test data
reg = DTR(max_depth=3)
reg.fit(X_train, y_train)
ax = subplots(figsize=(12,12))[1]
plot_tree(reg,feature_names=feature_names, ax=ax);                                
 
#make a grid of alpha values for pruning
#use 5-fold cross-validation to get the optimal alpha                                
ccp_path = reg.cost_complexity_pruning_path(X_train, y_train)
kfold = skm.KFold(5,
                  shuffle=True,
                  random_state=10)
grid = skm.GridSearchCV(reg,
                        {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
G = grid.fit(X_train, y_train)

#using the best alpha, get the MSE on the test data
best_ = grid.best_estimator_
errors = (np.float16(y_test) - np.float16(best_.predict(X_test)))
sq_errors = errors**2
MSE_regtree = np.mean(sq_errors)
RMSE_regtree = math.sqrt(MSE_regtree)

#plot the tree on the test data using the best alpha
ax = subplots(figsize=(20,16))[1]
plot = plot_tree(best_, feature_names=feature_names, ax=ax, impurity=False, 
                 precision=2, fontsize=15, rounded=True)

#####
###Bagging 
#####

#fit bagging. Random forest, but the max features includes everything in the model
bagging_grav = RF(max_features=X_train.shape[1], n_estimators=100, random_state=0).fit(X_train, y_train)
y_hat_bag = np.float32(bagging_grav.predict(X_test))
errors = (np.float16(y_test) - np.float16(y_hat_bag))
sq_errors = errors**2
MSE_bagging = np.mean(sq_errors)
RMSE_bagging = math.sqrt(MSE_bagging)

#####
###Random Forest
#####

#pick how many features to choose from
m = round((X_train.shape[1])**(1/2))

#random forest with m = sqrt(p) features
RF_grav = RF(max_features=m, n_estimators=100,random_state=0).fit(X_train, y_train)
y_hat_RF = RF_grav.predict(X_test)
errors = (np.float16(y_test) - np.float16(y_hat_RF))
sq_errors = errors**2
MSE_RF = np.mean(sq_errors)
RMSE_RF = math.sqrt(MSE_RF)

#####
###Boosting
#####

#Use 0.001 for learning rate
#estimate 1000 trees with max depth 3
boost_grav = GBR(n_estimators=1000,
                   learning_rate=0.001, #lambda term
                   max_depth=3,
                   random_state=0)
boost_grav.fit(X_train, y_train)

kfold = skm.KFold(5,
                  shuffle=True,
                  random_state=10)

y_hat_boost = boost_grav.predict(X_test);
errors = (np.float16(y_test) - np.float16(y_hat_boost))
sq_errors = errors**2
MSE_boost = np.mean(sq_errors)
RMSE_boost = math.sqrt(MSE_boost)

#produce the feature importance matrix
feature_imp = pd.DataFrame(
    {'importance':RF_grav.feature_importances_},
    index=feature_names)
feature_imp = feature_imp.sort_values(by='importance', ascending=False)

latex_table = feature_imp.to_latex(index=True)
print(latex_table)


#Report MSE for each method
#added to appendix table
MSE_regtree
RMSE_regtree

MSE_bagging
RMSE_bagging

MSE_RF
RMSE_RF

MSE_boost
RMSE_boost
