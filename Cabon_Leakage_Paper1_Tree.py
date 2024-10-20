####################
# References:

#Code for main results adapted from replication package of:
# Aichele, R. and G. Felbermayr (2015): “Kyoto and Carbon Leakage: An Empirical Analysis of the Carbon Content of
# Bilateral Trade,” The Review of Economics and Statistics, 97, 104–115.
# Obtained from AEJ

#Code for LASSO and Tree-based methods based on Jupyter codes from:
# James, G. M., D. Witten, T. Hastie, R. Tibshirani, and J. Taylor (2023): 
# An Introduction to Statistical Learning: with Applications in Python, Springer.

#Code for DAG and Refutation Tests from
# Molak, A. (2023): Causal Inference and Discovery in Python: Unlock the secrets of modern
# causal machine learning with DoWhy, EconML, PyTorch and more, Packt Publishing Ltd

import pandas as pd
import numpy as np
import os 


#change directory
os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\ECO2425\Term_Project\Replication")

#import the Aichele and Felbermayr replication data and the carbon tax policy data
#use _withEU version to include EU ETS in the tax policies
AF_data = pd.read_stata("data\data\data_kyotoandleakage_restat.dta")
CTAX_data = pd.read_excel(f"..\carbon_tax_countries_withEU.xlsx")

#merge carbon tax countries with AF_data and create the indicator variable
#must be done once for importers and once for exporters
merged_data = pd.merge(AF_data, CTAX_data[['ccode', 'year', 'ctax']], on=['year', 'ccode'], how='left')
merged_data['ctax'].fillna(0, inplace=True)
merged_data.rename(columns={'ctax': 'ctax_c'}, inplace=True)
AF_data = merged_data

merged_data = pd.merge(AF_data, CTAX_data[['pcode', 'year', 'ctax']], on=['year', 'pcode'], how='left')
merged_data['ctax'].fillna(0, inplace=True)
merged_data.rename(columns={'ctax': 'ctax_p'}, inplace=True)
AF_data = merged_data

######################################
##### Data cleaning/manipulation #####
######################################

#create carbon tax status minus Kyoto status treatment variable
#= 1 if importer has a carbon tax, exporter has not signed Kyoto
#= 0 if importer/exporter have same carbon tax status
#= -1 if exporter has a carbon tax, importer has not signed Kyoto
def create_dummy(row):
    if row['ctax_c'] == 1 and row['ckyoto'] == 1 and row['pkyoto'] == 0:
        return 1
    elif row['ctax_p'] == 1 and row['pkyoto'] == 1 and row['ckyoto'] == 0:
        return -1
    else:
        return 0

AF_data['dctax_K'] = AF_data.apply(create_dummy, axis=1)


# Define a list of the multilateral resistance variables
mrterms = ["mrdis", "mrcon", "mrcom", "mrwto", "mrfta", "mreu"]

#create a subset of the data without the carbon tax countries
names_tax = ["FIN", "POL", "NOR", "SWE", "DNK", "SVN", "EST"]
AF_data_notax = AF_data[-AF_data['ccode'].isin(names_tax)]

#############################################
########## Run Tree-Based Methods ###########
#############################################

from matplotlib.pyplot import subplots
import sklearn.model_selection as skm
import math
from sklearn.tree import (DecisionTreeRegressor as DTR, plot_tree)
from sklearn.ensemble import (RandomForestRegressor as RF, GradientBoostingRegressor as GBR)

#to produce regression trees from the appendix, make the following modifications to Y_df:
#limp for imports
#lint for intensity

#set up the X and Y matrices with the full set of variables we consider for variables selection
Y_df = AF_data['lbeim']
X_df = AF_data[['dk', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans', 'fta', 'wto', 'eu'] + mrterms] 

feature_names = list(['Difference in Kyoto', 'Importer GDP', 'Exporter GDP', 'Distance', 
                      'Contiguous', 'Colony', 'Transition Country', 'Language', 'FTA', 'WTO', 'EU', "MR Distance", 
                      "MR Contiguous", "MR Language", "MR WTO", "MR FTA", "MR EU"])

#get X and Y as arrays
X = np.array(X_df, dtype=np.float32)
Y = np.array(Y_df, dtype=np.float32)

#####
###Simple Regression Tree
#####

#split the data into test and train 20% test data
(X_train, X_test, y_train, y_test) = skm.train_test_split(X, Y, test_size=0.2, random_state=0)

#plot the regression tree estimated using the test data
reg = DTR(max_depth=3)
reg.fit(X_train, y_train)
ax = subplots(figsize=(12,12))[1]
plot_tree(reg,feature_names=feature_names, ax=ax);                                
 
#make a grid of cost complexity pruning alpha values
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
ax = subplots(figsize=(20,20))[1]
plot = plot_tree(G.best_estimator_, feature_names=feature_names, ax=ax);

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
