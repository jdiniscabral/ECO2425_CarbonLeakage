####################
# References:

#Code for main results adapted from replication package of:
# Aichele, R. and G. Felbermayr (2015): “Kyoto and Carbon Leakage: An Empirical Analysis of the Carbon Content of
# Bilateral Trade,” The Review of Economics and Statistics, 97, 104–115.
# Obtained from AEJ

#Code for LASSO and Tree-based methods based on Jupyter codes from:
# James, G. M., D. Witten, T. Hastie, R. Tibshirani, and J. Taylor (2023): 
# An Introduction to Statistical Learning: with Applications in Python, Springer.

#Code for DAG and Refutation Tests based on codes from:
# Molak, A. (2023): Causal Inference and Discovery in Python: Unlock the secrets of modern
# causal machine learning with DoWhy, EconML, PyTorch and more, Packt Publishing Ltd

import pandas as pd
import numpy as np
import os 


#change directory
os.chdir(r"")

#import the Aichele and Felbermayr replication data and the carbon tax policy data
#use _withEU version to include EU ETS in the tax policies
#use same file without "_withEU" suffix to produce the Table 6 results with ETS excluded
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

AF_data['dk_ctax'] = AF_data.apply(create_dummy, axis=1)

#same variable but treatment variable = 1 or -1 only if the Kyoto country does NOT have carbon tax policy in place

def create_dummy2(row):
    if row['ctax_c'] == 0 and row['ckyoto'] == 1 and row['pkyoto'] == 0:
        return 1
    elif row['ctax_p'] == 0 and row['pkyoto'] == 1 and row['ckyoto'] == 0:
        return -1
    else:
        return 0

AF_data['dk_notax'] = AF_data.apply(create_dummy2, axis=1)

# Define a list of the multilateral resistance variables
mrterms = ["mrdis", "mrcon", "mrcom", "mrwto", "mrfta", "mreu"]


##########################################
########## Run LASSO and Ridge ###########
##########################################

import sklearn.linear_model as skl
import sklearn.model_selection as skm
from matplotlib.pyplot import subplots
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#set up the variables
#exclude the treatment variable, add in additional variables that we had not considered previously
Y = AF_data[['lbeim']]
X_df = AF_data[['lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans', 'fta', 'wto', 'eu'] + mrterms] 
custom_labels = ['GDP M', 'GDP X', 'Distance', 'Contiguous', 'Language', 'Colony', 'Transition', 'FTA', 'WTO', 'EU',     
            'MR Distance', 'MR Contiguous', 'MR Language', 'MR WTO', 'MR FTA', 'MR EU'] 
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
                         columns=X_df.columns,
                         index=-np.log(lambdas))
soln_path.index.name = 'negative log(lambda)'

#Coefficients Plot
path_fig, ax = subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(custom_labels, loc='lower right');



#####RIDGE CV
ridge = skl.ElasticNet(alpha=lambdas[59], l1_ratio=0)
scaler = StandardScaler(with_mean=True,  with_std=True)
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
soln_path = pd.DataFrame(soln_array.T, columns=X_df.columns, index=-np.log(lambdas))
soln_path.index.name = 'negative log(lambda)'
soln_path

#coefficients figure
path_fig, ax = subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(custom_labels, loc='lower left');

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
coef_df = pd.DataFrame({'Feature': X_df.columns, 'Coefficient': lasso_coefficients})

# Filter to show only non-zero coefficients
selected_features = coef_df
print(selected_features)

############################

#Fit a LASSO Model with a specific lambda value 
#doing this to see what shrinks when we increase lambda a bit. Optimal lambda doesn't shrink anything
LASSO_Varselect = skl.Lasso(alpha = 0.06)
LASSO_Varselect.fit(Xs, Y)

#display the coefficients
Var_coefs = LASSO_Varselect.coef_
coef_varselect = pd.DataFrame({'Feature': X_df.columns, 'Coefficient': Var_coefs})
alpha_corres = pd.DataFrame({'negative log': -np.log(tuned_lasso.alphas_), 'alpha': tuned_lasso.alphas_})
