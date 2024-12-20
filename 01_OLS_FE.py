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


#change directory
os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\ECO2425\Term_Project")

#import cleaned dataset
AF_data = pd.read_csv("AF_data_clean.csv")

# Define a list of the multilateral resistance variables
mrterms = ["mrdis", "mrcon", "mrcom", "mrwto", "mrfta", "mreu"]


#####################################
# Plots for data and identification #
#####################################
#Replicate Figure 2

#NOTE: Using BEIM  as oppose to the log version (lbeim) we use in the paper as the dependent variable
#NOTE: This takes a long time to run. Comment it out for regresionn  tables

import matplotlib.pyplot as plt
import seaborn as sns

AF_data_plot = AF_data

#identify country-pairings where dk was equal to 1 or -1 at some point
#create the three flags to identify the countries belonging to each group
AF_data_plot['dk_flag'] = AF_data_plot['sid'].apply(
    lambda x: 'Treated (DK=1)' if (AF_data_plot[AF_data_plot['sid'] == x]['dk'] == 1).any() 
    else ('Treated (DK=-1)' if (AF_data_plot[AF_data_plot['sid'] == x]['dk'] == -1).any() 
    else 'Non-treated (DK=0)')
)

# Create the average by year and dk_flag
average_by_year = AF_data_plot.groupby(['year', 'dk_flag'])['BEIM'].mean().reset_index()


# Create the plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=average_by_year, x='year', y='BEIM', hue='dk_flag', marker='o', estimator=None)
plt.axvline(x=2002, linestyle='--', color='red', label='Beginning of Kyoto Ratification')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Average Bilateral Emissions Embodied in Imports (Tones)')
plt.xticks(rotation=45)
plt.legend(title='DK Treatment Status')
plt.grid()
plt.show()

############################################################
# Emissions Trends for Carbon Pricing vs no Carbon Pricing #
############################################################

#Replicate Figure 1

#first, subset the data to only have treated observations with DK = 1
#next, identify obsrevations with tax and observations with no tax
AF_data_plot = AF_data[AF_data['dk']==1]
AF_data_plot['dk_flag'] = AF_data_plot['sid'].apply(
    lambda x: 'With Carbon Pricing (DK = 1, T = 1)' if (AF_data_plot[AF_data_plot['sid'] == x]['dk_ctax'] == 1).any() 
    else ('No Carbon Pricing (DK = 1, T = 0)'))

# Create the average by year and dk_flag
average_by_year = AF_data_plot.groupby(['year', 'dk_flag'])['BEIM'].mean().reset_index()


# Create the plot
sns.set(style="whitegrid")
plt.figure(figsize=(11, 7))
sns.lineplot(data=average_by_year, x='year', y='BEIM', hue='dk_flag', marker='o', estimator=None)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Average Bilateral Emissions Embodied in Imports (Tones)')
plt.xticks()
plt.legend(title='DK Treatment Status')
plt.grid()
plt.show()


##################################
# Regression Results for Table 4 #
##################################

###############################
import statsmodels.formula.api as smf
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from stargazer.stargazer import Stargazer

#FOR Y VARIABLE
#lbeim for carbon embodied in exports
#limp for imports
#lint for intensity 

#to store resuls and create Stargazer tables
results = []

################
### BASELINE ###
################

#####
###Kyoto Treatment
Y = AF_data[['lbeim']]
X_obs = AF_data[['dk', 'ldist', 'lcy', 'lpy', 'contig', 'comlang_ethno', 'fta', 'wto', 'eu'] \
                + mrterms + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995'] \
            + [col for col in AF_data.columns if col.startswith('iii') and col != 'iii__1']]

# add a constant: note that one time fixed effect has been removed to avoid dummy variable trap
X_obs = sm.add_constant(X_obs)

# Fit the model
model = sm.OLS(Y, X_obs)
results_baseline_dk = model.fit(cov_type='HC1')

#print the output. dk is variable of interest
print(results_baseline_dk.summary())
results.append(results_baseline_dk)


#####
###Carbon Pricing Treatment
X_obs = AF_data[['dk_ctax', 'dk_notax', 'ldist', 'lcy', 'lpy', 'contig', 'comlang_ethno', 'fta', 'wto', 'eu'] \
                + mrterms + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']\
            + [col for col in AF_data.columns if col.startswith('iii') and col != 'iii__1']]

X_obs = sm.add_constant(X_obs)

model = sm.OLS(Y, X_obs)
results_baseline_tax = model.fit(cov_type='HC1')

#print the output. dctax_K is variable of interest
print(results_baseline_tax.summary())
results.append(results_baseline_tax)




##########################
### ML VARIABLE SELECT ###
##########################
#####
###Kyoto Treatment

#FE for countries 
AF_data['importer_id'] = pd.factorize(AF_data.ccode)[0]
AF_data['exporter_id'] = pd.factorize(AF_data.pcode)[0]

#prepare the data by setting up the time and panel dimensions
AF_data_FE = AF_data
AF_data_FE['year_int'] = AF_data_FE['year'].astype(int)
AF_data_FE.set_index(['importer_id', 'exporter_id'], inplace=True)

Y = AF_data['lbeim']
X = AF_data[['dk', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] \
            + [col for col in AF_data.columns if col.startswith('iii') and col != 'iii__1']
            + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]
X = sm.add_constant(X)


# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model_varsel = PanelOLS(Y, X, entity_effects=True, time_effects=True)
results_varsel_dk = model_varsel.fit(cov_type='clustered')  

#print the output. dk is variable of interest
print(results_varsel_dk.summary)
results.append(results_varsel_dk)

#####
###Carbon Pricing Treatment
X = AF_data[['dk_ctax', 'dk_notax', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] 
            + [col for col in AF_data.columns if col.startswith('iii') and col != 'iii__1']
            + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]
X = sm.add_constant(X)

model_varsel_tax = PanelOLS(Y, X, entity_effects=True, time_effects=True)
results_varsel_tax = model_varsel_tax.fit(cov_type='clustered') 

#print the output. dctax_K is variable of interest
print(results_varsel_tax.summary)
results.append(results_varsel_tax)








#####################
### FIXED EFFECTS ###
#####################
    

#prepare the data by setting up the time and panel dimensions
AF_data_FE = AF_data
AF_data_FE['year_int'] = AF_data_FE['year'].astype(int)
AF_data_FE.set_index(['sid', 'year_int'], inplace=True)

#####
###Kyoto Treatment
Y = AF_data_FE['lbeim']
X = AF_data_FE[['dk', 'eu', 'fta', 'wto']]
X = sm.add_constant(X)


# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = PanelOLS(Y, X, entity_effects=True, time_effects=True)
results_mainfe_dk = model.fit(cov_type='clustered')  

#print the output. dk is variable of interest
print(results_mainfe_dk.summary)
results.append(results_mainfe_dk)

#####
###Carbon Pricing Treatment
X = AF_data_FE[['dk_ctax', 'dk_notax', 'eu', 'fta', 'wto']]

X = sm.add_constant(X)

model = PanelOLS(Y, X, entity_effects=True, time_effects=True)
results_mainfe_tax = model.fit(cov_type='clustered') 

#print the output. dctax_K is variable of interest
print(results_mainfe_tax.summary)
results.append(results_mainfe_tax)



############################
# Create the Output Tables #
############################

#This replicates Table 4

stargazer = Stargazer([results_baseline_dk, results_baseline_tax, 
                       results_varsel_dk, results_varsel_tax,
                       results_mainfe_dk, results_mainfe_tax])

# Customize the output
stargazer.title('Regression Results for Fixed Effects Model')
stargazer.rename_covariates({'dk': '$DK_{mxt}$', 'dk_ctax': '$DKT_{mxt}$',
                             'dk_notax': '$DKNT_{mxt}$',
                             'ldist': '$Dist_{mx}$', 'lcy': '$GDP_{xt}$',
                              'lpy': '$GDP_{mt}$', 'contig': '$contig_{mx}$', 
                              'comlang_ethno': '$comlang_{mx}$',
                              'fta': '$FTA_{mxt}$', 'wto': '$WTO_{mxt}$',
                              'eu': '$EU_{mxt}$', 'trans': '$Transition_{mxt}$'})

# Report only the treatment variables
to_report = ['dk', 'dk_ctax', 'dk_notax'] 
stargazer.covariate_order(to_report)

# Output the results to LaTeX
latex_output = stargazer.render_latex()
print(latex_output)

#########################################################
# NOT USED IN THE MAIN PAPER - SUBCOMPONENT TABLES

########################
# Subcomponent Results #
########################
#Use the fixed effects specification but change the dependent variables to the two subcomponents

results2 = []

#############Import Volume

#####
###Kyoto Treatment 
Y = AF_data['limp']
X = AF_data[['dk', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] 
            + [col for col in AF_data.columns if col.startswith('iii') and col != 'iii__1']
            + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X = sm.add_constant(X)

# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = sm.OLS(Y, X)
results_fe_dk = model.fit(cov_type='HC1')  

#print the output. dk is variable of interest
print(results_fe_dk.summary())
results2.append(results_fe_dk)

#####
###Carbon Pricing Treatment 
Y = AF_data['limp']
X = AF_data[['dk_ctax', 'dk_notax', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] 
            + [col for col in AF_data.columns if col.startswith('iii') and col != 'iii__1']
            + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X = sm.add_constant(X)

# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = sm.OLS(Y, X)
results_fe_dk = model.fit(cov_type='HC1') 

#print the output. dk is variable of interest
print(results_fe_dk.summary())
results2.append(results_fe_dk)

#############Emissions Intensity

#####
###Kyoto Treatment 
Y = AF_data['lint']
X = AF_data[['dk', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] 
            + [col for col in AF_data.columns if col.startswith('iii') and col != 'iii__1']
            + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X = sm.add_constant(X)

# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = sm.OLS(Y, X)
results_fe_dk = model.fit(cov_type='HC1') 

#print the output. dk is variable of interest
print(results_fe_dk.summary())
results2.append(results_fe_dk)

#####
###Carbon Pricing Treatment 
Y = AF_data['lint']
X = AF_data[['dk_ctax', 'dk_notax', 'lcy', 'lpy', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] 
            + [col for col in AF_data.columns if col.startswith('iii') and col != 'iii__1']
            + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X = sm.add_constant(X)

# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = sm.OLS(Y, X)
results_fe_dk = model.fit(cov_type='HC1')  # Clustered standard errors

#print the output. dk is variable of interest
print(results_fe_dk.summary())
results2.append(results_fe_dk)

#########
#Create the output Tables
#########

stargazer = Stargazer(results2)

to_report = ['dk', 'dk_ctax', 'dk_notax'] 
stargazer.covariate_order(to_report)

# Output the results to LaTeX
latex_output = stargazer.render_latex()
print(latex_output)

