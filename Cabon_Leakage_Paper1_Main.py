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
#os.chdir(r"")

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

###############################
# Generate Summary Statistics #
###############################

AF_data_sum = AF_data[['lbeim', 'lcy', 'trans', 'wto', 'fta', 'eu', 'dk', 'dctax_K']]
summary_stats = AF_data_sum.describe()

# Add variable labels
variable_labels = {
    'lbeim': 'Emiss. in Imports',
    'lcy': 'Log GDP Per Capita',
    'trans': 'Transition',
    'fta': 'FTA',
    'eu': 'EU',
    'wto': 'WTO',
    'dk': 'Kyoto',
    'dctax_K': 'Kyoto (CPrice)'
}

# Rename the columns of summary_stats
summary_stats.columns = [variable_labels[col] for col in summary_stats.columns]
summary_stats.index = ['N', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']

latex_output = summary_stats.to_latex(float_format="%.2f", 
                                       label='tab:summary_stats',
                                       caption='Summary Statistics for Selected Variables')
# Print the LaTeX output
#print(latex_output)
del(AF_data_sum)

###############################
import statsmodels.formula.api as smf
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

#This code reproduces the main regression results in Table 3

#FOR Y VARIABLE
#lbeim for carbon embodied in exports
#limp for imports
#lint for intensity 

################
### BASELINE ###
################

#####
###Kyoto Treatment with full sample
Y = AF_data[['lbeim']]
X_obs = AF_data[['dk', 'ldist', 'lcy', 'lpy', 'contig', 'comlang_ethno', 'fta', 'wto', 'eu'] 
                + mrterms + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

# add a constant: note that one time fixed effect has been removed to avoid dummy variable trap
X_obs = sm.add_constant(X_obs)

# Fit the model
model = sm.OLS(Y, X_obs)
results_obs = model.fit(cov_type='HC1')

#print the output. dk is variable of interest
print(results_obs.summary())

#####
###Carbon Pricing Treatment with full sample
X_obs = AF_data[['dctax_K', 'ldist', 'lcy', 'lpy', 'contig', 'comlang_ethno', 'fta', 'wto', 'eu'] 
                + mrterms + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X_obs = sm.add_constant(X_obs)

model = sm.OLS(Y, X_obs)
results_obs = model.fit(cov_type='HC1')

#print the output. dctax_K is variable of interest
print(results_obs.summary())

#####
###Kyoto Treatment with carbon tax countries removed

Y = AF_data_notax[['lbeim']]
X_obs = AF_data_notax[['dk', 'ldist', 'lcy', 'lpy', 'contig', 'comlang_ethno', 'fta', 'wto', 'eu'] 
                + mrterms + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X_obs = sm.add_constant(X_obs)

model_notax = sm.OLS(Y, X_obs)
results_obs = model_notax.fit(cov_type='HC1')

#print the output. dk is the outcome variable of interest
print(results_obs.summary())

##########################
### ML VARIABLE SELECT ###
##########################

#####
###Kyoto Treatment with full sample
Y = AF_data[['lbeim']]
X_varsel = AF_data[['dk', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] + 
               [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X_varsel = sm.add_constant(X_varsel)

model_varsel = sm.OLS(Y, X_varsel)
results_varsel = model_varsel.fit(cov_type='HC1')

#print the output. dk is variable of interest
print(results_varsel.summary())

#####
###Carbon Pricing Treatment with full sample
X_varsel = AF_data[['dctax_K', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] + 
               [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X_varsel = sm.add_constant(X_varsel)

model_varsel = sm.OLS(Y, X_varsel)
results_varsel = model_varsel.fit(cov_type='HC1')

#print the output. dctax_K is variable of interest
print(results_varsel.summary())

#####
###Kyoto Treatment with carbon tax countries removed

Y = AF_data_notax[['lbeim']]
X_varsel_notax = AF_data_notax[['dk', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] + 
               [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X_varsel_notax = sm.add_constant(X_varsel_notax)

# Fit the model with observables
model_varsel_notax = sm.OLS(Y, X_varsel_notax)
results_varsel_notax = model_varsel_notax.fit(cov_type='HC1')

#print the output. dk is variable of interest
print(results_varsel_notax.summary())

#####################
### FIXED EFFECTS ###
#####################

#prepare the data by setting up the time and panel dimensions
AF_data_FE = AF_data
AF_data_FE['year_int'] = AF_data_FE['year'].astype(int)
AF_data_FE.set_index(['year_int', 'sid'], inplace=True)

#####
###Kyoto Treatment with full sample
Y = AF_data_FE['lbeim']
X = AF_data_FE[['dk', 'lcy', 'lpy', 'fta', 'wto', 'eu'] + mrterms]

X = sm.add_constant(X)

# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = PanelOLS(Y, X, entity_effects=True)
results = model.fit(cov_type='clustered', cluster_time=True)  # Clustered standard errors

#print the output. dk is variable of interest
print(results.summary)

#####
###Carbon Pricing Treatment with full sample
X = AF_data_FE[['dctax_K',  'lcy', 'lpy', 'fta', 'wto', 'eu'] + mrterms]
X = sm.add_constant(X)

model = PanelOLS(Y, X, entity_effects=True)
results = model.fit(cov_type='clustered', cluster_time=True)  # Clustered standard errors

#print the output. dctax_K is variable of interest
print(results.summary)

#####
###Kyoto Treatment with carbon tax countries removed

#prepare the data and set the time and panel dimensions
AF_data_notax_FE = AF_data_notax
AF_data_notax_FE['year'] = AF_data_notax_FE['year'].astype(int)
AF_data_notax_FE.set_index(['year', 'sid'], inplace=True)

Y = AF_data_notax_FE['lbeim']
X = AF_data_notax_FE[['dk', 'lcy', 'lpy', 'fta', 'wto', 'eu'] + mrterms]

X = sm.add_constant(X)

model_notax = PanelOLS(Y, X, entity_effects=False)
results_notax = model_notax.fit(cov_type='clustered', cluster_time=True)  # Clustered standard errors

#print the output. dk is variable of interest
print(results_notax.summary)

