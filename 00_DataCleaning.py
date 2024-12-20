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
AF_data['dk_ctax'] = np.where( ( ( (AF_data['ctax_c'] == 1) & (AF_data['ckyoto'] == 1) & (AF_data['pkyoto'] == 0) ) |
                                ( (AF_data['ctax_p'] == 1) & (AF_data['pkyoto'] == 1) & (AF_data['ckyoto'] == 0) )
                                ), AF_data['dk'], 0)

AF_data['dk_notax'] = np.where( ( ( (AF_data['ctax_c'] == 0) & (AF_data['ckyoto'] == 1) & (AF_data['pkyoto'] == 0) ) |
                                ( (AF_data['ctax_p'] == 0) & (AF_data['pkyoto'] == 1) & (AF_data['ckyoto'] == 0) )
                                ), AF_data['dk'], 0)

#SANITY CHECK:
#check to make sure that the dk_ctax and dk_notax variables sum up to the dk variable
#difference between original dk from A and F and sum of tax and notax sholud be 0 (check min and max are zero)
AF_data['dk_test'] = AF_data['dk_ctax'] + AF_data['dk_notax']
AF_data['dk_test_'] = AF_data['dk'] - AF_data['dk_test']
print(max(AF_data['dk_test_']))
print(min(AF_data['dk_test_']))

#create separated treatment variables
AF_data['dk_1'] = np.where(AF_data['dk'] == 1, 1, 0)
AF_data['dk_neg1'] = np.where(AF_data['dk'] == -1, 1, 0)

#separate treatments for carbon tax and no carbon tax
#might not need these
#carbon tax treatment
AF_data['dkt_1'] = np.where(AF_data['dk_ctax'] == 1, 1, 0)
AF_data['dkt_neg1'] = np.where(AF_data['dk_ctax'] == -1, 1, 0)

#no carbon tax treatment
AF_data['dknt_1'] = np.where(AF_data['dk_notax'] == 1, 1, 0)
AF_data['dknt_neg1'] = np.where(AF_data['dk_notax'] == -1, 1, 0)

#take out the country year fixed effects in their dataset. Not sure why they are there
AF_data = AF_data.loc[:, ~AF_data.columns.str.startswith('nnn')]

#create some stuff to use for fixed effects
#country pairing id
AF_data['countrypairing_id'] = pd.factorize(AF_data.ccode+AF_data.pcode)[0]

#country pairing year FE
AF_data['year_str'] = AF_data['year'].astype(str)
AF_data['countrypairing_year'] = pd.factorize(AF_data.ccode+AF_data.pcode+AF_data.year_str)[0]

#industry fixed effect
AF_data = pd.get_dummies(AF_data, columns=['category'], prefix='iii_', dtype = int)

###############################
# Generate Summary Statistics #
###############################

#list of Kyoto 
Kyoto_countries = ['AUS', 'AUT', 'BEL', 'CAN', 'CHE', 'CZE', 'DEU', 'ESP',
                   'FRA', 'GBR', 'GRC', 'HUN', 'IRL', 'ITA', 'JPN', 'NLD',
                   'PRT', 'RUS', 'SVK', 'ROU', 'NZL', 'FIN', 'POL', 'NOR', 
                   'SWE', 'DNK', 'SVN', 'EST']

#full sample
AF_data_sum = AF_data[['lbeim', 'lcy', 'wto', 'fta', 'eu', 'dk', 'ctax_c']]
summary_stats = AF_data_sum.describe()

AF_data_sum_kyoto = AF_data[(AF_data['ccode'].isin(Kyoto_countries)) & (AF_data['pcode'].isin(Kyoto_countries))][['lbeim', 'lcy', 'trans', 'wto', 'fta', 'eu', 'dk']]
summary_stats = AF_data_sum_kyoto.describe()

AF_data_sum_nonkyoto = AF_data[(~AF_data['ccode'].isin(Kyoto_countries)) & (~AF_data['pcode'].isin(Kyoto_countries))][['lbeim', 'lcy', 'trans', 'wto', 'fta', 'eu', 'dk']]
summary_stats = AF_data_sum_nonkyoto.describe()

AF_data_sum_nonkyoto = AF_data[(AF_data['dk'] > 0) | (AF_data['dk'] < 0)][['lbeim', 'lcy', 'trans', 'wto', 'fta', 'eu', 'dk']]
summary_stats = AF_data_sum_nonkyoto.describe()

# Add variable labels
variable_labels = {
    'lbeim': '$E_{mxst}$',
    'lcy': '$GDP_{mxt}$',
    'trans': '$Trans_{mt}$',
    'fta': '$FTA_{mxt}$',
    'eu': '$EU_{mxt}$',
    'wto': '$WTO_{mxt}$',
    'dk': '$DK_{mxt}$',
}

# Rename the columns of summary_stats
summary_stats.columns = [variable_labels[col] for col in summary_stats.columns]
summary_stats = summary_stats.loc[['count', 'mean', 'std', 'min', 'max']]
summary_stats.index = ['N', 'Mean', 'Std Dev', 'Min', 'Max']
summary_stats = summary_stats.T

latex_output = summary_stats.to_latex(float_format="%.2f", 
                                       label='tab:summary_stats',
                                       caption='Summary Statistics for Selected Variables')
# Print the LaTeX output
print(latex_output)
del(AF_data_sum)


#export the cleaned dataset for other codes
AF_data.to_csv('../AF_data_clean.csv', index=False)
