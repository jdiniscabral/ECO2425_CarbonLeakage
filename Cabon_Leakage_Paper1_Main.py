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
CTAX_data = pd.read_excel(f"..\carbon_tax_countries.xlsx")

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


###############################
# Generate Summary Statistics #
###############################

#subset of the data for summary statistics
AF_data_sum = AF_data[['lbeim', 'lcy', 'wto', 'fta', 'eu', 'dk', 'ctax_p']]
summary_stats = AF_data_sum.describe()

# Add variable labels
variable_labels = {
    'lbeim': 'Emiss. in Imports',
    'lcy': 'Log GDP Per Capita',
    'fta': 'FTA',
    'eu': 'EU',
    'wto': 'WTO',
    'dk': 'Kyoto',
    'ctax_p': 'Carbon Pricing',
}

# Rename the columns of summary_stats
summary_stats.columns = [variable_labels[col] for col in summary_stats.columns]
summary_stats.index = ['N', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']

latex_output = summary_stats.to_latex(float_format="%.2f", 
                                       label='tab:summary_stats',
                                       caption='Summary Statistics for Selected Variables')
# Print the LaTeX output
print(latex_output)
del(AF_data_sum)

#####################################
# Plots for data and identification #
#####################################
#NOTE: Using BEIM  as oppose to the log version (lbeim) we use in the paper as the dependent variable
#NOTE: This takes a long time to run

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
X_obs = AF_data[['dk', 'ldist', 'lcy', 'lpy', 'contig', 'comlang_ethno', 'fta', 'wto', 'eu'] 
                + mrterms + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

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
X_obs = AF_data[['dk_ctax', 'dk_notax', 'ldist', 'lcy', 'lpy', 'contig', 'comlang_ethno', 'fta', 'wto', 'eu'] 
                + mrterms + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

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
Y = AF_data[['lbeim']]
X_varsel = AF_data[['dk', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] + 
               [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X_varsel = sm.add_constant(X_varsel)

model_varsel = sm.OLS(Y, X_varsel)
results_varsel_dk = model_varsel.fit(cov_type='HC1')

#print the output. dk is variable of interest
print(results_varsel_dk.summary())
results.append(results_varsel_dk)

#####
###Carbon Pricing Treatment
X_varsel = AF_data[['dk_ctax', 'dk_notax', 'lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  
                "mrcon", "mrcom", "mrwto", "mrfta", "mreu", "mrdis"] + 
               [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']]

X_varsel = sm.add_constant(X_varsel)

model_varsel = sm.OLS(Y, X_varsel)
results_varsel_tax = model_varsel.fit(cov_type='HC1')

#print the output. dctax_K is variable of interest
print(results_varsel_tax.summary())
results.append(results_varsel_tax)


#####################
### FIXED EFFECTS ###
#####################

#prepare the data by setting up the time and panel dimensions
AF_data_FE = AF_data
AF_data_FE['year_int'] = AF_data_FE['year'].astype(int)
AF_data_FE.set_index(['year_int', 'sid'], inplace=True)

#####
###Kyoto Treatment
Y = AF_data_FE['lbeim']
X = AF_data_FE[['dk', 'lcy', 'lpy', 'fta', 'wto', 'eu'] + mrterms]

X = sm.add_constant(X)

# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = PanelOLS(Y, X, entity_effects=True)
results_fe_dk = model.fit(cov_type='clustered')  

#print the output. dk is variable of interest
print(results_fe_dk.summary)
results.append(results_fe_dk)

#####
###Carbon Pricing Treatment
X = AF_data_FE[['dk_ctax', 'dk_notax',  'lcy', 'lpy', 'fta', 'wto', 'eu'] + mrterms]
X = sm.add_constant(X)

model = PanelOLS(Y, X, entity_effects=True)
results_fe_tax = model.fit(cov_type='clustered') 

#print the output. dctax_K is variable of interest
print(results_fe_tax.summary)
results.append(results_fe_tax)


############################
# Create the Output Tables #
############################

stargazer = Stargazer([results_baseline_dk, results_baseline_tax, 
                       results_varsel_dk, results_varsel_tax,
                       results_fe_dk, results_fe_tax])

# Customize the output
stargazer.title('Regression Results for Fixed Effects Model')
stargazer.rename_covariates({'dk': '$DK_{mxt}$', 'dk_ctax': '$DKT_{mxt}$',
                             'dk_notax': '$DKNT_{mxt}$',
                             'ldist': 'Distance', 'lcy': 'Country Y',
                              'lpy': 'Country Y Price', 'contig': 'Contiguous', 
                              'comlang_ethno': 'Common Language/Ethnicity',
                              'fta': 'Free Trade Agreement', 'wto': 'WTO Membership',
                              'eu': 'EU Membership'})

# Report only the treatment variables
to_report = ['dk', 'dk_ctax', 'dk_notax'] 
stargazer.covariate_order(to_report)

# Output the results to LaTeX
latex_output = stargazer.render_latex()
print(latex_output)


################################
# Subcomponent Results Table 5 #
################################
#Use the fixed effects specification but change the dependent variables

results2 = []

#############Import Volume

#####
###Kyoto Treatment 
Y = AF_data_FE['limp']
X = AF_data_FE[['dk', 'lcy', 'lpy', 'fta', 'wto', 'eu'] + mrterms]

X = sm.add_constant(X)

# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = PanelOLS(Y, X, entity_effects=True)
results_fe_dk = model.fit(cov_type='clustered', cluster_time=True)  

#print the output. dk is variable of interest
print(results_fe_dk.summary)
results2.append(results_fe_dk)

#####
###Carbon Pricing Treatment 
Y = AF_data_FE['limp']
X = AF_data_FE[['dk_ctax', 'dk_notax', 'lcy', 'lpy', 'fta', 'wto', 'eu'] + mrterms]

X = sm.add_constant(X)

# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = PanelOLS(Y, X, entity_effects=True)
results_fe_dk = model.fit(cov_type='clustered') 

#print the output. dk is variable of interest
print(results_fe_dk.summary)
results2.append(results_fe_dk)

#############Emissions Intensity

#####
###Kyoto Treatment 
Y = AF_data_FE['lint']
X = AF_data_FE[['dk', 'lcy', 'lpy', 'fta', 'wto', 'eu'] + mrterms]

X = sm.add_constant(X)

# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = PanelOLS(Y, X, entity_effects=True)
results_fe_dk = model.fit(cov_type='clustered') 

#print the output. dk is variable of interest
print(results_fe_dk.summary)
results2.append(results_fe_dk)

#####
###Carbon Pricing Treatment 
Y = AF_data_FE['lint']
X = AF_data_FE[['dk_ctax', 'dk_notax', 'lcy', 'lpy', 'fta', 'wto', 'eu'] + mrterms]

X = sm.add_constant(X)

# Fit the fixed effects model. Entity effects includes effects by sid (country pairing-industry FE)
model = PanelOLS(Y, X, entity_effects=True)
results_fe_dk = model.fit(cov_type='clustered')  # Clustered standard errors

#print the output. dk is variable of interest
print(results_fe_dk.summary)
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
