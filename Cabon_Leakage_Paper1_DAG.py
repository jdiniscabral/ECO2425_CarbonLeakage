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
os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\ECO2425\Term_Project\Replication")

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

#############
# DAG Stuff #
#############

#import additional packages
from dowhy import CausalModel
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import networkx as nx

##################################
# Make a nice DAG for the report #
##################################

COLORS = [
    '#FFA500',
    '#FF0000'
]

#Define the matrix
matrix_w5 = np.array([
#TO: Co2, DK, GDP, Cost, EU, MR, Time
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],   
    [1, 1, 0, 0, 0, 0, 0]        
])

# Get the graphs
graph_exog = nx.from_numpy_array(matrix_w5, create_using=nx.DiGraph)

# Define custom labels
labels = {
    0: "Embodied CO2",
    1: "Differential Kyoto Ratification",
    2: "GDP per Capita",
    3: "Bilateral Costs",
    4: "EU, WTO Status",
    5: "Multilateral Resistance",
    6: "Time"
}


# Plotting the endogenous graph with custom labels
plt.figure(figsize=(10, 8))
nx.draw(
    G=graph_exog,
    with_labels=True,
    labels=labels,  # Use the custom labels
    node_size=7000,
    node_color=COLORS[0],  # Replace with COLORS[0] if defined
    font_color='black',
    pos=nx.planar_layout(graph_exog)  # You can use nx.planar_layout if desired
)

plt.show()

##########################
# Test the DAG with data #
##########################

#NOTE: Multilateral resistance excluded to allow the refutation tests to run

#Define the matrix for the baseline treatment variable for A and F (2015)
gml_graph = """
graph [
    directed 1

    node [id "lbeim" label "lbeim"]    
    node [id "dk" label "dk"]
    node [id "lcy" label "lcy"]
    node [id "lpy" label "lpy"]
    node [id "ldist" label "ldist"]
    node [id "comlang_ethno" label "Language"]
    node [id "eu" label "eu"]
    node [id "wto" label "wto"]
    node [id "mrdis" label "mrdis"]
    node [id "mreu" label "mreu"]
    node [id "mrwto" label "mrwto"]
    node [id "mrfta" label "mrfta"]
    
    edge [source "dk" target "lbeim"]
    edge [source "lcy" target "dk"]
    edge [source "lpy" target "dk"]  
    edge [source "eu" target "dk"]  
    edge [source "wto" target "dk"]  
    
    edge [source "eu" target "ldist"]  
    edge [source "wto" target "ldist"]
    edge [source "eu" target "comlang_ethno"]  
    edge [source "wto" target "comlang_ethno"]

    edge [source "lcy" target "lbeim"]
    edge [source "lpy" target "lbeim"]
    edge [source "ldist" target "lbeim"]
    edge [source "eu" target "lbeim"]
    edge [source "wto" target "lbeim"]
    edge [source "comlang_ethno" target "lbeim"]

]
"""

#Define the matrix for the notax treatment variable
#Include the notax treatment variable

gml_graph_notax = """
graph [
    directed 1

    node [id "lbeim" label "lbeim"]   
    node [id "dk_ctax" label "dk_ctax"]
    node [id "dk_notax" label "dk_notax"]
    node [id "lcy" label "lcy"]
    node [id "lpy" label "lpy"]
    node [id "ldist" label "ldist"]
    node [id "contig" label "contig"]
    node [id "comlang_ethno" label "Language"]
    node [id "eu" label "eu"]
    node [id "wto" label "wto"]
    node [id "mrdis" label "mrdis"]
    node [id "mreu" label "mreu"]
    node [id "mrwto" label "mrwto"]
    node [id "mrfta" label "mrfta"]
    node [id "mrcom" label "mrcom"]
    node [id "mrcon" label "mrcon"]

    edge [source "dk_ctax" target "lbeim"]    
    edge [source "dk_ctax" target "dk_notax"]    

    edge [source "dk_notax" target "lbeim"]
    edge [source "lcy" target "dk_notax"]
    edge [source "lpy" target "dk_notax"]  
    edge [source "eu" target "dk_notax"]  
    edge [source "wto" target "dk_notax"]  
    
    edge [source "eu" target "contig"]  
    edge [source "wto" target "contig"]
    edge [source "eu" target "ldist"]  
    edge [source "wto" target "ldist"]
    edge [source "eu" target "comlang_ethno"]  
    edge [source "wto" target "comlang_ethno"]

    edge [source "lcy" target "lbeim"]
    edge [source "lpy" target "lbeim"]
    edge [source "ldist" target "lbeim"]
    edge [source "eu" target "lbeim"]
    edge [source "wto" target "lbeim"]
    edge [source "contig" target "lbeim"]
    edge [source "comlang_ethno" target "lbeim"]
]
"""


# Graph the DAG that will be tested on the data
# generate the model
model = CausalModel(
    data=AF_data,
    treatment='dk',
    outcome='lbeim',
    graph=gml_graph
)
model.view_model()

model_notax = CausalModel(
    data=AF_data,
    treatment='dk_notax',
    outcome='lbeim',
    graph=gml_graph_notax
)
model_notax.view_model()


#identify the estimand. Should be backdoor
estimand = model.identify_effect()
print(estimand)


#Main estimations
estimate = model.estimate_effect(
identified_estimand=estimand,
method_name='backdoor.linear_regression')
print(estimate) 

estimate_notax = model_notax.estimate_effect(
identified_estimand=estimand,
method_name='backdoor.linear_regression')
print(estimate_notax) 

## Refutation tests to reproduce the table in the main paper

#####
# Refutation Test using subset
#####
refute_subset = model.refute_estimate(
estimand=estimand,
estimate=estimate,
method_name="data_subset_refuter",
subset_fraction=0.4)

print(f'Estimate of causal effect (linear regression): {estimate.value}')
print(refute_subset)

refute_subset = model_notax.refute_estimate(
estimand=estimand,
estimate=estimate_notax,
method_name="data_subset_refuter",
subset_fraction=0.4)

print(f'Estimate of causal effect (linear regression): {estimate.value}')
print(refute_subset)

#####
# Refutation Test using placebo treatment
#####
placebo_refuter = model.refute_estimate(
    estimand=estimand, 
    estimate=estimate,
    method_name='placebo_treatment_refuter'
)
print(placebo_refuter)

placebo_refuter = model_notax.refute_estimate(
    estimand=estimand, 
    estimate=estimate_notax,
    method_name='placebo_treatment_refuter'
)
print(placebo_refuter)

