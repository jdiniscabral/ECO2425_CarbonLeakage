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


#change directory
os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\ECO2425\Term_Project")

#import cleaned dataset
AF_data = pd.read_csv("AF_data_clean.csv")

# Define a list of the multilateral resistance variables
mrterms = ["mrdis", "mrcon", "mrcom", "mrwto", "mrfta", "mreu"]

###############
# PSM and IPW #
###############

#Source: Causal Inference for the Brave and True
import warnings
warnings.filterwarnings('ignore')

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

#Replicate Figure 6, Figure 7, Table5, Table 6

######################
# Matching Estimator #
######################
#CBIT implementation

#set up X and Y matrices
#for X, I am only keeping variables that I think are likely to predict the treatment variable
#lbeim for emissions embodied in imports
#limp for imports
#lint for carbon intensity
Y = 'lbeim'
X = ['lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  'mrcon', 'mrcom', 'mrwto', 'mrfta', 'mreu', 'mrdis'] + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']
T1 = 'dk_1'
T2 = 'dk_neg1'

#scale the features
data_match = AF_data

#Function for Matching
#Based on CBIT implementation
def matching(df, X, Y, T):
    
    #separate out treated and untreated observations
    treated_1 = df[df[T] == 1]
    untreated_1 = df[data_match[T] == 0]

    
    mt0_1 = KNeighborsRegressor(n_neighbors=5).fit(untreated_1[X], untreated_1[Y])
    mt1_1 = KNeighborsRegressor(n_neighbors=5).fit(treated_1[X], treated_1[Y])

    #match the observations
    predicted_1 = pd.concat([
        # find matches for the treated looking at the untreated knn model
        treated_1.assign(match=mt0_1.predict(treated_1[X])),
        
        # find matches for the untreated looking at the treated knn model
        untreated_1.assign(match=mt1_1.predict(untreated_1[X])),
    ])
    ate_1 = np.mean((2*predicted_1[T] - 1)*(predicted_1[Y] - predicted_1["match"]))
    return(ate_1)

ate_1 = matching(AF_data, X, Y, T1)
ate_2 = matching(AF_data, X, Y, T2)

ate_1 - ate_2

#NOT USED IN THE MAIN PAPER:

#carbon intensity
ate_1 = matching(AF_data, X, 'lint', T1)
ate_2 = matching(AF_data, X, 'lint', T2)

ate_1 - ate_2

#total imports
ate_1 = matching(AF_data, X, 'limp', T1)
ate_2 = matching(AF_data, X, 'limp', T2)

ate_1 - ate_2

######################################
# Bootstrapping Confidence Intervals #
######################################

np.random.seed(2425)
bootstrap_sample = 10000

ates1 = []
ates2 = []

for _ in range(bootstrap_sample):
    bs_sample = AF_data[X + [Y] + ['dk_1', 'dk_neg1']].sample(frac=0.1, replace=True)
    ate_1 = matching(bs_sample, X, Y, T1)
    ate_2 = matching(bs_sample, X, Y, T2)
    ates1.append(ate_1)
    ates2.append(ate_2)
    print(_)

ates1 = np.array(ates1)
ates2 = np.array(ates2)
sns.histplot(ates1, kde=False, color = "blue")
sns.histplot(ates2, kde=False, color = "red")
plt.vlines(np.percentile(ates2, 5), 0, 500, linestyles="dotted", color = "red")
plt.vlines(np.percentile(ates2, 95), 0, 500, linestyles="dotted", color = "red", label="DK = -1, 90% CI")
plt.vlines(np.percentile(ates1, 5), 0, 500, linestyles="dotted", color = "blue")
plt.vlines(np.percentile(ates1, 95), 0, 500, linestyles="dotted", label="DK = 1, 90% CI", color = "blue")
plt.xlabel("Average Treatment Effect", color='black') 
plt.ylabel("Frequency", color='black')
plt.legend();


#####################
# Propensity Scores #
#####################
#CBIT Implementation

#positivity not satisfied if EU or year FE included
X =  ['lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'trans', 'colony']
T1 = 'dk_1'
T2 = 'dk_neg1'

#run a logistic regression to get propensity scores
ps_model_1 = LogisticRegression(C=1e6).fit(data_match[X], data_match[T1])
ps_model_2 = LogisticRegression(C=1e6).fit(data_match[X], data_match[T2])

data_ps_1 = data_match.assign(propensity_score=ps_model_1.predict_proba(data_match[X])[:, 1])
data_ps_2 = data_match.assign(propensity_score=ps_model_2.predict_proba(data_match[X])[:, 1])

#weight the observations
weight_t_1 = 1 / data_ps_1.loc[data_ps_1[T1] == 1, "propensity_score"]
weight_nt_1 = 1 / (1 - data_ps_1.loc[data_ps_1[T1] == 0, "propensity_score"])
print("Original Sample Size", data_match.shape[0])
print("Treated Population Sample Size", sum(weight_t_1))
print("Untreated Population Sample Size", sum(weight_nt_1))

weight_t_2= 1 / data_ps_2.loc[data_ps_1[T2] == 1, "propensity_score"]
weight_nt_2 = 1 / (1 - data_ps_2.loc[data_ps_1[T2] == 0, "propensity_score"])
print("Original Sample Size", data_match.shape[0])
print("Treated Population Sample Size", sum(weight_t_2))
print("Untreated Population Sample Size", sum(weight_nt_2))

####################
# Positivity Plots #
####################

sns.distplot(data_ps_1.loc[data_ps_1[T1] == 0, "propensity_score"], kde=False, label="Non Treated")
sns.distplot(data_ps_1.loc[data_ps_1[T1] == 1, "propensity_score"], kde=False, label="Treated")
plt.title("Positivity Check DK = 1 Treatment")
plt.xlabel("Propensity Score", color='black') 
plt.ylabel("Frequency", color='black')
plt.tick_params(axis='y', labelsize=10, color='black')
plt.legend();

sns.distplot(data_ps_2.loc[data_ps_1[T2] == 0, "propensity_score"], kde=False, label="Non Treated")
sns.distplot(data_ps_2.loc[data_ps_1[T2] == 1, "propensity_score"], kde=False, label="Treated")
plt.title("Positivity Check DK = -1 Treatment")
plt.xlabel("Propensity Score", color='black') 
plt.ylabel("Frequency", color='black')
plt.tick_params(axis='y', labelsize=10, color='black')
plt.legend();

#################################
# Inverse Probability Weighting #
#################################
#CBIT Implementation

###compute weighting
weight_1 = ((data_ps_1[T1]-data_ps_1["propensity_score"]) /
          (data_ps_1["propensity_score"]*(1-data_ps_1["propensity_score"])))

weight_2 = ((data_ps_2[T2]-data_ps_2["propensity_score"]) /
          (data_ps_2["propensity_score"]*(1-data_ps_2["propensity_score"])))

y1_1 = sum(data_ps_1.loc[data_ps_1[T1] == 1, "lbeim"]*weight_t_1) / len(data_ps_1)
y0_1 = sum(data_ps_1.loc[data_ps_1[T1] == 0, "lbeim"]*weight_nt_1) / len(data_ps_1)
ate_1 = np.mean(weight_1 * data_ps_1["lbeim"])

y1_2 = sum(data_ps_2.loc[data_ps_1[T2] == 1, "lbeim"]*weight_t_2) / len(data_ps_2)
y0_2 = sum(data_ps_2.loc[data_ps_1[T2] == 0, "lbeim"]*weight_nt_2) / len(data_ps_2)
ate_2 = np.mean(weight_2 * data_ps_2["lbeim"])

print("Y1:", y1_1)
print("Y0:", y0_1)
print("ATE", ate_1)

print("Y1:", y1_2)
print("Y0:", y0_2)
print("ATE", ate_2)

ate_1 - ate_2

###############
# Bootstrapping

def bootstrap_ps(df, X, Y, T):
    
    #run a logistic regression model to predict the treatment variable using the X covariates
    ps_model_1 = LogisticRegression(C=1e6).fit(df[X], df[T])
    
    #add the propensity scores to the data frame
    data_ps_1 = df.assign(propensity_score=ps_model_1.predict_proba(df[X])[:, 1])

    ###Compute the Treatment Effect
    weight_1 = ((data_ps_1[T]-data_ps_1["propensity_score"]) /
              (data_ps_1["propensity_score"]*(1-data_ps_1["propensity_score"])))

    ate_1 = np.mean(weight_1 * data_ps_1["lbeim"])
    return(ate_1)

np.random.seed(2425)
bootstrap_sample = 5000

ates1 = []
ates2 = []

for _ in range(bootstrap_sample):
    # Sample with replacement and calculate ATE
    bs_sample = AF_data[X + [Y] + ['dk_1', 'dk_neg1']].sample(frac=0.1, replace=True)
    ate_1 = bootstrap_ps(bs_sample, X, Y, 'dk_1')
    ate_2 = bootstrap_ps(bs_sample, X, Y, 'dk_neg1')
    ates1.append(ate_1)
    ates2.append(ate_2)
    print(_)

ates1 = np.array(ates1)
ates2 = np.array(ates2)
sns.histplot(ates1, kde=False, color = 'blue')
sns.histplot(ates2, kde=False, color = 'red')
plt.vlines(np.percentile(ates1, 5), 0, 350, linestyles="dotted", color = "blue")
plt.vlines(np.percentile(ates1, 95), 0, 350, linestyles="dotted", label="DK = 1, 90% CI", color = "blue")
plt.vlines(np.percentile(ates2, 5), 0, 350, linestyles="dotted", color = "red")
plt.vlines(np.percentile(ates2, 95), 0, 350, linestyles="dotted", color = "red", label="DK = -1, 90% CI")
plt.xlabel("Average Treatment Effect", color='black') 
plt.ylabel("Frequency", color='black')
plt.legend();



#####################################
# Linear DR Learner

#CIBT implementation

import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression

pd.set_option("display.max_columns", 6)

Y = 'lbeim'
X =  ['lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'trans', 'colony']
T_i = 'dk_1'
T_x = 'dk_neg1'

def doubly_robust(df, X, Y, T):
    ps = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

doubly_robust(AF_data, X, Y, T_i)
doubly_robust(AF_data, X, Y, T_x)

###############
# Bootstrapping

ates1 = []
ates2 = []
bootstrap_sample = 10000

for _ in range(bootstrap_sample):
    # Sample with replacement and calculate ATE
    bs_sample = AF_data[X + [Y] + ['dk_1', 'dk_neg1']].sample(frac=0.1, replace=True)
    ate_1 = doubly_robust(bs_sample, X, Y, 'dk_1')
    ate_2 = doubly_robust(bs_sample, X, Y, 'dk_neg1')
    ates1.append(ate_1)
    ates2.append(ate_2)
    if _ % 100 == 0:
        print(_)

ates1 = np.array(ates1)
ates2 = np.array(ates2)
sns.histplot(ates1, kde=False, color = 'blue')
sns.histplot(ates2, kde=False, color = 'red')
plt.xlabel("Average Treatment Effect", color='black') 
plt.ylabel("Frequency", color='black')
plt.vlines(np.percentile(ates1, 5), 0, 500, linestyles="dotted", color = "blue")
plt.vlines(np.percentile(ates1, 95), 0, 500, linestyles="dotted", label="DK = 1, 90% CI", color = "blue")
plt.vlines(np.percentile(ates2, 5), 0, 500, linestyles="dotted", color = "red")
plt.vlines(np.percentile(ates2, 95), 0, 500, linestyles="dotted", color = "red", label="DK = -1, 90% CI")
plt.legend();


#################################
# Forest and Sparse DR Learners #
#################################
# EconML Notebook Implementation 

from sklearn.linear_model import LassoCV
from econml.dr import LinearDRLearner
from econml.dr import SparseLinearDRLearner
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from econml.dr import ForestDRLearner
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from econml.sklearn_extensions.linear_model import WeightedLassoCV

#Set up the data 
T_i = 'dk_1'
T_x = 'dk_neg1'
Y = 'lbeim'
X = ['lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans']

np.random.seed(2425)

#scale the features for the Sparse DR Learner
AF_data_scale = AF_data.assign(**{f: (AF_data[f] - AF_data[f].mean())/AF_data[f].std() for f in X})


#Forest and Sparse DR Learners included
#Comment out the one that is not needed

#Forest Learner
est = ForestDRLearner(model_regression=GradientBoostingRegressor(),
                      model_propensity=RandomForestClassifier(n_estimators=100, max_depth=10),
                      cv=3,
                      n_estimators=1000,
                      min_samples_leaf=10,
                      verbose=0, min_weight_fraction_leaf=.01)


# Sparse DR Learner
est = SparseLinearDRLearner(model_regression=WeightedLassoCV(cv=3),
                      model_propensity=RandomForestClassifier(max_depth=5, 
                      n_estimators=200, min_samples_leaf=30, random_state = 2425),
                      featurizer=PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))

est.fit(AF_data_scale[Y].ravel(), AF_data_scale[T_i], X=AF_data_scale[X])
ate_i_sum = est.effect_inference(AF_data_scale[X]).population_summary()
AF_data_scale["dr_learner_cate_1"] = est.effect_inference(AF_data_scale[X]).summary_frame()['point_estimate']
ate_i = est.ate(AF_data_scale[X], T0=0, T1=1)
print(ate_i_sum)

est.fit(AF_data_scale[Y].ravel(), AF_data_scale[T_x], X=AF_data_scale[X])
ate_x_sum = est.effect_inference(AF_data_scale[X]).population_summary()
est.effect_inference(AF_data_scale[X]).summary_frame()
AF_data_scale["dr_learner_cate_2"] = est.effect_inference(AF_data_scale[X]).summary_frame()['point_estimate']
ate_x = est.ate(AF_data_scale[X], T0=0, T1=1)
print(ate_x_sum)

ate_i - ate_x
 
#separate CATE differences for Tax countries
diff_ate_R_tax = (AF_data_scale["dr_learner_cate_1"][AF_data_scale['ctax_c'] == 1]).mean() - (AF_data_scale["dr_learner_cate_2"][AF_data_scale['ctax_p'] == 1]).mean()
diff_ate_R_tax
diff_ate_R_notax = (AF_data_scale["dr_learner_cate_1"][AF_data_scale['ctax_c'] == 0]).mean() - (AF_data_scale["dr_learner_cate_2"][AF_data_scale['ctax_p'] == 0]).mean()
diff_ate_R_notax


#Cumulative gain plots from implementation of CBIT
#To assess performance of models for different parameter values
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


#split into test and train data to get cummulative gain and assess model performance
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(AF_data[X + [T_i]+ [Y]], AF_data[Y], test_size=0.33, random_state=2425)

est.fit(x_train[Y].ravel(), x_train[T_i], X=x_train[X])
x_train["dr_learner_cate_1_train"] = est.effect_inference(x_train[X]).summary_frame()['point_estimate']
x_test["dr_learner_cate_1_test"] = est.effect_inference(x_test[X]).summary_frame()['point_estimate']
ate_i = est.ate(AF_data[X], T0=0, T1=1)


gain_curve_1_train = cumulative_gain(x_train, "dr_learner_cate_1_train", y="lbeim", t="dk_1")
gain_curve_1_test = cumulative_gain(x_test, "dr_learner_cate_1_test", y="lbeim", t="dk_1")

plt.plot(gain_curve_1_test, color="C1", label="Test")
plt.plot(gain_curve_1_train, color="C0", label="Train")

plt.plot([0, 100], [0, elast(x_train, "lbeim", "dk_1")], linestyle="--", color="black", label="Baseline")
plt.legend();
plt.title("Tree-Based DR Learner, DK = 1 Treatment");
