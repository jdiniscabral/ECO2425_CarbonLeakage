####################
# References:

#Code for regression results adapted from replication package of:
# Aichele, R. and G. Felbermayr (2015): “Kyoto and Carbon Leakage: An Empirical Analysis of the Carbon Content of
# Bilateral Trade,” The Review of Economics and Statistics, 97, 104–115.
# Obtained from AEJ

#Code for LASSO and Tree-based methods based on Jupyter codes from:
# James, G. M., D. Witten, T. Hastie, R. Tibshirani, and J. Taylor (2023): 
# An Introduction to Statistical Learning: with Applications in Python, Springer.

#Code for DAG, Refutation Tests based on codes from:
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier


#change directory
os.chdir(r"C:\Users\jcabr\OneDrive - University of Toronto\Coursework\ECO2425\Term_Project")

#import cleaned dataset
AF_data = pd.read_csv("AF_data_clean.csv")

#scale the features
data_match = AF_data
#.assign(**{f: (AF_data[f] - AF_data[f].mean())/AF_data[f].std() for f in X})

np.random.seed(2425)

#Replicate the results from Table 7 and Table 9 (End of the code)
# Also replicates max depth tuning from appendix

#################
# Meta-Learners #
#################
#Implemented as in CBIT Textbook

Y = 'lbeim'
X = ['lcy', 'lpy', 'ldist', 'contig', 'comlang_ethno', 'colony', 'trans',  'mrcon', 'mrcom', 'mrwto', 'mrfta', 'mreu', 'mrdis']\
    + [col for col in AF_data.columns if col.startswith('ttt') and col != 'tttyear_1995']
T1 = 'dk_1'
T2 = 'dk_neg1'


from lightgbm import LGBMRegressor

#META-Learners for the ATE and CATE
np.random.seed(2425)

#S-Learner
s_learner_1 = LGBMRegressor(max_depth=7, min_child_samples=30, random_state = 2425)
s_learner_2 = LGBMRegressor(max_depth=7, min_child_samples=30, random_state = 2425)

s_learner_1.fit(data_match[X+[T1]], data_match[Y])
s_learner_2.fit(data_match[X+[T2]], data_match[Y])

data_match["s_learner_cate_1"] = (s_learner_1.predict(data_match[X].assign(**{T1: 1})) -
                        s_learner_1.predict(data_match[X].assign(**{T1: 0})))

data_match["s_learner_cate_2"] = (s_learner_2.predict(data_match[X].assign(**{T2: 1})) -
                        s_learner_2.predict(data_match[X].assign(**{T2: 0})))

diff_ate_S = (data_match["s_learner_cate_1"]).mean() - (data_match["s_learner_cate_2"]).mean()
diff_ate_S

#separate CATE differences for Tax countries
diff_ate_S_tax = (data_match["s_learner_cate_1"][data_match['ctax_c'] == 1]).mean() - (data_match["s_learner_cate_2"][data_match['ctax_p'] == 1]).mean()
diff_ate_S_tax
diff_ate_S_notax = (data_match["s_learner_cate_1"][data_match['ctax_c'] == 0]).mean() - (data_match["s_learner_cate_2"][data_match['ctax_p'] == 0]).mean()
diff_ate_S_notax


#T-Learner

m0_1 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)
m1_1 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)
m0_2 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)
m1_2 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)

m0_1.fit(data_match.loc[data_match[T1] == 0, X], data_match.loc[data_match[T1] == 0, Y])
m1_1.fit(data_match.loc[data_match[T1] == 1, X], data_match.loc[data_match[T1] == 1, Y])
m0_2.fit(data_match.loc[data_match[T2] == 0, X], data_match.loc[data_match[T2] == 0, Y])
m1_2.fit(data_match.loc[data_match[T2] == 1, X], data_match.loc[data_match[T2] == 1, Y])

# estimate the CATE
data_match["t_learner_cate_1"] = m1_1.predict(data_match[X]) - m0_1.predict(data_match[X])
data_match["t_learner_cate_2"] = m1_2.predict(data_match[X]) - m0_2.predict(data_match[X])

diff_ate_T = (data_match["t_learner_cate_1"]).mean() - (data_match["t_learner_cate_2"]).mean()
diff_ate_T

diff_ate_T_tax = (data_match["t_learner_cate_1"][data_match['ctax_c'] == 1]).mean() - (data_match["t_learner_cate_2"][data_match['ctax_p'] == 1]).mean()
diff_ate_T_tax
diff_ate_T_notax = (data_match["t_learner_cate_1"][data_match['ctax_c'] == 0]).mean() - (data_match["t_learner_cate_2"][data_match['ctax_p'] == 0]).mean()
diff_ate_T_notax


#X-Learner

# first stage models
m0_1 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)
m1_1 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)
m0_2 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)
m1_2 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)

m0_1.fit(data_match.loc[data_match[T1] == 0, X], data_match.loc[data_match[T1] == 0, Y])
m1_1.fit(data_match.loc[data_match[T1] == 1, X], data_match.loc[data_match[T1] == 1, Y])
m0_2.fit(data_match.loc[data_match[T2] == 0, X], data_match.loc[data_match[T2] == 0, Y])
m1_2.fit(data_match.loc[data_match[T2] == 1, X], data_match.loc[data_match[T2] == 1, Y])                   

# propensity score model
g1 = RandomForestClassifier(max_depth=5, n_estimators=100, min_samples_leaf=30, random_state = 2425)
g2 = RandomForestClassifier(max_depth=5, n_estimators=100, min_samples_leaf=30, random_state = 2425)
g1.fit(data_match[X], data_match[T1]);
g2.fit(data_match[X], data_match[T2]);

d_1 = np.where(data_match[T1]==0,
                   m1_1.predict(data_match[X]) - data_match[Y],
                   data_match[Y] - m0_1.predict(data_match[X]))

d_2 = np.where(data_match[T2]==0,
                   m1_2.predict(data_match[X]) - data_match[Y],
                   data_match[Y] - m0_2.predict(data_match[X]))

mx0_1 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)
mx1_1 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)
mx0_2 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)
mx1_2 = LGBMRegressor(max_depth=5, min_child_samples=60, random_state = 2425)

mx0_1.fit(data_match.loc[data_match[T1] == 0, X], d_1[data_match[T1]==0])
mx1_1.fit(data_match.loc[data_match[T1] == 1, X], d_1[data_match[T1]==1]);
mx0_2.fit(data_match.loc[data_match[T2] == 0, X], d_2[data_match[T2]==0])
mx1_2.fit(data_match.loc[data_match[T2] == 1, X], d_2[data_match[T2]==1]);

def ps_predict1(df, t): 
    return g1.predict_proba(df[X])[:, t]
def ps_predict2(df, t): 
    return g2.predict_proba(df[X])[:, t]

data_match["x_learner_cate_1"] = (ps_predict1(data_match,1)*mx0_1.predict(data_match[X]) +
                ps_predict1(data_match,0)*mx1_1.predict(data_match[X]))
data_match["x_learner_cate_2"] = (ps_predict2(data_match,1)*mx0_2.predict(data_match[X]) +
                ps_predict2(data_match,0)*mx1_2.predict(data_match[X]))

diff_ate_X = (data_match["x_learner_cate_1"]).mean() - (data_match["x_learner_cate_2"]).mean()
diff_ate_X

diff_ate_X_tax = (data_match["x_learner_cate_1"][data_match['ctax_c'] == 1]).mean() - (data_match["x_learner_cate_2"][data_match['ctax_p'] == 1]).mean()
diff_ate_X_tax
diff_ate_X_notax = (data_match["x_learner_cate_1"][data_match['ctax_c'] == 0]).mean() - (data_match["x_learner_cate_2"][data_match['ctax_p'] == 0]).mean()
diff_ate_X_notax


#############################
# Tuning the Max Tree Depth #
#############################

#cummulative gain as implemented by CIBT textbook
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

#Use DK = -1 treatment
T1 = 'dk_neg1'

#set a max depth that will apply to all of the gradient boosting models and forest predictor models 
maxdepth = 3

#Split the data into test and train
x_train, x_test, y_train, y_test = train_test_split(AF_data[X + [T1]+ [Y]], AF_data[Y], test_size=0.33, random_state=2425)

### S-Learner:

s_learner_1 = LGBMRegressor(max_depth=maxdepth, min_child_samples=30)
s_learner_1.fit(x_train[X+[T1]], x_train[Y])

s_learner_cate_train_1 = (s_learner_1.predict(x_train[X].assign(**{T1: 1})) -
                        s_learner_1.predict(x_train[X].assign(**{T1: 0})))

s_learner_cate_test_1 = x_test.assign(
    cate=(s_learner_1.predict(x_test[X].assign(**{T1: 1})) - # predict under treatment
          s_learner_1.predict(x_test[X].assign(**{T1: 0}))) # predict under control
)

gain_curve_test_1 = cumulative_gain(s_learner_cate_test_1, "cate", y="lbeim", t="dk_neg1")
gain_curve_train_1 = cumulative_gain(x_train.assign(cate=s_learner_cate_train_1), "cate", y="lbeim", t="dk_neg1")
plt.plot(gain_curve_test_1, color="C0", label="Test")
plt.plot(gain_curve_train_1, color="C1", label="Train")
plt.plot([0, 100], [0, elast(x_train, "lbeim", "dk_neg1")], linestyle="--", color="black", label="Baseline")
plt.legend()
plt.title("S-Learner, Depth = 9");

###########################################

# T-learner

m0_1 = LGBMRegressor(max_depth=maxdepth, min_child_samples=60)
m1_1 = LGBMRegressor(max_depth=maxdepth, min_child_samples=60)

m0_1.fit(x_train.loc[x_train[T1] == 0, X], x_train.loc[x_train[T1] == 0, Y])
m1_1.fit(x_train.loc[x_train[T1] == 1, X], x_train.loc[x_train[T1] == 1, Y])

# estimate the CATE
t_learner_cate_1_train = m1_1.predict(x_train[X]) - m0_1.predict(x_train[X])
t_learner_cate_1_test = x_test.assign(cate=m1_1.predict(x_test[X]) - m0_1.predict(x_test[X]))

gain_curve_1_test = cumulative_gain(t_learner_cate_1_test, "cate", y="lbeim", t="dk_neg1")
gain_curve_1_train = cumulative_gain(x_train.assign(cate=t_learner_cate_1_train), "cate", y="lbeim", t="dk_neg1")
plt.plot(gain_curve_1_test, color="C0", label="Test")
plt.plot(gain_curve_1_train, color="C1", label="Train")
plt.plot([0, 100], [0, elast(x_test, "lbeim", "dk_neg1")], linestyle="--", color="black", label="Baseline")
plt.legend();
plt.title("T-Learner, Depth = 5");

###########################################

#X-Learner

# first stage models
m0_1 = LGBMRegressor(max_depth=maxdepth, min_child_samples=30)
m1_1 = LGBMRegressor(max_depth=maxdepth, min_child_samples=30)

m0_1.fit(x_train.loc[x_train[T1] == 0, X], x_train.loc[x_train[T1] == 0, Y])
m1_1.fit(x_train.loc[x_train[T1] == 1, X], x_train.loc[x_train[T1] == 1, Y])

# propensity score model
g1 = RandomForestClassifier(max_depth=maxdepth, n_estimators=100, min_samples_leaf=30, random_state = 2425)
g2 = RandomForestClassifier(max_depth=maxdepth, n_estimators=100, min_samples_leaf=30, random_state = 2425)
g1.fit(x_train[X], x_train[T1]);
g2.fit(x_train[X], x_train[T2]);

d_1 = np.where(x_train[T1]==0,
                   m1_1.predict(x_train[X]) - x_train[Y],
                   x_train[Y] - m0_1.predict(x_train[X]))


mx0_1 = LGBMRegressor(max_depth=maxdepth, min_child_samples=30)
mx1_1 = LGBMRegressor(max_depth=maxdepth, min_child_samples=30)

mx0_1.fit(x_train.loc[x_train[T1] == 0, X], d_1[x_train[T1]==0])
mx1_1.fit(x_train.loc[x_train[T1] == 1, X], d_1[x_train[T1]==1]);

def ps_predict1(df, t): 
    return g1.predict_proba(df[X])[:, t]
def ps_predict2(df, t): 
    return g2.predict_proba(df[X])[:, t]

x_cate_1_train = (ps_predict1(x_train,1)*mx0_1.predict(x_train[X]) +
                ps_predict1(x_train,0)*mx1_1.predict(x_train[X]))

x_cate_1_test = x_test.assign(cate=(ps_predict1(x_test,1)*mx0_1.predict(x_test[X]) +
                                ps_predict1(x_test,0)*mx1_1.predict(x_test[X])))

gain_curve_1_test = cumulative_gain(x_cate_1_test, "cate", y="lbeim", t="dk_neg1")
gain_curve_1_train = cumulative_gain(x_train.assign(cate=x_cate_1_train), "cate", y="lbeim", t="dk_neg1")
plt.plot(gain_curve_1_test, color="C0", label="Test")
plt.plot(gain_curve_1_train, color="C1", label="Train")
plt.plot([0, 100], [0, elast(x_test, "lbeim", "dk_neg1")], linestyle="--", color="black", label="Baseline")
plt.legend();
plt.title("X-Learner, Depth = 3");

#########################################
# Country-Specific Leakage Rates

Kyoto_countries = ['AUS', 'AUT', 'BEL', 'CAN', 'CHE', 'CZE', 'DEU', 'ESP',
                   'FRA', 'GBR', 'GRC', 'HUN', 'IRL', 'ITA', 'JPN', 'NLD',
                   'PRT', 'RUS', 'SVK', 'ROU', 'NZL', 'FIN', 'POL', 'NOR', 
                   'SWE', 'DNK', 'SVN', 'EST']

#get the country indicator and print the CATE for that country
#Note: For each country (AUS), this is the CATE conditional on AUS being either the importer OR exporter
# Restrict observations to 2002 when countries signed Kyoto
#actual heatmap is made in Excel
for country in Kyoto_countries:
    data_match[country + "_ind"] = ((data_match['ccode'] == country) | (data_match['pcode'] == country)).astype(int) 
    print(country)
    print((data_match["x_learner_cate_1"][(data_match[country + "_ind"] == 1)]).mean() - 
          (data_match["x_learner_cate_2"][(data_match[country + "_ind"] == 1)]).mean())
