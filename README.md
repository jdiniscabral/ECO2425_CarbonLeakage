Programs and data used to replicate Term Project 2 Kyoto and Carbon Leakage: Anticipation vs Policy by James Cabral
The two data files used for this project are:
-	Data_kyotoandleakage_restat.dta, which is obtained directly from the replication package for Aichele and Felbermayr (2015)
-	Carbon_tax_countries_withEU.csv is constructed based on information from the World Bank Carbon Pricing Dashboard. These data can be downloaded directly from: https://carbonpricingdashboard.worldbank.org/
o	Values identified as having carbon pricing schemes are the ones reporting a positive share of global emissions covered in the compliance_emissions tab between 1995 and 2007. 

To merge the files, clean the data and produce the summary statistics tables, run the following program:
00_DataCleaning.py
-	Note that this code was partially based on the Stata replication package for Aichele and Felbermayr (2015)
-	Produces Summary Statistics Table 2

To replicate the figures in the data section and regression results, run the following program:
01_OLS_FE.py
-	Note that this code was partially based on the Stata replication package for Aichele and Felbermayr (2015)
-	Produces Figure 1, Figure 2, Table 4

To replicate the shrinkage and tree-based models used in our variable selection, run the following programs:
02_Shrinkage.py and 03_Tree.py
-	Note that codes for the shrinkage models and tree-based models are based on codes from (James et al., 2023)
-	Produces Figure 4, Figure 5, Table 3, Appendix Tables and Figure for Shrinkage and Tree-based Models

To replicate the DAG and accompanying refutation tests, run the following program:
04_DAG.py
-	Note that codes used to produce the DAG diagrams and the refutation tests are based on codes from (Molak, 2023)
-	Produces Figure 3

To replicate results for the matching estimator, IPW and DR learners, run the following program:
05_Match_DRLearn.py
-	Note that codes used to produce the Matching and IPW estimators and the linear DR estimator were based on notebooks from Causal Inference for the Brave and True. The other DR Learners were implemented based on EconML notebooks.
-	Produces Figure 6, Figure 7, Table 5, Table 6

To replicate the meta-learner results, run the following program:
06_MetaLearn.py
-	Note that codes used to produce the meta-learners were based on notebooks from Causal Inference for the Brave and True.
-	Produces Table 7, Figure 9, Appendix Figure for max depth tuning

To replicate the double ML and Causal forest results, run the following program:
07_DoubleML_CausalForest.py
-	Note that codes used to produce the double ML estimates were based on notebooks from Causal Inference for the Brave and True. Causal forests were implemented based on EconML notebooks.
-	Produces Table 8, Figure 8 and estimates included in Figure 9

References for Code:

Aichele, R. and G. Felbermayr (2015): “Kyoto and Carbon Leakage: An Empirical Analysis of the Carbon Content of Bilateral Trade,” The Review of Economics and Statistics, 97, 104–115.

Battocchi, K., Dillon, E., Hei, M., Lewis, G., Oka, P., Oprescu, M., Syrgkanis, V. “EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation,”. Available under “Notebooks” at: https://github.com/py-why/EconML

Facure, M. (2022).
Causal Inference for the Brave and True. Available at: https://matheusfacure.github.io/python-causality-handbook/landing-page.html.

James, G. M., D. Witten, T. Hastie, R. Tibshirani, and J. Taylor (2023): An Introduction to Statistical Learning: with Applications in Python, Springer.

Molak, A. (2023): Causal Inference and Discovery in Python: Unlock the secrets of modern causal machine learning with DoWhy, EconML, PyTorch and more, Packt Publishing Ltd
