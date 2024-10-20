Programs used to replicate Term Project 1 Kyoto and Carbon Leakage: Anticipation vs Policy by James Cabral

The two data files used for this project are:
-	Data_kyotoandleakage_restat.dta, which is obtained directly from the replication package for Aichele and Felbermayr (2015)
-	Carbon_tax_countries_withEU.csv is constructed based on information from the World Bank Carbon Pricing Dashboard. These data can be downloaded directly from: https://carbonpricingdashboard.worldbank.org/
- Values identified as having carbon pricing schemes are the ones reporting a positive share of global emissions covered in the compliance_emissions tab between 1995 and 2007. 


To replicate the main results of the paper, run the following program:
-	Carbon_Leakage_Paper1_Main.py
-	Note that parts of this code were written based on the Stata replication package for Aichele and Felbermayr (2015)

To replicate the shrinkage models used in our variable selection, run the following program:
-	Carbon_Leakage_Paper1_Shrinkage.py
-	Note that codes for the shrinkage models are based on codes from (James et al., 2023)

To replicate the tree-based models used in our variable selection, run the following program:
-	Carbon_Leakage_Paper1_Tree.py
-	Note that codes for the tree-based models are based on codes from (James et al., 2023)

To replicate the DAG and accompanying refutation tests, run the following program:
-	Carbon_Leakage_Paper1_DAG.py
-	Note that codes used to produce the DAG diagrams and the refutation tests are based on codes from (Molak, 2023)

References for Code:

Aichele, R. and G. Felbermayr (2015): “Kyoto and Carbon Leakage: An Empirical Analysis of the Carbon Content of Bilateral Trade,” The Review of Economics and Statistics, 97, 104–115.

James, G. M., D. Witten, T. Hastie, R. Tibshirani, and J. Taylor (2023): An Introduction to Statistical Learning: with Applications in Python, Springer.

Molak, A. (2023): Causal Inference and Discovery in Python: Unlock the secrets of modern causal machine learning with DoWhy, EconML, PyTorch and more, Packt Publishing Ltd
