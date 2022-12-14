# covid-spillovers
Code to generate results in "Estimating geographic spillover effects of COVID-19 policies from large-scale mobility networks" (AAAI 2023) by Serina Chang, Damir Vrabac, Jure Leskovec, and Johan Ugander.

Please find the extended version of our paper here: https://arxiv.org/abs/2212.06224. 

## Files

**covid_constants_and_util.py**: constants (eg, file paths) and utility functions 

**data_prep.ipynb**: notebook to process mobility data and county-level tier data into the format used in dataset.py

**dataset.py**: contains DataLoader object to load CBG-POI networks and covariates

**helper_methods_for_aggregate_data_analysis.py**: helper functions for loading and preprocessing data

**make_figures.ipynb**: make figures and check stats reported in paper

**poisson_reg_model.py**: code to fit zero-inflated Poisson regression model on sampled data

**results.py**: code to analyze model results and spillover costs across county partitions

**sampling.py**: code to sample from data and analyze sampled data
