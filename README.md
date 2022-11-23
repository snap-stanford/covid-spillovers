# covid-spillovers
Code for "Estimating geographic spillover effects of COVID-19 policies from large-scale mobility networks" (2022) by Serina Chang, Damir Vrabac, Jure Leskovec, and Johan Ugander.

## Files

**dataset.py**: contains DataLoader object to load CBG-POI networks and covariates

**sampling.py**: code to sample from data and analyze sampled data

**poisson_reg_model.py**: code to fit zero-inflated Poisson regression model on sampled data

**results.py**: code to analyze model results 

**covid_constants_and_util.py**: constants (eg, file paths) and utility functions 

**helper_methods_for_aggregate_data_analysis.py**: helper functions for loading and preprocessing data
