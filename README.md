# covid-spillovers
Code to generate results in "Estimating geographic spillover effects of COVID-19 policies from large-scale mobility networks" (AAAI 2023) by Serina Chang, Damir Vrabac, Jure Leskovec, and Johan Ugander.

Please find the extended version of our paper here: https://arxiv.org/abs/2212.06224. 

## Coding Files

`covid_constants_and_util.py`: constants (eg, file paths) and utility functions 

`helper_methods_for_aggregate_data_analysis.py`: helper functions for loading and preprocessing data (not all functions are used, this file is adapted from our [covid-mobility](https://github.com/snap-stanford/covid-mobility) repository)

`data_prep.ipynb`: process mobility data and county-level tier data into the format used in dataset.py

`dataset.py`: defines CBGPOIDataset object to load CBG-POI networks and covariates for model fitting and data analysis

`sampling.py`: sample from data (eg, distance-weighted negative sampling) and visualize sampled data

`poisson_reg_model.py`: fit zero-inflated Poisson regression model on sampled data

`results.py`: analyze model results and spillover costs across county partitions

`make_figures.ipynb`: make figures and check stats reported in paper

`requirements.txt`: our versions for all Python packages (not all packages may be used in this code)

## Data Files

**blueprints_cdph**: weekly archives from the California Department of Public Health (CDPH) for the Blueprint for a Safer Economy. Each spreadsheet provides each county's assigned Blueprint tier for that week as well as the COVID metrics (e.g., adjusted case rate) used to make the assignment. Archives are currently available on [CDPH website](https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/COVID-19/CaliforniaBlueprintDataCharts.aspx), but provided here as well for convenience.

**county_dynamic_attrs_2021_t1t2.pkl**: dynamic county-level variables used in our causal inference framework. This pickle file is a tuple of 6 objects:
1. FIPS code for each county (vector of length 58)
2. Population size for each county (vector of length 58)
3. Date of tier announcement (vector of length 9): from Feb 2 to Mar 30, 2021; note that tiers were announced on Tuesdays
4. Blueprint stage (vector of length 9): indicates whether the date is before or after the first statewide vaccine goal was met on Mar 12, 2021, after which the purple-red threshold for adjusted case rate was changed
5. County tier (matrix of length 9 x 58): tier for each county, indexed by the provided dates and FIPS codes; 1 indicates purple tier, 2 indicates red tier, 3 indicates orange tier, and 4 indicates yellow tier
6. Z variable (matrix of length 9 x 58): our constructed Z variable, which almost perfectly separates the counties in the purple and red tiers in these weeks, enabling a regression discontinuity design framework to estimate the effects of tiers on mobility

See Sections 3 and 4 of our paper for further descriptions of the California Blueprint and our algorithm to construct the Z variable. See `data_prep.ipynb`, "Prep county-level tiers in California" for our code to construct this pickle file. 

## Other data sources

* **Mobility data**: we use a dynamic mobility network that encodes the aggregated movements of individuals from census block groups (CBGs) to points-of-interest (POIs). Our mobility data comes from [SafeGraph Weekly Patterns](https://docs.safegraph.com/docs/weekly-patterns).

* **POI features**: we use POI features (subcategory, topcategory, latitude, longitude, and area) from [SafeGraph Places](https://docs.safegraph.com/docs/places).

* **CBG features**: we obtain CBG features (age percentages, race percentages, median income, geometry) from [SafeGraph Open Census Data](https://docs.safegraph.com/docs/open-census-data), which compiles data from the US Census Bureau's American Community Survey 5-year Estimates. This data is also available at [data.census.gov](https://www.census.gov/programs-surveys/acs/data.html).

*Note: We downloaded SafeGraph data in 2020 and 2021, when it was available to researchers through the SafeGraph consortium. SafeGraph data is [now available](https://www.safegraph.com/blog/safegraph-partners-with-dewey) through [Dewey](https://www.deweydata.io/), including historical data in the time period that we study.
