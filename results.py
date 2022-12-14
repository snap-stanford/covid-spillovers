import argparse
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import metis
import numpy as np
import networkx as nx
import os
from omegaconf import OmegaConf
import pickle
from scipy.stats import pearsonr, norm
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import torch as t
from torch.utils.data import DataLoader, SubsetRandomSampler

import covid_constants_and_util as cu
from dataset import *
import helper_methods_for_aggregate_data_analysis as helper
from poisson_reg_model import *
from sampling import load_train_test_indices_from_cfg


#####################################################################
# Functions to evaluate model results and fitting
#####################################################################
def plot_loss_over_epochs(exts, labels, ax=None, skip_n=0, label_axes=True):
    """
    Plot train losses over epochs for multiple experiments.
    """
    assert len(exts) == len(labels)
    ext2losses = {}
    first_losses = []
    for ext in exts:
        fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_results', '%s.pkl' % ext)
        with open(fn, 'rb') as f:
            results_per_epoch = pickle.load(f)
        num_epochs = len(results_per_epoch)
        losses = []
        for ep in range(num_epochs):
            losses.append(results_per_epoch['epoch_%d' % ep]['train_loss'])
        ext2losses[ext] = (np.arange(skip_n, num_epochs), losses[skip_n:])
        first_losses.append(losses[skip_n])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    for ext, lbl in zip(exts, labels):
        epochs, losses = ext2losses[ext]
        if losses[0] > np.percentile(first_losses, 90):
            ax.plot(epochs, losses, label=lbl)  # only label outliers
        else:
            ax.plot(epochs, losses)
    ax.grid(alpha=0.3)
    ax.legend()
    if label_axes:
        ax.set_ylabel('Negative log likelihood', fontsize=12)
        ax.set_xlabel('Epoch', fontsize=12)
    return ax
        
def plot_parameter_over_epochs(exts, labels, num_epochs, device, param_func, param_name=None, ax=None, label_axes=True):
    """
    Plot how a parameter converges over epochs for multiple experiments.
    """
    assert len(exts) == len(labels)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    final_vals = []
    for ext, lbl in zip(exts, labels):
        vals = []
        for ep in range(num_epochs):
            model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', '%s_ep%d' % (ext, ep))
            state_dict = t.load(model_fn, map_location=device)
            vals.append(param_func(state_dict))
        ax.plot(np.arange(num_epochs), vals, label=lbl)
        final_vals.append(vals[-1])
    if param_name is None:
        param_name = 'Parameter'
    else:
        ax.set_title(param_name, fontsize=12)
    print('%s final val: mean=%.3f, std=%.3f' % (param_name, np.mean(final_vals), np.std(final_vals)))
    ax.grid(alpha=0.3)
    if label_axes:
        ax.set_ylabel('%s value' % param_name, fontsize=12)
        ax.set_xlabel('Epoch', fontsize=12)
    return ax

def get_final_parameter_values(exts, device, param_func, param_name):
    """
    Extracts final value for a given parameter over multiple experiments.
    """
    vals = []
    for ext in exts:
        model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', ext)
        state_dict = t.load(model_fn, map_location=device)
        vals.append(param_func(state_dict))
    print('%s: mean=%.3f, std=%.3f' % (param_name, np.mean(vals), np.std(vals)))
    return vals

def get_tier_pair_weights_for_experiment(ext, device, group2idx, print_results=False):
    """
    Gets final spillover estimates for a single experiment. Returns dataframe with tier pair weights
    per POI group.
    """
    model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', ext)
    state_dict = t.load(model_fn, map_location=device)    
    results = []
    for group, idx in group2idx.items():
        results_dict = {'group':group}
        # pairwise weights normalized by 0->0
        for s, d in [(0, 1), (1, 0), (1, 1)]:
            weight = get_tier_pair_weight(state_dict, s, d, group_idx=idx)
            results_dict['%d->%d' % (s,d)] = np.exp(float(weight))
        tier_self_weight = get_tier_self_weight(state_dict, group_idx=idx)
        results_dict['self'] = np.exp(float(tier_self_weight))
        if print_results:
            print(group, np.round([results_dict[k] for k in results_dict if k != 'group'], 4))
        results.append(results_dict)
    columns = list(results[-1].keys())
    df = pd.DataFrame(results, columns=columns)
    return df

def summarize_tier_pair_weights_over_multiple_models(exts, device, group2idx):
    """
    Summarizes spillover estimates over multiple experiments. Returns dataframe of mean and SE per POI group.
    """
    ext2df = {}
    for ext in exts:
        ext2df[ext] = get_tier_pair_weights_for_experiment(ext, device, group2idx, print_results=False)
    results = {'group': ext2df[exts[0]].group.values}
    for s,d in [(0, 1), (1, 0), (1, 1)]:
        vecs = []
        for ext, df in ext2df.items():
            vec = df['%d->%d' % (s, d)].values
            vecs.append(vec)
        vecs = np.array(vecs)  # n_experiments x n_groups
        mean = np.mean(vecs, axis=0)
        se = np.std(vecs, axis=0, ddof=1)
        results['%d->%d mean' % (s, d)] = mean
        results['%d->%d se' % (s, d)] = se 
    df = pd.DataFrame(results)
    return df 

def plot_tier_pair_weights(results_df, tier_pairs, sort_col, bar_height=10, xlim=None, labels=None, 
                           a=0.05, use_bonferroni=False):
    """
    Plots spillover estimates based on dataframe returned by summarize_tier_pair_weights_over_multiple_models.
    """
    results_df = results_df.sort_values(by=sort_col)
    fig, ax = plt.subplots(figsize=(8,15))
    spacing = bar_height * (len(tier_pairs)+1.5)
    bar_pos = np.arange(0, len(results_df)*spacing, spacing)
    if use_bonferroni:
        a /= len(results_df)
    zscore = norm.ppf(1-(a/2))
    print('a=%.3f (using Bonferroni=%s) -> zscore=%.3f' % (a, use_bonferroni, zscore))
    for i, tier_pair in enumerate(tier_pairs):
        x_label = '%d->%d mean' % tier_pair
        x = results_df[x_label].values - 1  # subtract 1 to get diff
        err = zscore * results_df['%d->%d se' % tier_pair].values
        num_above = np.sum((x-err) > 0)  # significantly above 0
        num_below = np.sum((x+err) < 0)  # significantly below 0
        print('%d->%d: %d above, %d below' % (tier_pair[0], tier_pair[1], num_above, num_below))
        y = bar_pos - (i*bar_height)
        label = labels[i] if labels is not None else None
        ax.barh(y, x, xerr=err, align='center', height=bar_height, alpha=0.8, 
                error_kw={'capsize':1.5, 'linewidth':1}, label=label)
        
    categories = [cu.TOPCATEGORIES_TO_SHORTER_NAMES[g] if g in cu.TOPCATEGORIES_TO_SHORTER_NAMES else g for g in results_df.group.values]
    ax.set_yticks(bar_pos - bar_height)
    ax.set_yticklabels(categories, fontsize=14)
    ax.set_xlabel('Percent change in visits relative to PP', fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    cu.set_xaxis_labels_as_percentages(ax)
    ax.grid(alpha=0.3)
    if labels is not None:
        ax.legend(fontsize=12)

def plot_percent_change_in_weights(df1, df2, tier_pairs, bar_height=10, xlim=None, labels=None):
    """
    Compares two dataframes (returned by summarize_tier_pair_weights_over_multiple_models) and computes
    the percent change in parameter mean.
    """
    assert all(df1.group.values == df2.group.values)
    fig, ax = plt.subplots(figsize=(8,15))
    spacing = bar_height * (len(tier_pairs)+1.5)
    bar_pos = np.arange(0, len(df1)*spacing, spacing)
    for i, tier_pair in enumerate(tier_pairs):
        col = '%d->%d mean' % tier_pair
        vec1 = df1[col].values
        vec2 = df2[col].values
        x = (vec1 - vec2) / vec2
        y = bar_pos - (i*bar_height)
        label = labels[i] if labels is not None else None
        ax.barh(y, x, left=np.zeros(len(x)), align='center', height=bar_height, alpha=0.8, label=label)
        
    categories = [cu.TOPCATEGORIES_TO_SHORTER_NAMES[g] if g in cu.TOPCATEGORIES_TO_SHORTER_NAMES else g for g in df1.group.values]
    ax.set_yticks(bar_pos - bar_height)
    ax.set_yticklabels(categories, fontsize=14)
    ax.set_xlabel('Percent change in parameters', fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    cu.set_xaxis_labels_as_percentages(ax)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    
def plot_cbg_weights_over_multiple_experiments(exts, device, cbg_attrs, bar_height=5, xlim=None, a=0.05):
    """
    Plots CBG weights, summarized over multiple experiments.
    """
    all_weights = []
    for ext in exts:
        model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', ext)
        state_dict = t.load(model_fn, map_location=device)    
        weights = np.exp(get_cbg_weights(state_dict).detach().numpy())
        assert len(weights) == len(cbg_attrs)
        all_weights.append(weights)
    all_weights = np.array(all_weights)  # n_experiments x n_weights
    mean = np.mean(all_weights, axis=0)
    se = np.std(all_weights, axis=0, ddof=1)
    df = pd.DataFrame({'attr': cbg_attrs, 'mean': mean, 'se': se}).sort_values('mean')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    spacing = 2*bar_height
    bar_pos = np.arange(0, len(cbg_attrs)*spacing, spacing)
    zscore = norm.ppf(1-(a/2))
    err = zscore * df['se'].values
    ax.barh(bar_pos, df['mean'].values - 1, xerr=err, align='center', height=bar_height, alpha=0.8,
            error_kw={'capsize':1.5, 'linewidth':1})
    ax.set_yticks(bar_pos)
    ax.set_yticklabels(cbg_attrs, fontsize=14)
    ax.set_xlabel('Percent change in visits', fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    cu.set_xaxis_labels_as_percentages(ax)
    ax.grid(alpha=0.3)
    return df

def get_poi_subcat_weights_over_multiple_experiments(exts, device, subcats, indices):
    """
    Get POI weights per subcategory, summarized over multiple experiments.
    """
    assert len(subcats) == len(indices)
    all_weights = []
    for ext in exts:
        model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', ext)
        state_dict = t.load(model_fn, map_location=device)    
        weights = np.exp(get_poi_weights(state_dict).detach().numpy())[indices]
        all_weights.append(weights)
    all_weights = np.array(all_weights)  # n_experiments x n_weights
    mean = np.mean(all_weights, axis=0)
    se = np.std(all_weights, axis=0, ddof=1)
    df = pd.DataFrame({'subcat': subcats, 'mean': mean, 'se': se}).sort_values('mean')
    return df
        
def plot_distance_vs_pi(exts, device):
    """
    Plot the curves for pi across distances.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    all_param1 = []
    all_param2 = []
    distances = np.linspace(0, 50, 1000)  # in km
    for ext in exts:
        model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', ext)
        state_dict = t.load(model_fn, map_location=device)   
        param1 = get_pi_dist_scaling_param(state_dict).detach().numpy()
        all_param1.append(param1)
        param2 = get_pi_dist_exp_param(state_dict).detach().numpy()
        all_param2.append(param2)
        dists_pow = distances ** param2
        pis = 1 / (1 + (param1 * dists_pow))
        ax.plot(distances, pis, alpha=0.5)
    print('Param 1: mean=%.4f, std=%.4f' % (np.mean(all_param1), np.std(all_param1)))
    print('Param 2: mean=%.4f, std=%.4f' % (np.mean(all_param2), np.std(all_param2)))
    ax.set_xlabel('CBG-POI distance (km)', fontsize=12)
    ax.set_ylabel('Zero-inflation parameter pi', fontsize=12)
    ax.grid(alpha=0.3)
    plt.show()

# functions to get individual model parameters
def get_lambda_dist_scaling_param(state_dict):
    return state_dict['scaling_params'][0]

def get_pop_scaling_param(state_dict):
    return state_dict['scaling_params'][1]
    
def get_pi_dist_scaling_param(state_dict):
    return state_dict['scaling_params'][2]

def get_pi_dist_exp_param(state_dict):
    return state_dict['scaling_params'][3]

def get_tier_pair_weight(state_dict, source_tier, dest_tier, group_idx=None):
    if group_idx is None:
        return state_dict['tier_pair_weights'][source_tier, dest_tier] - state_dict['tier_pair_weights'][0, 0]
    return state_dict['tier_pair_weights'][group_idx, source_tier, dest_tier] - state_dict['tier_pair_weights'][group_idx, 0, 0]
    
def get_tier_self_weight(state_dict, group_idx=None):
    if group_idx is None:
        return state_dict['tier_self_weights'][1] - state_dict['tier_self_weights'][0]
    return state_dict['tier_self_weights'][group_idx, 1] - state_dict['tier_self_weights'][group_idx, 0]
    
def get_z_source_weight(state_dict, small_county=False, stage=0):
    return state_dict['z_source_weights'][int(small_county), stage]

def get_z_dest_weight(state_dict, small_county=False, stage=0):
    return state_dict['z_dest_weights'][int(small_county), stage]

def get_z_self_weight(state_dict, small_county=False, stage=0):
    return state_dict['z_self_weights'][int(small_county), stage]

def get_poi_weights(state_dict):
    return state_dict['poi_weights.weight'][0]

def get_cbg_weights(state_dict):
    return state_dict['cbg_weights.weight'][0]

    
#####################################################################
# Functions for local vs global analysis
#####################################################################
def get_main_experiment_names():
    """
    Return extension names for main experiment: 30 bootstrap samples, with negative sampling and 
    sampling nonzero data points with replacement.
    """
    main_exts = []  # main experiment
    for v in range(30):
        if v < 15:  # split into two runs
            main_exts.append(f'main_experiment_v{v}_2022_11_22_08_09_44')
        else:
            main_exts.append(f'main_experiment_v{v}_2022_11_23_08_31_57')
    return main_exts

def compute_county_county_weights(dset, mdl, mode='adj'):
    """
    Computes county-county weights for local vs global analysis.
    """
    def _compute_group_weights_for_i_and_j(fips_i, fips_j):
        """
        Helper function to compute group weights for county i and county j.
        See \phi(g, A, B) from Appendix.
        """
        cbgs = dset.county2cbgs[fips_i]
        cbg_attrs_i = all_cbg_attrs[cbgs]
        pois = dset.county2pois[fips_j]
        poi_attrs_j = all_poi_attrs[pois]

        # prepare relevant data in batch (no visits, tier, assignment variable, etc)
        c_vec = np.repeat(cbgs, len(pois))  # repeat each element
        cbg_attrs_expanded = np.repeat(cbg_attrs_i, len(pois), axis=0)
        p_vec = np.tile(pois, len(cbgs))  # repeat entire array
        poi_attrs_expanded = np.tile(poi_attrs_j, (len(cbgs), 1))
        edge_attrs = np.zeros((len(cbgs)*len(pois), dset.FEATURE_DICT['edge_num_attrs']))
        distances = dset.get_cbg_poi_dists(c_vec, p_vec) 
        edge_attrs[:, dset.FEATURE_DICT['cbg_poi_dist']] = distances        
        batch = (None, None, t.tensor(cbg_attrs_expanded), t.tensor(poi_attrs_expanded), 
                 t.tensor(edge_attrs))

        lambdas = mdl.get_lambdas_without_tier_or_z(batch)
        pis = mdl.get_pis(t.tensor(distances))
        expected_visits = lambdas * pis
        poi_groups = dset.poi_group_indices[p_vec]
        group_weights = np.zeros(len(dset.poi_group_labels)) 
        for group_idx, group in enumerate(dset.poi_group_labels):
            in_group = poi_groups == group_idx
            group_weights[group_idx] = float(expected_visits[in_group].sum())  # this is \phi(g,A,B)
        return group_weights

    assert mode in {'adj', 'within'}  # adjacent counties or within the same county
    # all CBG attrs - leave out county tier and assignment variable
    all_cbg_attrs = np.zeros((dset.num_cbgs(), dset.FEATURE_DICT['cbg_num_attrs']))
    start_idx, end_idx = dset.FEATURE_DICT['cbg_static_attrs']
    all_cbg_attrs[:, start_idx:end_idx] = dset.cbg_attrs.values
    # take the mean device count per CBG
    all_cbg_attrs[:, dset.FEATURE_DICT['cbg_device_ct']] = np.mean(dset.cbg_device_counts.values, axis=1)
    
    # all POI attrs - leave out county tier and assignment variable
    all_poi_attrs = np.zeros((dset.num_pois(), dset.FEATURE_DICT['poi_num_attrs']))
    subcats = dset.poi_subcat_indices
    all_poi_attrs[np.arange(dset.num_pois()), subcats] = 1  # one-hot
    start_idx, end_idx = dset.num_subcat_classes, dset.FEATURE_DICT['poi_static_attrs'][1]
    all_poi_attrs[:, start_idx:end_idx] = dset.poi_attrs.values[:, 2:].astype(float)
            
    # fill in weights
    adj_dict = helper.load_county_adjacency_dict()  # leaves out self-loops
    if mode == 'adj':
        weights = np.zeros((dset.num_counties(), dset.num_counties(), dset.num_poi_groups()))
    else:
        weights = np.zeros((dset.num_counties(), dset.num_poi_groups()))
    for ci, fips_i in enumerate(dset.indices['counties']):
        if mode == 'adj':
            for fips_j in adj_dict[fips_i]:
                if fips_j in dset.county2idx:  # some neighboring counties are outside of CA
                    cj = dset.county2idx[fips_j]
                    weights[ci, cj, :] = _compute_group_weights_for_i_and_j(fips_i, fips_j)
                    print('%s -> %d CBGs; %s -> %d POIs; total weight = %.3f' % 
                          (fips_i, len(dset.county2cbgs[fips_i]), fips_j, len(dset.county2pois[fips_j]), 
                           np.sum(weights[ci, cj, :])))
        else:
            weights[ci, :] = _compute_group_weights_for_i_and_j(fips_i, fips_i)
            print('%s -> %d CBGs, %d POIs; total weight = %.3f' % 
              (fips_i, len(dset.county2cbgs[fips_i]), len(dset.county2pois[fips_i]), np.sum(weights[ci, :])))
    return weights

def compute_county_county_weights_in_parallel(args):
    """
    Run this to compute county-county weights in parallel over fitted models.
    """
    if args.mode == 'outer':  
        exts = get_main_experiment_names()
        for i, experiment in enumerate(exts):
            out_str = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'outputs', 'county_county_weights_v%d' % i)
            cmd = f'nohup python -u results.py --mode inner --experiment {experiment} ' \
                f'--weights_fn county_county_weights_v{i}.pkl > {out_str}.out 2>&1 &'
            print(cmd)
            os.system(cmd)
    else:
        device = t.device('cpu')  # assume we're on cpu
        mdl, dset = recreate_mdl_and_dset_from_experiment(args.experiment, device, dset=None, set_final_mdl_weights=True)
        weights = compute_county_county_weights(dset, mdl, 'adj')
        save_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'CA', 'adj_county_weights', args.weights_fn)
        with open(save_path, 'wb') as f:
            pickle.dump((dset.indices['counties'], dset.poi_group_labels, weights), f)
            

def get_within_county_weights():
    """
    Return within-county tier weights and county weights. 
    """
    save_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'CA', 'same_county_weights_2022_04_18_16_45_59.pkl')
    with open(save_path, 'rb') as f:
        counties, group_labels, within_county_weights = pickle.load(f)
    device = t.device('cpu')
    model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', 'test_poireg_t1t2_h5_same_2022_04_18_16_45_59')
    state_dict_same = t.load(model_fn, map_location=device)
    tier_self_weights = np.exp(state_dict_same['tier_self_weights'].detach().numpy())
    return within_county_weights, tier_self_weights

def compute_percent_reduction(dset, tier_self_weights, within_county_weights, tier_pair_weights, county_county_weights,
                              group_idx=None, county2macrocounty=None, actual_tiers=None):
    """
    Compute the percent of mobility reduction kept per county under condition X vs when all counties go to purple.
    Default: county goes to purple while all other counties remain in red.
    If county2macrocounty is not None: county's macrocounty goes to purple while all other counties remain in red.
    If actual_tiers is not None: we use actual configuration of CA tiers (with all non-purple tiers set to red).
    """
    assert tier_self_weights.shape == (dset.num_poi_groups(), 2)
    assert (tier_self_weights > 0).all()  # should be exponentiated already 
    assert within_county_weights.shape == (dset.num_counties(), dset.num_poi_groups())
    assert tier_pair_weights.shape == (dset.num_poi_groups(), 2, 2)
    assert (tier_pair_weights > 0).all()  # should be exponentiated already 
    assert county_county_weights.shape == (dset.num_counties(), dset.num_counties(), dset.num_poi_groups())
    if county2macrocounty is not None:
        assert len(county2macrocounty) == dset.num_counties()
        assert all([f in county2macrocounty for f in dset.indices['counties']])
        assert actual_tiers is None
    if actual_tiers is not None:
        assert len(actual_tiers) == dset.num_counties()
        assert np.isin(actual_tiers, [0, 1]).all()  # 0 for purple and 1 for red
        assert county2macrocounty is None
        
    adj_dict = helper.load_county_adjacency_dict()
    percent_reductions = np.zeros(dset.num_counties()) * np.nan
    for ci, fips_i in enumerate(dset.indices['counties']):
        if actual_tiers is None or actual_tiers[ci] == 0:  # only compute over counties in purple
            # split neighbors into those also in purple vs those in red
            purple_neighbors = []
            red_neighbors = []            
            for fips_j in adj_dict[fips_i]:
                if fips_j in dset.county2idx:  # only keep neighbors in CA
                    cj = dset.county2idx[fips_j]
                    if county2macrocounty is not None:
                        if county2macrocounty[fips_i] == county2macrocounty[fips_j]:
                            purple_neighbors.append(cj)
                        else:
                            red_neighbors.append(cj)
                    elif actual_tiers is not None:
                        if actual_tiers[cj] == 0:
                            purple_neighbors.append(cj)
                        else:
                            red_neighbors.append(cj)
                    else:
                        red_neighbors.append(cj)  # in default, all neighbors are in red
            neighbors = purple_neighbors + red_neighbors

            if group_idx is None:  # sum over groups
                num = np.sum(within_county_weights[ci, :] * (tier_self_weights[:, 1] - tier_self_weights[:, 0]))
                if len(purple_neighbors) > 0:
                    outflow_to_purple = county_county_weights[ci, purple_neighbors, :]  # num_neighbors x num_groups
                    num += np.sum(outflow_to_purple * (tier_pair_weights[:, 1, 1] - tier_pair_weights[:, 0, 0]))  # RR-PP
                if len(red_neighbors) > 0:
                    outflow_to_red = county_county_weights[ci, red_neighbors, :]
                    num += np.sum(outflow_to_red * (tier_pair_weights[:, 1, 1] - tier_pair_weights[:, 0, 1]))  # RR-PR
                outflow = county_county_weights[ci, neighbors, :]
                denom = np.sum(within_county_weights[ci, :] * (tier_self_weights[:, 1] - tier_self_weights[:, 0]))
                denom += np.sum(outflow * (tier_pair_weights[:, 1, 1] - tier_pair_weights[:, 0, 0]))  # RR-PP
            else:
                num = np.sum(within_county_weights[ci, group_idx] * (tier_self_weights[group_idx, 1] - tier_self_weights[group_idx, 0]))
                if len(purple_neighbors) > 0:
                    outflow_to_purple = county_county_weights[ci, purple_neighbors, group_idx]  # num_neighbors
                    num += np.sum(outflow_to_purple * (tier_pair_weights[group_idx, 1, 1] - tier_pair_weights[group_idx, 0, 0]))  # RR-PP
                if len(red_neighbors) > 0:
                    outflow_to_red = county_county_weights[ci, red_neighbors, group_idx]
                    num += np.sum(outflow_to_red * (tier_pair_weights[group_idx, 1, 1] - tier_pair_weights[group_idx, 0, 1]))  # RR-PR
                outflow = county_county_weights[ci, neighbors, group_idx]
                denom = np.sum(within_county_weights[ci, group_idx] * (tier_self_weights[group_idx, 1] - tier_self_weights[group_idx, 0]))
                denom += np.sum(outflow * (tier_pair_weights[group_idx, 1, 1] - tier_pair_weights[group_idx, 0, 0]))  # RR-PP
            percent_reductions[ci] = num / denom
    return percent_reductions
    
def summarize_percent_reduction_over_experiments(dset, device, group_name=None, county2macrocounty=None, 
                                                 actual_tiers=None, a=0.05, verbose=False, county_type='all'):
    """
    Report mean percent mobility reduction over counties, summarized over main experiments.
    """
    assert county_type in {'all', 'small', 'large'}
    if actual_tiers is not None:
        counties_to_keep = actual_tiers == 0  # only compute over counties in purple
    else:
        counties_to_keep = np.ones(dset.num_counties()).astype(bool)
    if county_type == 'large':
        counties_to_keep = counties_to_keep & (dset.county_populations >= cu.LARGE_COUNTY_CUTOFF)
    elif county_type == 'small':
        counties_to_keep = counties_to_keep & (dset.county_populations < cu.LARGE_COUNTY_CUTOFF)
    if np.sum(counties_to_keep) < dset.num_counties():
        print('Computing mean percent of mobility reduction kept over %d counties' % np.sum(counties_to_keep))
    
    within_county_weights, tier_self_weights = get_within_county_weights()
    main_exts = get_main_experiment_names()
    all_prs = []
    for i, experiment in enumerate(main_exts):
        save_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'CA/adj_county_weights/county_county_weights_v%s.pkl' % i)
        with open(save_path, 'rb') as f:
            counties, group_labels, county_county_weights = pickle.load(f)
        model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', experiment)
        state_dict_adj = t.load(model_fn, map_location=device)
        tier_pair_weights = np.exp(state_dict_adj['tier_pair_weights'].detach().numpy())
        if group_name is not None:
            group_idx = list(group_labels).index(group_name)
        else:
            group_idx = None
        prs = compute_percent_reduction(dset, tier_self_weights, within_county_weights, 
                                        tier_pair_weights, county_county_weights, group_idx=group_idx,
                                        county2macrocounty=county2macrocounty, actual_tiers=actual_tiers)
        county_mean = np.mean(prs[counties_to_keep])
        if verbose:
            print(i, county_mean)
        all_prs.append(county_mean)
    
    mean_of_means = np.mean(all_prs)
    se = np.std(all_prs, ddof=1)
    zscore = norm.ppf(1-(a/2))
    print('%.3f (%.3f-%.3f)' % (mean_of_means, mean_of_means - (zscore * se), mean_of_means + (zscore * se)))
    return all_prs
    
    
def compute_kcut_cost(dset, tier_self_weights, within_county_weights, tier_pair_weights, 
                      county_county_weights, county2macrocounty):
    """
    Compute minimum k-cut objective (Eq 9). Use this to confirm that maximizing percent reduction and minimizing
    k-cut objective are the same.
    """
    assert tier_self_weights.shape == (dset.num_poi_groups(), 2)
    assert (tier_self_weights > 0).all()  # should be exponentiated already 
    assert within_county_weights.shape == (dset.num_counties(), dset.num_poi_groups())
    assert (tier_pair_weights > 0).all()
    assert tier_pair_weights.shape == (dset.num_poi_groups(), 2, 2)
    assert county_county_weights.shape == (dset.num_counties(), dset.num_counties(), dset.num_poi_groups())
    
    adj_dict = helper.load_county_adjacency_dict()
    total_cost = 0
    for ci, fips_i in enumerate(dset.indices['counties']):
        neighbors = []  # all neighbors
        diff_macro_neighbors = []  # neighbors in different macrocounty
        macrocounty_i = county2macrocounty[fips_i]
        for fips_j in adj_dict[fips_i]:
            if fips_j in dset.county2idx:
                cj = dset.county2idx[fips_j]
                neighbors.append(cj)
                if county2macrocounty[fips_j] != macrocounty_i:
                    diff_macro_neighbors.append(cj)
        diff_macro_outflow = county_county_weights[ci, diff_macro_neighbors, :]
        num = np.sum(diff_macro_outflow * (tier_pair_weights[:, 0, 1] - tier_pair_weights[:, 0, 0]))
        outflow = county_county_weights[ci, neighbors, :]
        denom = np.sum(outflow * (tier_pair_weights[:, 1, 1] - tier_pair_weights[:, 0, 0]))
        denom += np.sum(within_county_weights[ci, :] * (tier_self_weights[:, 1] - tier_self_weights[:, 0]))
        total_cost += (num / denom)
    return total_cost

def make_networkx_graph_for_metis(dset, tier_self_weights, within_county_weights, tier_pair_weights, 
                                  county_county_weights):
    """
    Make undirected county-county graph with edges weighted based on spillover estimates (Eq 9).
    """
    assert tier_self_weights.shape == (dset.num_poi_groups(), 2)
    assert (tier_self_weights > 0).all()  # should be exponentiated already 
    assert within_county_weights.shape == (dset.num_counties(), dset.num_poi_groups())
    assert (tier_pair_weights > 0).all()
    assert tier_pair_weights.shape == (dset.num_poi_groups(), 2, 2)
    assert county_county_weights.shape == (dset.num_counties(), dset.num_counties(), dset.num_poi_groups())
    
    adj_dict = helper.load_county_adjacency_dict()
    edge2weight = {}
    county_county_mat = np.zeros((dset.num_counties(), dset.num_counties()))
    for ci, fips_i in enumerate(dset.indices['counties']):
        neighbors = [dset.county2idx[fips_j] for fips_j in adj_dict[fips_i] if fips_j in dset.county2idx]
        outflow = county_county_weights[ci, neighbors, :]  # num_neighbors x num_groups
        nums = np.sum(outflow * (tier_pair_weights[:, 0, 1] - tier_pair_weights[:, 0, 0]), axis=1)  # num_neighbors
        denom = np.sum(outflow * (tier_pair_weights[:, 1, 1] - tier_pair_weights[:, 0, 0]))
        denom += np.sum(within_county_weights[ci, :] * (tier_self_weights[:, 1] - tier_self_weights[:, 0]))
        directed_weights = nums / denom  # from Eq 9
        for cj, w in zip(neighbors, directed_weights):
            edge = tuple(sorted((ci, cj)))  # standardize ordering
            if edge in edge2weight:
                edge2weight[edge] = edge2weight[edge] + w
            else:
                edge2weight[edge] = w
    
    G = nx.Graph()
    for (ci, cj), w in edge2weight.items():
        G.add_edge(ci, cj, capacity=w)
        G.adj[ci][cj]['weight'] = int(10000 * w)  # weight needs to be integer for metis.part_graph
    G.graph['edge_weight_attr'] = 'weight'
    return G

def compute_metis_partition_per_k(dset, device, G, ks, a=0.05):
    """
    Use METIS to compute macrocounty partitions for each value of k.
    """
    mean_for_k = []
    lower_for_k = []
    upper_for_k = []
    exts = get_main_experiment_names()
    for k in ks:
        cut, parts = metis.part_graph(G, k, ubvec=[1.05], contig=True, recursive=True)
        num_parts = len(set(parts))
        if num_parts != k:
            raise Exception('WARNING: only %d parts' % num_parts)
        print(k, Counter(parts))
        county2macrocounty = {}
        for ci, p in zip(G.nodes, parts):
            county2macrocounty[dset.indices['counties'][ci]] = p
        all_prs = summarize_percent_reduction_over_experiments(dset, device, county2macrocounty=county2macrocounty)
        mean_of_means = np.mean(all_prs)
        se = np.std(all_prs, ddof=1)
        mean_for_k.append(mean_of_means)
        zscore = norm.ppf(1-(a/2))
        lower_for_k.append(mean_of_means - (zscore * se))
        upper_for_k.append(mean_of_means + (zscore * se))
    return mean_for_k, lower_for_k, upper_for_k
    
def visualize_partition(county2macrocounty, plot_kwargs=None):
    if plot_kwargs is None:
        plot_kwargs = {}
    geoData = gpd.read_file('https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson')
    ca_geom = geoData[geoData['STATE'] == '06'].sort_values(by='id')
    macrocounties = [county2macrocounty[f] for f in sorted(county2macrocounty.keys())]
    ca_geom['macrocounty'] = macrocounties  # ca_geom has counties in FIPS order too
    ca_geom['coords'] = ca_geom['geometry'].apply(lambda x: x.representative_point().coords[:])
    ca_geom['coords'] = [coords[0] for coords in ca_geom['coords']]

    fig, ax = plt.subplots(figsize=(6, 6))
    ca_geom.plot('macrocounty', ax=ax, alpha=0.8, **plot_kwargs)
    for macrocounty, subdf in ca_geom.groupby('macrocounty'):
        polygon = subdf.geometry.unary_union
        # plot boundaries of counties within each macrocounty
        gpd.GeoDataFrame(geometry=[polygon], crs=subdf.crs).boundary.plot(color='white', ax=ax)
    ax.set_axis_off()  # remove axes
    return ax

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['inner', 'outer'], default='outer')
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--weights_fn', type=str)
    args = parser.parse_args()
    compute_county_county_weights_in_parallel(args)