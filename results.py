import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import numpy as np
import networkx as nx
import os
from omegaconf import OmegaConf
import pickle
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import torch as t
from torch.utils.data import DataLoader, SubsetRandomSampler

import covid_constants_and_util as cu
from dataset import *
import helper_methods_for_aggregate_data_analysis as helper
from model_experiments import *
from sampling import load_train_test_indices_from_cfg

CBG_COLOR = 'tab:blue'
POI_COLOR = 'tab:orange'

def plot_parameter_over_epochs(exts, labels, num_epochs, device, param_func, param_name=None, ax=None, add_labels=True):
    """
    Plot how a parameter converges over epochs for multiple experiments.
    """
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
    ax.legend()
    if add_labels:
        ax.set_ylabel('%s value' % param_name, fontsize=12)
        ax.set_xlabel('Epoch', fontsize=12)
    return ax

def report_parameter_mean_and_std(exts, min_epoch, max_epoch, device, param_func, param_name):
    """
    Extracts the value for a single parameter, reports mean and std for potentially multiple epochs.
    """
    vals = []
    for ext in exts:
        for ep in range(min_epoch, max_epoch):
            model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', '%s_ep%d' % (ext, ep))
            state_dict = t.load(model_fn, map_location=device)
            vals.append(param_func(state_dict))
    print('%s: mean=%.3f, std=%.3f' % (param_name, np.mean(vals), np.std(vals)))

def summarize_model_results(ext, device, group2idx, max_epochs=None, print_results=True,
                            plot_tier_pair_weights=True, xlim=None):
    """
    Summarizes final parameter estimates for a single experiment. Returns dataframe with tier pair weights
    per POI group.
    """
    if max_epochs is not None:
        last_epoch = 0
        for ep in range(max_epochs):
            model_fn_ep = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', '%s_ep%d' % (ext, ep))
            if os.path.isfile(model_fn_ep):
                last_epoch = ep
            else:
                break
        print('Completed %d epochs so far' % (last_epoch+1))
    model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', ext)
    state_dict = t.load(model_fn, map_location=device)
    if print_results:
        print('Scaling params:', state_dict['scaling_params'])
        print('Z source weights:', state_dict['z_source_weights'])
    
    results = []
    for group, idx in group2idx.items():
        results_dict = {'group':group}
        # pairwise weights normalized by 0->0
        for s, d in [(0, 1), (1, 0), (1, 1)]:
            weight = state_dict['tier_pair_weights'][idx, s, d] - state_dict['tier_pair_weights'][idx, 0, 0]
            results_dict['%d->%d' % (s,d)] = np.exp(float(weight))
        tier_self_weight = state_dict['tier_self_weights'][idx, 1] - state_dict['tier_self_weights'][idx, 0]
        results_dict['self'] = np.exp(float(tier_self_weight))
        if print_results:
            print(group, np.round([results_dict[k] for k in results_dict if k != 'group'], 4))
        results.append(results_dict)
    columns = list(results[-1].keys())
    df = pd.DataFrame(results, columns=columns)
    
    if plot_tier_pair_weights:
        fig, ax = plt.subplots(figsize=(8,12))
        bar_pos = np.arange(0, len(df)*3, 3)
        ax.barh(bar_pos+0.7, df['0->1'].values, align='center', label='0->1', height=0.7)
        ax.barh(bar_pos, df['1->0'].values, align='center', label='1->0', height=0.7)
        ax.barh(bar_pos-0.7, df['1->1'].values, align='center', label='1->1', height=0.7)
        ax.set_yticks(bar_pos)
        ax.set_yticklabels(df.group.values, fontsize=12)
        ax.set_xlabel('Multiplicative weight on visits\nrelative to 0->0', fontsize=12)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        ymin, ymax = ax.get_ylim()
        ax.vlines([1], ymin, ymax, color='grey')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.show()
    return df

def summarize_tier_pair_weights_over_multiple_models(exts, device, group2idx):
    ext2df = {}
    for ext in exts:
        ext2df[ext] = summarize_model_results(ext, device, group2idx, print_results=False, plot_tier_pair_weights=False)
    results = {'group': ext2df[exts[0]].group.values}
    for s,d in [(0, 1), (1, 0), (1, 1)]:
        vecs = []
        for ext, df in ext2df.items():
            vec = df['%d->%d' % (s, d)].values
            vecs.append(vec)
        vecs = np.array(vecs)
        mean = np.mean(vecs, axis=0)
        se = np.std(vecs, axis=0, ddof=1)
        results['%d->%d mean' % (s, d)] = mean
        results['%d->%d se' % (s, d)] = se 
        results['%d->%d lower CI' % (s, d)] = mean - (1.96 * se)
        results['%d->%d upper CI' % (s, d)] = mean + (1.96 * se)
    df = pd.DataFrame(results)
    return df 


def plot_tier_pair_weights(results_df, tier_pairs, sort_col, bar_height=10, xlim=None, labels=None):
    results_df = results_df.sort_values(by=sort_col)
    fig, ax = plt.subplots(figsize=(8,15))
    spacing = bar_height * (len(tier_pairs)+1.5)
    bar_pos = np.arange(0, len(results_df)*spacing, spacing)
    for i, tier_pair in enumerate(tier_pairs):
        x_label = '%d->%d mean' % tier_pair
        x = results_df[x_label].values - 1
        err = 1.96 * results_df['%d->%d se' % tier_pair].values
        y = bar_pos - (i*bar_height)
        label = labels[i] if labels is not None else None
        ax.barh(y, x, left=np.ones(len(x)), xerr=err, align='center', height=bar_height, alpha=0.8, 
                error_kw={'capsize':1.5, 'linewidth':1}, label=label)
        
    categories = [cu.TOPCATEGORIES_TO_SHORTER_NAMES[g] if g in cu.TOPCATEGORIES_TO_SHORTER_NAMES else g for g in results_df.group.values]
    ax.set_yticks(bar_pos - bar_height)
    ax.set_yticklabels(categories, fontsize=14)
    ax.set_xlabel('Percent change in visits relative to PP', fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
        xticks = np.arange(xlim[0], xlim[1]+0.01, 0.05)
        ax.set_xticks(xticks)
        percent_change = np.round((xticks - 1) * 100)
        ax.set_xticklabels(['%d%%' % d if d <= 0 else '+%d%%' % d for d in percent_change], fontsize=14)
    ymin, ymax = ax.get_ylim()
    ax.vlines([1], ymin, ymax, color='grey')
    ax.grid(alpha=0.3)
    if labels is not None:
        ax.legend(fontsize=12)

    
def plot_percent_change_in_weights(df1, df2, tier_pairs, bar_height=10, xlim=None, labels=None):
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
        xticks = np.arange(xlim[0], xlim[1]+0.01, 0.05)
        ax.set_xticks(xticks)
        percent_change = np.round(xticks * 100)
        ax.set_xticklabels(['%d%%' % d if d <= 0 else '+%d%%' % d for d in percent_change], fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)

    
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
    
def get_tier_relative_self_weight(state_dict, group_idx=None):
    if group_idx is None:
        return state_dict['tier_self_weights'][1] - state_dict['tier_self_weights'][0]
    return state_dict['tier_self_weights'][group_idx, 1] - state_dict['tier_self_weights'][group_idx, 0]
    
def get_z_source_weight(state_dict, small_county=False, stage=0):
    return state_dict['z_source_weights'][int(small_county), stage]

def get_z_dest_weight(state_dict, small_county=False, stage=0):
    return state_dict['z_dest_weights'][int(small_county), stage]

def get_z_self_weight(state_dict, small_county=False, stage=0):
    return state_dict['z_self_weights'][int(small_county), stage]

def get_poi_area_weight(state_dict):
    return state_dict['poi_weights.weight'][0, -1]

def recreate_mdl_and_dset_from_experiment(experiment, device, dset=None, set_final_mdl_weights=True):
    # get cfg and args
    save_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'outputs', '%s_cfg_args.pkl' % experiment)
    with open(save_path, 'rb') as f:
        cfg, args = pickle.load(f)
    
    # get dset and kwargs
    directory = os.path.join(cu.PATH_TO_CBG_POI_DATA, cfg.data.name, 'sampled_data', cfg.data.train_test_dir)
    with open(os.path.join(directory, 'kwargs.pkl'), 'rb') as f:
        dset_kwargs, sampling_kwargs = pickle.load(f)
    if dset is None:
        dset = CBGPOIDataset(**dset_kwargs)
    
    # get model
    mdl = PoissonRegModel(dset.FEATURE_DICT, control_tier=sampling_kwargs['control_tier'],
                          treatment_tier=sampling_kwargs['treatment_tier'],
                          zero_inflated=cfg.model.zero_inflated,
                          use_poi_cat_groups=cfg.data.use_poi_cat_groups)
    mdl = mdl.to(device)
    if set_final_mdl_weights:
        model_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', experiment)
        mdl.load_state_dict(t.load(model_fn, map_location=device))
    return mdl, dset
    
def evaluate_loss(dset, idx, mdls, device, batch_size=200000):
    """
    Evaluate model losses over a set of validation data points. This is a simplified version of the
    calculate_losses_over_data function in poisson_reg_model.py and calculates losses over 
    multiple models at once.
    """
    for mdl in mdls:
        mdl.eval()
    with t.no_grad():
        # prepare batches
        sampler = []  # fetch datapoints in batches; this is faster than getting datapoints individually then collating
        num_batches = math.ceil(len(idx) / batch_size)
        for i in range(num_batches):
            start_batch = i * batch_size
            end_batch = min(start_batch + batch_size, len(idx))
            sampler.append(idx[start_batch:end_batch])
        dl = DataLoader(dset, shuffle=False, sampler=sampler, collate_fn=collate_batch,
                        num_workers=16, pin_memory=True)     
        print('Found %d batches, batch size = %d' % (len(sampler), batch_size))

        total_loss = np.zeros(len(mdls))    
        avg_loss_per_batch = np.zeros((len(sampler), len(mdls)))
        labels = []
        mdl_preds = {i:[] for i in range(len(mdls))}  # model predictions
        for b, batch in enumerate(dl, start=0):
            batch_start = time.time()
            batch = [d.to(device) for d in batch]
            correction_terms = None  # no sampling or corrections in test data
            for i, mdl in enumerate(mdls):
                loss, pred = mdl.predict_and_compute_loss_on_data(batch, 
                                            correction_terms=correction_terms, return_pred=True)
                labels.append(batch[1].detach().numpy())  # batch[1] is visits
                mdl_preds[i].append(pred.detach().numpy())
                total_loss[i] += float(loss)
                avg_loss_per_batch[b, i] = float(loss) / len(batch[1])
            print('Batch %d: avg losses =' % b, np.round(avg_loss_per_batch[b], 4))
        return total_loss, avg_loss_per_batch, labels, mdl_preds

def compute_county_county_weights(dset, mdl):
    # all CBG attrs - leave out county tier and assignment variable
    all_cbg_attrs = np.zeros((dset.num_cbgs(), dset.FEATURE_DICT['cbg_num_attrs']))
    start_idx, end_idx = dset.FEATURE_DICT['cbg_static_attrs']
    all_cbg_attrs[:, start_idx:end_idx] = dset.cbg_attrs.values
    all_cbg_attrs[:, dset.FEATURE_DICT['cbg_device_ct']] = np.mean(dset.cbg_device_counts.values, axis=1)
    
    # all POI attrs - leave out county tier and assignment variable
    all_poi_attrs = np.zeros((dset.num_pois(), dset.FEATURE_DICT['poi_num_attrs']))
    subcats = dset.poi_subcat_indices
    all_poi_attrs[np.arange(dset.num_pois()), subcats] = 1  # one-hot
    start_idx, end_idx = dset.num_subcat_classes, dset.FEATURE_DICT['poi_static_attrs'][1]
    all_poi_attrs[:, start_idx:end_idx] = dset.poi_attrs.values[:, 2:].astype(float)
            
    # fill in weights
    adj_dict = helper.load_county_adjacency_dict()
    weights = np.zeros((dset.num_counties(), dset.num_counties(), dset.num_poi_groups()))
    for ci, fips_i in enumerate(dset.indices['counties']):
        cbgs = dset.county2cbgs[fips_i]
        cbg_attrs_i = all_cbg_attrs[cbgs]
        for fips_j in adj_dict[fips_i]:
            if fips_j in dset.county2idx:  # some neighboring counties are outside of CA
                cj = dset.county2idx[fips_j]
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
                for group_idx, group in enumerate(dset.poi_group_labels):
                    in_group = poi_groups == group_idx
                    weights[ci, cj, group_idx] = float(expected_visits[in_group].sum())
                print('%s -> %d CBGs; %s -> %d POIs; total weight = %.3f' % 
                      (fips_i, len(cbgs), fips_j, len(pois), np.sum(weights[ci, cj, :])))
    return weights

def generate_mobility_network(dset, same_mdl, adj_mdl, fips_i, tier_mode):
    assert tier_mode in ['control', 'treatment', 'direct', 'indirect']    
    # all CBG attrs - leave out county tier and assignment variable
    all_cbg_attrs = np.zeros((dset.num_cbgs(), dset.FEATURE_DICT['cbg_num_attrs']))
    start_idx, end_idx = dset.FEATURE_DICT['cbg_static_attrs']
    all_cbg_attrs[:, start_idx:end_idx] = dset.cbg_attrs.values
    all_cbg_attrs[:, dset.FEATURE_DICT['cbg_device_ct']] = np.mean(dset.cbg_device_counts.values, axis=1)
    
    # all POI attrs - leave out county tier and assignment variable
    all_poi_attrs = np.zeros((dset.num_pois(), dset.FEATURE_DICT['poi_num_attrs']))
    subcats = dset.poi_subcat_indices
    all_poi_attrs[np.arange(dset.num_pois()), subcats] = 1  # one-hot
    start_idx, end_idx = dset.num_subcat_classes, dset.FEATURE_DICT['poi_static_attrs'][1]
    all_poi_attrs[:, start_idx:end_idx] = dset.poi_attrs.values[:, 2:].astype(float)
    all_poi_attrs[:, dset.FEATURE_DICT['poi_group']] = dset.poi_group_indices
    
    cbgs_i = dset.county2cbgs[fips_i]
    pois_i = dset.county2pois[fips_i]
    if tier_mode == 'control':
        pass  # every CBG and POI has tier 0 already
    elif tier_mode == 'treatment':
        all_cbg_attrs[:, dset.FEATURE_DICT['cbg_tier']] = 1
        all_poi_attrs[:, dset.FEATURE_DICT['poi_tier']] = 1
    elif tier_model == 'direct':
        all_cbg_attrs[:, dset.FEATURE_DICT['cbg_tier']] = 0
        all_cbg_attrs[cbgs_i, dset.FEATURE_DICT['cbg_tier']] = 1
        all_poi_attrs[:, dset.FEATURE_DICT['poi_tier']] = 0
        all_poi_attrs[pois_i, dset.FEATURE_DICT['poi_tier']] = 0
    else:  # indirect 
        all_cbg_attrs[:, dset.FEATURE_DICT['cbg_tier']] = 1
        all_cbg_attrs[cbgs_i, dset.FEATURE_DICT['cbg_tier']] = 0
        all_poi_attrs[:, dset.FEATURE_DICT['poi_tier']] = 1
        all_poi_attrs[:, dset.FEATURE_DICT['poi_tier']] = 0
        
    nnz_cbgs = []
    nnz_pois = []
    nnz_visits = []
    adj_dict = helper.load_county_adjacency_dict()
    counties = [fips_i] + adj_dict[fips_i]
    region_cbgs = np.concatenate([dset.county2cbgs[ci] for ci in counties])
    region_pois = np.concatenate([dset.county2pois[ci] for ci in counties])
    for ci in counties:
        cbgs_i = dset.county2cbgs[ci]
        for cj in counties:
            pois_j = dset.county2pois[cj]
            same_county = ci == cj
            mdl = same_mdl if same_county else adj_mdl
            c_vec, p_vec, visits = generate_visits_for_cbgs_and_pois(
                all_cbg_attrs, all_poi_attrs, cbgs_i, pois_j, int(same_county), mdl)
            nonzero = visits > 0
            nnz_cbgs.append(c_vec[nonzero])
            nnz_pois.append(p_vec[nonzero])
            nnz_visits.append(visits[nonzero])
            nnz_prop = np.sum(nonzero) / (len(cbgs_i) * len(pois_j))
            print(ci, cj, nnz_prop)
    nnz_cbgs = np.concatenate(nnz_cbgs)
    nnz_pois = np.concatenate(nnz_pois)
    nnz_visits = np.concatenate(nnz_visits)
    return region_cbgs, region_pois, nnz_cbgs, nnz_pois, nnz_visits
    
    
def generate_visits_for_cbgs_and_pois(all_cbg_attrs, all_poi_attrs, cbgs, pois, same_county, mdl):
    c_vec = np.repeat(cbgs, len(pois))  # repeat each element
    p_vec = np.tile(pois, len(cbgs))  # repeat entire array
    visits = np.zeros(len(c_vec))
    distances = dset.get_cbg_poi_dists(c_vec, p_vec) 
    pis = mdl.get_pis(t.tensor(distances))
    sample_from_poisson = (np.random.binomial(1, pis)).astype(bool)
    
    # prepare relevant data in batch (no visits, assignment variable, etc)
    cbg_attrs = all_cbg_attrs[cbgs]
    poi_attrs = all_poi_attrs[pois]
    cbg_attrs_expanded = np.repeat(cbg_attrs, len(pois), axis=0)[sample_from_poisson]
    poi_attrs_expanded = np.tile(poi_attrs, (len(cbgs), 1))[sample_from_poisson]
    edge_attrs = np.zeros((np.sum(sample_from_poisson), dset.FEATURE_DICT['edge_num_attrs']))
    edge_attrs[:, dset.FEATURE_DICT['cbg_poi_dist']] = distances[sample_from_poisson]
    edge_attrs[:, 'same_county'] = same_county
    batch = (None, None, t.tensor(cbg_attrs_expanded), t.tensor(poi_attrs_expanded), 
             t.tensor(edge_attrs))
    lambdas = mdl.get_lambdas(batch)
    poisson_visits = np.random.poisson(lambdas)
    visits[sample_from_poisson] = poisson_visit
    return c_vec, p_vec, visits
        

def compute_within_county_weights(dset, mdl):
    # all CBG attrs - leave out county tier and assignment variable
    all_cbg_attrs = np.zeros((dset.num_cbgs(), dset.FEATURE_DICT['cbg_num_attrs']))
    start_idx, end_idx = dset.FEATURE_DICT['cbg_static_attrs']
    all_cbg_attrs[:, start_idx:end_idx] = dset.cbg_attrs.values
    all_cbg_attrs[:, dset.FEATURE_DICT['cbg_device_ct']] = np.mean(dset.cbg_device_counts.values, axis=1)
    
    # all POI attrs - leave out county tier and assignment variable
    all_poi_attrs = np.zeros((dset.num_pois(), dset.FEATURE_DICT['poi_num_attrs']))
    subcats = dset.poi_subcat_indices
    all_poi_attrs[np.arange(dset.num_pois()), subcats] = 1  # one-hot
    start_idx, end_idx = dset.num_subcat_classes, dset.FEATURE_DICT['poi_static_attrs'][1]
    all_poi_attrs[:, start_idx:end_idx] = dset.poi_attrs.values[:, 2:].astype(float)
            
    weights = np.zeros((dset.num_counties(), dset.num_poi_groups()))
    for ci, fips_i in enumerate(dset.indices['counties']):
        cbgs = dset.county2cbgs[fips_i]
        cbg_attrs_i = all_cbg_attrs[cbgs]
        pois = dset.county2pois[fips_i]
        poi_attrs_i = all_poi_attrs[pois]

        # prepare relevant data in batch (no visits, tier, assignment variable, etc)
        c_vec = np.repeat(cbgs, len(pois))  # repeat each element
        cbg_attrs_expanded = np.repeat(cbg_attrs_i, len(pois), axis=0)
        p_vec = np.tile(pois, len(cbgs))  # repeat entire array
        poi_attrs_expanded = np.tile(poi_attrs_i, (len(cbgs), 1))
        edge_attrs = np.zeros((len(cbgs)*len(pois), dset.FEATURE_DICT['edge_num_attrs']))
        distances = dset.get_cbg_poi_dists(c_vec, p_vec) 
        edge_attrs[:, dset.FEATURE_DICT['cbg_poi_dist']] = distances        
        batch = (None, None, t.tensor(cbg_attrs_expanded), t.tensor(poi_attrs_expanded), 
                 t.tensor(edge_attrs))

        lambdas = mdl.get_lambdas_without_tier_or_z(batch)
        pis = mdl.get_pis(t.tensor(distances))
        expected_visits = lambdas * pis
        poi_groups = dset.poi_group_indices[p_vec]
        for group_idx, group in enumerate(dset.poi_group_labels):
            in_group = poi_groups == group_idx
            weights[ci, group_idx] = float(expected_visits[in_group].sum())
        print('%s -> %d CBGs, %d POIs; total weight = %.3f' % 
              (fips_i, len(cbgs), len(pois), np.sum(weights[ci, :])))
    return weights


def compute_county_county_weights_in_parallel():
    """
    Run this to compute county county weights in parallel.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['inner', 'outer'], default='outer')
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--weights_fn', type=str)
    args = parser.parse_args()
    if args.mode == 'outer':  
        fns = ['test_poireg_t1t2_h5_adj_v0_2022_04_24_15_48_47',  # 2% negative sampling and nnz with replacement
               'test_poireg_t1t2_h5_adj_v1_2022_04_24_15_49_40',
               'test_poireg_t1t2_h5_adj_v2_2022_04_24_15_50_14',
               'test_poireg_t1t2_h5_adj_v3_2022_04_24_15_50_44',
               'test_poireg_t1t2_h5_adj_v4_2022_04_24_15_51_20',
               'test_poireg_t1t2_h5_adj_v5_2022_04_24_15_53_01',
               'test_poireg_t1t2_h5_adj_v6_2022_04_24_15_53_32',
               'test_poireg_t1t2_h5_adj_v7_2022_04_24_15_54_00',
               'test_poireg_t1t2_h5_adj_v8_2022_04_24_15_55_14',
               'test_poireg_t1t2_h5_adj_v9_2022_04_24_15_55_47']
        for i, experiment in enumerate(fns):
            out_str = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'outputs', 'county_county_weights_v%d' % i)
            cmd = f'nohup python -u results.py --mode inner --experiment {experiment} ' \
                f'--weights_fn county_county_weights_v{i}.pkl > {out_str}.out 2>&1 &'
            print(cmd)
            os.system(cmd)
    else:
        device = t.device('cpu')  # assume we're on cpu
        mdl, dset = recreate_mdl_and_dset_from_experiment(args.experiment, device, dset=None, set_final_mdl_weights=True)
        weights = compute_county_county_weights(dset, mdl)
        save_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'CA', args.weights_fn)
        with open(save_path, 'wb') as f:
            pickle.dump((dset.indices['counties'], dset.poi_group_labels, weights), f)
            
    
def compute_spillout_indices(dset, tier_self_weights, within_county_weights, 
                             tier_pair_weights, county_county_weights, normalize=True):
    assert tier_self_weights.shape == (dset.num_poi_groups(), 2)
    assert (tier_self_weights > 0).all()  # should be exponentiated already 
    assert within_county_weights.shape == (dset.num_counties(), dset.num_poi_groups())
    assert (tier_pair_weights > 0).all()
    assert tier_pair_weights.shape == (dset.num_poi_groups(), 2, 2)
    assert county_county_weights.shape == (dset.num_counties(), dset.num_counties(), dset.num_poi_groups())
    
    adj_dict = helper.load_county_adjacency_dict()
    spillout_indices = np.zeros(dset.num_counties())
    for ci, fips_i in enumerate(dset.indices['counties']):
        neighbors = [dset.county2idx[fips_j] for fips_j in adj_dict[fips_i] if fips_j in dset.county2idx]
        outflow = county_county_weights[ci, neighbors, :]  # num_neighbors x num_groups
        tier_diff = tier_pair_weights[:, 0, 1] - tier_pair_weights[:, 0, 0]
        num = np.sum(outflow * tier_diff)  # numerator 
        if normalize:
            denom = np.sum(outflow * tier_pair_weights[:, 0, 0])  # denominator
            denom += np.sum(within_county_weights[ci, :] * tier_self_weights[:, 0])  # E[Y_AA | T_A=0]
            spillout_indices[ci] = num / denom
        else:
            spillout_indices[ci] = num
    return spillout_indices

def compute_percent_reduction(dset, tier_self_weights, within_county_weights, tier_pair_weights, county_county_weights,
                              group_idx=None):
    assert tier_self_weights.shape == (dset.num_poi_groups(), 2)
    assert (tier_self_weights > 0).all()  # should be exponentiated already 
    assert within_county_weights.shape == (dset.num_counties(), dset.num_poi_groups())
    assert (tier_pair_weights > 0).all()
    assert tier_pair_weights.shape == (dset.num_poi_groups(), 2, 2)
    assert county_county_weights.shape == (dset.num_counties(), dset.num_counties(), dset.num_poi_groups())
    
    adj_dict = helper.load_county_adjacency_dict()
    percent_reductions = np.zeros(dset.num_counties())
    for ci, fips_i in enumerate(dset.indices['counties']):
        neighbors = [dset.county2idx[fips_j] for fips_j in adj_dict[fips_i] if fips_j in dset.county2idx]
        if group_idx is None:  # sum over groups
            outflow = county_county_weights[ci, neighbors, :]  # num_neighbors x num_groups
            num = np.sum(outflow * (tier_pair_weights[:, 1, 1] - tier_pair_weights[:, 0, 1]))
            num += np.sum(within_county_weights[ci, :] * (tier_self_weights[:, 1] - tier_self_weights[:, 0]))
            denom = np.sum(outflow * (tier_pair_weights[:, 1, 1] - tier_pair_weights[:, 0, 0]))
            denom += np.sum(within_county_weights[ci, :] * (tier_self_weights[:, 1] - tier_self_weights[:, 0]))
        else:
            outflow = county_county_weights[ci, neighbors, group_idx]  # num_neighbors
            num = np.sum(outflow * (tier_pair_weights[group_idx, 1, 1] - tier_pair_weights[group_idx, 0, 1]))
            num += np.sum(within_county_weights[ci, group_idx] * (tier_self_weights[group_idx, 1] - tier_self_weights[group_idx, 0]))
            denom = np.sum(outflow * (tier_pair_weights[group_idx, 1, 1] - tier_pair_weights[group_idx, 0, 0]))
            denom += np.sum(within_county_weights[ci, group_idx] * (tier_self_weights[group_idx, 1] - tier_self_weights[group_idx, 0]))
        percent_reductions[ci] = num / denom
    return percent_reductions

def compute_percent_reduction_macrocounty(dset, tier_self_weights, within_county_weights, tier_pair_weights, 
                                          county_county_weights, county2macrocounty, group_idx=None):
    assert tier_self_weights.shape == (dset.num_poi_groups(), 2)
    assert (tier_self_weights > 0).all()  # should be exponentiated already 
    assert within_county_weights.shape == (dset.num_counties(), dset.num_poi_groups())
    assert (tier_pair_weights > 0).all()
    assert tier_pair_weights.shape == (dset.num_poi_groups(), 2, 2)
    assert county_county_weights.shape == (dset.num_counties(), dset.num_counties(), dset.num_poi_groups())
    
    adj_dict = helper.load_county_adjacency_dict()
    percent_reductions = np.zeros(dset.num_counties())
    for ci, fips_i in enumerate(dset.indices['counties']):
        same_macro_neighbors = []  # neighbors in same macrocounty
        diff_macro_neighbors = []  # neighbors in different macrocounty
        macrocounty_i = county2macrocounty[fips_i]
        for fips_j in adj_dict[fips_i]:
            if fips_j in dset.county2idx:  # we don't have counties outside of California
                cj = dset.county2idx[fips_j]
                if county2macrocounty[fips_j] == macrocounty_i:
                    same_macro_neighbors.append(cj)
                else:
                    diff_macro_neighbors.append(cj)
        neighbors = same_macro_neighbors + diff_macro_neighbors
        if group_idx is None:  # sum over groups
            same_macro_outflow = county_county_weights[ci, same_macro_neighbors, :]  # num_neighbors x num_groups
            diff_macro_outflow = county_county_weights[ci, diff_macro_neighbors, :]
            num = np.sum(same_macro_outflow * (tier_pair_weights[:, 1, 1] - tier_pair_weights[:, 0, 0]))
            num += np.sum(diff_macro_outflow * (tier_pair_weights[:, 1, 1] - tier_pair_weights[:, 0, 1]))
            num += np.sum(within_county_weights[ci, :] * (tier_self_weights[:, 1] - tier_self_weights[:, 0]))
            outflow = county_county_weights[ci, neighbors, :]
            denom = np.sum(outflow * (tier_pair_weights[:, 1, 1] - tier_pair_weights[:, 0, 0]))
            denom += np.sum(within_county_weights[ci, :] * (tier_self_weights[:, 1] - tier_self_weights[:, 0]))
        else:
            same_macro_outflow = county_county_weights[ci, same_macro_neighbors, group_idx]  # num_neighbors
            diff_macro_outflow = county_county_weights[ci, diff_macro_neighbors, group_idx]
            num = np.sum(same_macro_outflow * (tier_pair_weights[group_idx, 1, 1] - tier_pair_weights[group_idx, 0, 0]))
            num += np.sum(diff_macro_outflow * (tier_pair_weights[group_idx, 1, 1] - tier_pair_weights[group_idx, 0, 1]))
            num += np.sum(within_county_weights[ci, group_idx] * (tier_self_weights[group_idx, 1] - tier_self_weights[group_idx, 0]))
            outflow = county_county_weights[ci, neighbors, group_idx]
            denom = np.sum(outflow * (tier_pair_weights[group_idx, 1, 1] - tier_pair_weights[group_idx, 0, 0]))
            denom += np.sum(within_county_weights[ci, group_idx] * (tier_self_weights[group_idx, 1] - tier_self_weights[group_idx, 0]))
        percent_reductions[ci] = num / denom
    return percent_reductions
    
def compute_kcut_cost(dset, tier_self_weights, within_county_weights, tier_pair_weights, 
                      county_county_weights, county2macrocounty):
    """
    Compute minimum k-cut objective. Use this to confirm that maximizing percent reduction and minimizing
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
            if fips_j in dset.county2idx:  # we don't have counties outside of California
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
    Make undirected county-county graph.
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
        directed_weights = nums / denom
        for cj, w in zip(neighbors, directed_weights):
            edge = tuple(sorted((ci, cj)))  # standardize ordering
            if edge in edge2weight:
                edge2weight[edge] = edge2weight[edge] + w
            else:
                edge2weight[edge] = w
    
    G = nx.Graph()
    for (ci, cj), w in edge2weight.items():
        G.add_edge(ci, cj, capacity=w)
        G.adj[ci][cj]['weight'] = int(10000 * w)  # weight needs to be integer for metis part_graph
    G.graph['edge_weight_attr'] = 'weight'
    return G 

    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to experiment config file.')
    parser.add_argument('--extension', type=str)
    parser.add_argument('--trial', type=int, default=None)
    return parser

def read_log_without_warnings(fn, print_lines=True):
    """
    Read logs without warnings.
    """
    full_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'outputs', fn)
    f = open(full_fn, 'r')
    lines = f.readlines()
    f.close()
    stripped_lines = []
    for l in lines:
        if not('pthreadpool-cpp' in l):
            stripped_lines.append(l.strip())
            if print_lines:
                print(l.strip())
    return stripped_lines

def get_model_results(ext, min_week=0, max_week=None):
    """
    Parses the saved results from model experiment. Returns the train_loss, test_loss,
    test_thresh (best), test_f1 (best), and test_auc over epochs. Test is not evaluated
    in every epoch, so the train_epochs and test_epochs are also returned. min_week and
    max_week (inclusive) controls which weeks we are aggregating losses/performance over.
    This is so that we can evaluate test performance on observed weeks with held-out data
    vs completely held-out weeks separately.
    """
    save_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_results', '%s.pkl' % ext)
    f = open(save_path, 'rb')
    results = pickle.load(f)
    f.close()
    
    num_epochs = len(results)
    num_weeks = len(results['epoch_0'])
    if max_week is None:
        max_week = num_weeks - 1
    num_weeks_kept = max_week - min_week + 1
    
    train_epochs = np.zeros(num_epochs)
    test_epochs = np.zeros(num_epochs)
    values = {'train_loss': np.zeros(num_epochs),
              'train_reg_loss': np.zeros(num_epochs),
              'test_loss': np.zeros(num_epochs),
              'test_threshold': np.zeros(num_epochs),
              'test_F1': np.zeros(num_epochs),
              'test_AUC': np.zeros(num_epochs)}
    
    # structure: ep_key -> week_key -> train_loss, test_loss, etc
    for ep in range(num_epochs):
        for week in range(min_week, max_week+1):
            week_results = results['epoch_%d' % ep]['week_%d' % week]
            if 'train_loss' in week_results:  # whether any train results were recorded for this epoch
                train_epochs[ep] = 1.
            if 'test_loss' in week_results:  # whether any test results were recorded for this epoch
                test_epochs[ep] = 1.
            for key, val in week_results.items():
                if key in values:
                    values[key][ep] += val
    
    # post-processing
    for key in values.keys():
        if key.startswith('train'):
            values[key] = values[key][train_epochs.astype(bool)]
        else:
            values[key] = values[key][test_epochs.astype(bool)]
        if not(key.endswith('loss')):
            values[key] = values[key] / num_weeks_kept  # want average, not sum, for these metrics
    values['train_epochs'] = np.arange(num_epochs)[train_epochs.astype(bool)]
    values['test_epochs'] = np.arange(num_epochs)[test_epochs.astype(bool)]
    return values 
    
def get_change_in_params_from_log(fn):
    log_lines = read_log_without_warnings(fn, print_lines=False)
    change_in_params = []
    # line in the form: "Change in model parameters = 0.614217"
    for l in log_lines:
        if l.startswith('Change in model parameters'):
            delta = l.split()[5] 
            change_in_params.append(float(delta))
    return np.array(change_in_params)
    
def plot_results_over_experiments(labels, exts, result_key, min_week=0, max_week=None, 
                                  first_ep=1, ax=None, make_legend=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    if result_key.startswith('train'):
        x_key = 'train_epochs'
    else:
        x_key = 'test_epochs'
    for lbl, ext in zip(labels, exts):
        result_dict = get_model_results(ext, min_week=min_week, max_week=max_week)
        x = result_dict[x_key]
        to_include = x >= first_ep
        x = x[to_include]
        y = result_dict[result_key][to_include]
        ax.plot(x, y, label=lbl)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(result_key, fontsize=12)
    ax.grid(alpha=0.5)
    if make_legend:
        ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
        
def load_model_weights(fn, device=None):
    """
    Load model weights
    """
    full_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', fn)
    state_dict = t.load(full_fn, map_location=device)
    return state_dict

def load_model_with_weights(cfg_fn, ext, device, dset=None):
    cfg = OmegaConf.load(os.path.join(cu.PATH_TO_CBG_POI_DATA, 'configs', cfg_fn))
    cfg = helper.fill_in_cfg_with_defaults(cfg, cu.cfg_field_reqs)
    if dset is None:
        directory = os.path.join(cu.PATH_TO_CBG_POI_DATA, cfg.data.name)
        dset = CBGPOIDataset(directory=directory, start_date=cfg.data.start_date, end_date=cfg.data.end_date,
                         load_dynamic=True)  # always load dynamic, and then we can turn it off later for static models    
    mdl = init_model(dset, cfg, device)
    weights = load_model_weights(ext, device=device)
    mdl.load_state_dict(weights)
    return mdl
    
def get_labels_and_model_pred(dset, test_idx, exts, mdls, device, max_sample=1e7):  
    """
    Evaluates models on test data. test_idx should be a list of length num_weeks.
    """
    assert len(exts) == len(mdls)
    for mdl in mdls:
        if hasattr(mdl, 'reset_embeddings'):  # not all models store dynamic embeddings
            mdl.reset_embeddings()  # reset embeddings to 0 since we don't have t-1 embeddings yet
    
    results = {ext:[] for ext in exts}
    results['labels'] = []
    for w, idx in enumerate(test_idx):  # go week-by-week, this is necessary for RNN    
        if len(idx) == 0:
            print('No data found in week %d -> skipping' % w)
        else:
            if len(idx) >= max_sample:
                idx = np.random.choice(idx, size=int(max_sample), replace=False)
            batch = dset.get_batch(idx.astype(int))
            batch = [d.to(device) for d in batch]
            y = batch[1]
            nonzero = y > 0
            if mdl.predict_binary:
                y = nonzero
            num_nnz = nonzero.sum()
            print('Week %d, %d datapoints, %d (%.2f%%) nonzeros' % (w, len(idx), num_nnz, 100. * num_nnz / len(idx)))
            results['labels'].append(y.detach().numpy())

            for ext, mdl in zip(exts, mdls):
                week_pred = mdl(batch).squeeze()
                results[ext].append(week_pred.detach().numpy())

                # update embeddings, potentially based on current attributes and/or previous messages
                if hasattr(mdl, 'update_embeddings'):
                    all_cbg_attrs = dset.get_all_dyn_cbg_attrs_for_week(w, include_device_count=True).float().to(device)
                    all_poi_attrs = dset.get_all_dyn_poi_attrs_for_week(w).float().to(device)
                    mdl.update_embeddings(all_cbg_attrs, all_poi_attrs)
                if hasattr(mdl, 'update_messages'):
                    mdl.update_messages(dset, w)  # update messages at the end of this week, so we don't have leakage
    return results
        
def get_best_threshold_for_fscore(labels, pred, make_plot=False):
    for i in range(len(labels)):  # week by week
        t, f = helper.get_best_threshold_and_fscore(labels[i], pred[i])
        print('Train, week %d: best threshold = %.3f, fscore = %.3f' % (i, t, f))
    all_labels = np.concatenate(labels)
    all_pred = np.concatenate(pred)
    prec_vec, rec_vec, thresholds = precision_recall_curve(all_labels, all_pred)
    fscore = (2 * prec_vec * rec_vec) / (prec_vec + rec_vec)
    idx = np.argmax(fscore)
    print('Train, overall: best threshold = %.3f, fscore = %.3f' % (thresholds[idx], fscore[idx]))
    
    if make_plot:
        plt.plot(prec_vec, rec_vec)
        plt.scatter(prec_vec[idx], rec_vec[idx], color='black')
        plt.xlabel('Precision')
        plt.ylabel('Recall')

def summarize_cbg_biases(week_dts, mdl=None, state_dict=None, demo_df=None, ax=None):
    if mdl is not None:
        cbg_biases = mdl.get_parameter('cbg_biases').cpu().detach().numpy()
    elif state_dict is not None:
        cbg_biases = state_dict['cbg_biases'].cpu().detach().numpy()
    else:
        raise Exception('Must provide either mdl or state_dict')
    
    # plot median/IQR of CBG bias over time
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    y = np.median(cbg_biases, axis=0)
    ax.plot_date(week_dts, y, linestyle='-', label='Median', color=CBG_COLOR)
    upper = np.percentile(cbg_biases, 75, axis=0)
    lower = np.percentile(cbg_biases, 25, axis=0)
    ax.fill_between(week_dts, lower, upper, alpha=0.3, color=CBG_COLOR)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xlabel('Week', fontsize=14)
    ax.set_ylabel('CBG bias', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.show()
    
    # avg bias per CBG
    avg_bias = np.mean(cbg_biases, axis=1)
    plt.hist(avg_bias, bins=25, color=CBG_COLOR)
    plt.grid(alpha=0.3)
    plt.title('Avg bias per CBG', fontsize=14)
    plt.show()
    if demo_df is not None:
        y = avg_bias
        for col in demo_df:
            x = demo_df[col].values
            valid = (~np.isnan(x)) & (~np.isnan(y)) 
            r, p = pearsonr(x[valid],y[valid])
            print('%s vs avg bias: r = %.4f, p = %.4f' % (col, r, p))
    
    # drop in bias per CBG
    first_part = np.mean(cbg_biases[:, :2], axis=1)
    second_part = np.mean(cbg_biases[:, 2:], axis=1)
    drop_in_bias = second_part - first_part
    plt.hist(drop_in_bias, bins=25, color=CBG_COLOR)
    plt.grid(alpha=0.3)
    plt.title('Drop in bias per CBG', fontsize=14)
    plt.show()
    if demo_df is not None:
        y = drop_in_bias
        for col in demo_df:
            x = demo_df[col].values
            valid = (~np.isnan(x)) & (~np.isnan(y)) 
            r, p = pearsonr(x[valid],y[valid])
            print('%s vs drop in bias: r = %.4f, p = %.4f' % (col, r, p))
            
def visualize_cbg_biases_on_map(mdl, cbg_geom, week_idx=None):
    cbg_biases = mdl.get_parameter('cbg_biases').cpu().detach().numpy()
    if week_idx is None:
        bias = np.mean(cbg_biases, axis=1)  # if week is not specified, take average over weeks
        title = 'CBG avg bias'
    else:
        bias = cbg_biases[:, week_idx]
        title = 'CBG bias from week %d' % week_idx
    cbg_geom['bias'] = bias
    fig, ax = plt.subplots(figsize=(20, 8))
    cbg_geom.plot(ax=ax, column='bias', cmap='RdBu', vmin=-5, vmax=-1, legend=True, alpha=0.7)
    ax.set_title(title, fontsize=14)
    plt.show()
            
        
def summarize_poi_biases(mdl, week_dts, core_df=None):
    poi_biases = mdl.get_parameter('poi_biases').cpu().detach().numpy()
    
    # plot median/IQR of POI bias over time
    fig, ax = plt.subplots(figsize=(6, 5))
    y = np.median(poi_biases, axis=0)
    ax.plot_date(week_dts, y, linestyle='-', label='Median', color=POI_COLOR)
    upper = np.percentile(poi_biases, 75, axis=0)
    lower = np.percentile(poi_biases, 25, axis=0)
    ax.fill_between(week_dts, lower, upper, alpha=0.3, color=POI_COLOR)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xlabel('Week', fontsize=14)
    ax.set_ylabel('POI bias', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.show()
    
    # avg bias per POI
    avg_bias = np.mean(poi_biases, axis=1)
    plt.hist(avg_bias, bins=25, color=POI_COLOR)
    plt.grid(alpha=0.3)
    plt.title('Avg bias per POI', fontsize=14)
    plt.show()
    if core_df is not None:
        core_df['avg_bias'] = avg_bias
        cats = []
        medians = []
        for cat, subdf in core_df.groupby('sub_category'):
            if len(subdf) >= 100:
                cats.append(cat)
                medians.append(np.median(subdf.avg_bias.values))
        order = [cats[i] for i in np.argsort(medians)]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(y='sub_category', x='avg_bias',
                    data=core_df, order=order,
                    ax=ax, fliersize=3)
        ax.set_ylabel("")
        ax.set_xlabel('POI\'s avg bias', fontsize=14)
        ax.tick_params(labelsize=12)
        plt.show()
    
    # drop in bias per POI
    first_part = np.mean(poi_biases[:, :2], axis=1)
    second_part = np.mean(poi_biases[:, 2:], axis=1)
    drop_in_bias = second_part - first_part
    plt.hist(drop_in_bias, bins=25, color=POI_COLOR)
    plt.grid(alpha=0.3)
    plt.title('Drop in bias per POI', fontsize=14)
    plt.show()
    if core_df is not None:
        core_df['drop_in_bias'] = drop_in_bias
        cats = []
        medians = []
        for cat, subdf in core_df.groupby('sub_category'):
            if len(subdf) >= 100:
                cats.append(cat)
                medians.append(np.median(subdf.drop_in_bias.values))
        order = [cats[i] for i in np.argsort(medians)]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(y='sub_category', x='drop_in_bias',
                    data=core_df, order=order,
                    ax=ax, fliersize=3)
        ax.set_ylabel("")
        ax.set_xlabel('POI\'s drop in bias', fontsize=14)
        ax.tick_params(labelsize=12)
        plt.show()
        
        
def summarize_distance_params(mdl, demo_df=None):
    dist_params = mdl.get_parameter('cbg_dist_params').cpu().detach().numpy()
    alphas = np.clip(dist_params[:, 0], 0, None)
    betas = np.clip(dist_params[:, 1], 0, None)
    at_5km = alphas * np.power(5000, -betas)
    at_10km = alphas * np.power(10000, -betas)
    dist_diff = at_5km - at_10km
    
    plt.figure(figsize=(7, 6))
    plt.scatter(alphas, betas, c=dist_diff, s=10, alpha=0.8)
    plt.xlabel('alpha', fontsize=12)
    plt.ylabel('beta', fontsize=12)
    plt.grid(alpha=0.3)
    plt.colorbar()
    plt.show()
    
    cbg_distance_curves = []
    test_dists = np.arange(500, 20001, 500)
    for d in test_dists:
        dist_term = alphas * np.power(d, -betas)
        cbg_distance_curves.append(dist_term)
    cbg_distance_curves = np.array(cbg_distance_curves).T
    
    plt.figure(figsize=(7, 6))
    for i in range(len(cbg_distance_curves)):
        plt.plot(np.array(test_dists) / 1000, cbg_distance_curves[i])
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Distance term', fontsize=12)
    plt.grid(alpha=0.2)
    plt.show()
    
    if demo_df is not None:
        y = dist_diff
        for col in demo_df:
            x = demo_df[col].values
            valid = (~np.isnan(x)) & (~np.isnan(y)) 
            r, p = pearsonr(x[valid],y[valid])
            print('%s vs 5km - 10km diff: r = %.4f, p = %.4f' % (col, r, p))
            

def analyze_cbg_preferences(state_dict, dset, demo_df=None, max_subcat_idx=40, version=2):
    idx2subcat = {i:c for c, i in dset.subcat2idx.items() if i > 0}
    
    assert version in {1, 2}
    cbg_attrs = dset.cbg_attrs.values  # n_cbgs x n_attrs
    if version == 1:
        cbg_attr_to_emb_weight = state_dict['cbg_attr_to_emb.weight'].cpu().detach().numpy()  # n_latent x n_cbg_attrs
        cbg_attr_to_emb_bias = state_dict['cbg_attr_to_emb.bias'].cpu().detach().numpy()  # n_latent
        cbg_embs = (cbg_attrs @ cbg_attr_to_emb_weight.T) + cbg_attr_to_emb_bias  # n_cbgs x n_latent
        cbg_emb_to_pref_weight = state_dict['cbg_emb_to_pref.weight'].cpu().detach().numpy()  # n_poi_attrs x n_latent
        cbg_emb_to_pref_bias = state_dict['cbg_emb_to_pref.bias'].cpu().detach().numpy()  # n_poi_attrs
        cbg_prefs = (cbg_embs @ cbg_emb_to_pref_weight.T) + cbg_emb_to_pref_bias  # n_cbgs x n_poi_attrs
    else:
        cbg_attr_to_pref_weight = state_dict['cbg_attr_to_pref.weight'].cpu().detach().numpy()  # n_poi_attrs x n_cbg_attrs
        cbg_attr_to_pref_bias = state_dict['cbg_attr_to_pref.bias'].cpu().detach().numpy()  # n_poi_attrs
        cbg_prefs = (cbg_attrs @ cbg_attr_to_pref_weight.T) + cbg_attr_to_pref_bias
    
    pref_means = np.mean(cbg_prefs, axis=1)  # n_cbgs
    plt.hist(pref_means, bins=25, color=CBG_COLOR)
    plt.grid(alpha=0.3)
    plt.title('Preference means per CBG', fontsize=14)
    plt.show()
    if demo_df is not None:
        assert len(demo_df) == len(cbg_prefs)
        y = pref_means
        for col in demo_df:
            x = demo_df[col].values
            valid = (~np.isnan(x)) & (~np.isnan(y)) 
            r, p = pearsonr(x[valid],y[valid])
            print('%s vs pref means: r = %.4f, p = %.4f' % (col, r, p))
    
    cbg_prefs_normalized = np.exp(cbg_prefs)
    row_sums = np.sum(cbg_prefs_normalized, axis=1)
    cbg_prefs_normalized = (cbg_prefs_normalized.T / row_sums).T  # do softmax per row to normalize per CBG
    plt.figure(figsize=(18,5))
    plt.title('CBG-POI preferences (softmax normalized per CBG)', fontsize=14)
    plt.imshow(cbg_prefs_normalized[:300].T)
    plt.xlabel('First 300 CBGs', fontsize=12)
    plt.ylabel('POI categories', fontsize=12)
    plt.colorbar()
    plt.show()
    
    if demo_df is not None:
        assert len(demo_df) == len(cbg_prefs_normalized)
        for col in demo_df:
            x = demo_df[col].values
            top_decile = x >= np.nanpercentile(x, 90) 
            top_avg_prefs = np.mean(cbg_prefs_normalized[top_decile], axis=0)
            bottom_decile = x <= np.nanpercentile(x, 10)
            bottom_avg_prefs = np.mean(cbg_prefs_normalized[bottom_decile], axis=0)
            pref_diff = top_avg_prefs - bottom_avg_prefs  # more positive means more favored by top
            
            print(col, '-> most favored by top decile')
            order = np.argsort(-pref_diff)  # order of POI attribute indices, from most to least favored by top decile
            top_subcats = ['%s (diff=%.2f)' % (cu.SUBCATEGORIES_TO_PRETTY_NAMES[idx2subcat[i]], pref_diff[i]) for i in order[:20] if i > 0 and i < max_subcat_idx and abs(pref_diff[i]) > 0.01]
            print('; '.join(top_subcats))
            
            print(col, '-> most favored by bottom decile')
            order = np.argsort(pref_diff)  # order of POI attribute indices, from most to least favored by bottom decile
            top_subcats = ['%s (diff=%.2f)' % (cu.SUBCATEGORIES_TO_PRETTY_NAMES[idx2subcat[i]], pref_diff[i]) for i in order[:20] if i > 0 and i < max_subcat_idx and abs(pref_diff[i]) > 0.01]
            print('; '.join(top_subcats))
            print()
    

def get_params_over_trials(experiment_name, num_trials, param_name, week=0):
    assert num_trials >= 1
    all_params = []
    use_week = False
    if num_trials > 1:
        for i in range(num_trials):
            state_dict = t.load('%s_%d.mdl' % (experiment_name, i), map_location=t.device('cpu'))
            params = state_dict[param_name].cpu().detach().numpy()
            if len(params.shape) > 1:
                params = params[:, week]
                use_week = True
            all_params.append(params.copy())
    else:
        state_dict = t.load('%s.mdl' % experiment_name, map_location=t.device('cpu'))
        params = state_dict[param_name].cpu().detach().numpy()
        if len(params.shape) > 1:
            params = params[:, week]
            use_week = True
        all_params.append(params.copy())
    all_params = np.array(all_params)
    return all_params, use_week

    
def summarize_params_over_trials(experiment_name, num_trials, param_name, week=0):
    all_params, use_week = get_params_over_trials(experiment_name, num_trials, param_name, week=week)
    variance_over_trials = np.var(all_params, axis=0)
    print('mean variance: %.3f' % np.mean(variance_over_trials))
    print('max variance: %.3f' % np.max(variance_over_trials))
    mean = np.mean(all_params, axis=0)
    upper = np.percentile(all_params, 75, axis=0)
    lower = np.percentile(all_params, 25, axis=0)
    order = np.argsort(mean)
    plt.plot(np.arange(len(mean)), mean[order], color='blue')
    plt.fill_between(np.arange(len(mean)), lower[order], upper[order], alpha=0.3, color='blue')
    title = param_name
    if use_week:
        title += ' [week=%d]' % week
    plt.title(title, fontsize=14)
    plt.show()
    
    
def compare_mean_params_from_two_experiments(name1, num_trials1, label1, 
                                             name2, num_trials2, label2, 
                                             param_name, week=0):
    params1, use_week = get_params_over_trials(name1, num_trials1, param_name, week=week)
    means1 = np.mean(params1, axis=0)
    params2, use_week = get_params_over_trials(name2, num_trials2, param_name, week=week)
    means2 = np.mean(params2, axis=0)
    
    order = np.argsort(means1)
    plt.figure(figsize=(8,8))
    plt.plot(means1[order], means1[order], alpha=0.5, color='grey')
    plt.scatter(means1[order], means2[order], alpha=0.7, s=10)
    title = param_name
    if use_week:
        title += ' [week=%d]' % week
    plt.title(title, fontsize=14)
    plt.xlabel(label1, fontsize=12)
    plt.ylabel(label2, fontsize=12)
    
    
if __name__ == '__main__':
    compute_county_county_weights_in_parallel()