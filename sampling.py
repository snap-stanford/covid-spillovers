import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import covid_constants_and_util as cu
import helper_methods_for_aggregate_data_analysis as helper
from dataset import *


def sample_train_test_indices(dset, dset_kwargs, control_tier, treatment_tier, save_dir, 
                              min_Z=-5, max_Z=5, cbg_treatment=None, poi_treatment=None,
                              train_ratio=cu.TRAIN_RATIO, existing_train_test_idx=None, 
                              neg_sample_rate=0.01, distance_based=True, dist_scalar=1,
                              num_versions=1, visit_type='both', only_keep_adjacent=False):
    """
    Filters and samples data. First filters data points to only keep counties that are "valid": compliers
    based on Z, in the corrent control/treatment tiers, and within the bandwidth (min_Z and max_Z). 
    Then, further filters based on desired visit type: only within county, only cross-county, or both; or 
    only adjacent counties.
    Then, splits filtered data points into train/test, then samples zero data points in train.
    If distance_based, zero data points in train are sampled with probability proportional to 1 / (1 + cd), 
    where d is distance between POI and CBG and c is dist_scalar. Otherwise, data points are sampled uniformly.
    Returns a list of (train_idx, test_idx) per week in the dataset, each list of indices is a numpy array. 
    The indices are in single-index (not wcp) form; see dataset.py for details. Also returns correction terms 
    as a dict mapping each negative sample in train to its correction term (inverse of sampling probability).
    """
    sampling_kwargs = {'control_tier': control_tier,
                       'treatment_tier': treatment_tier,
                       'min_Z': min_Z,
                       'max_Z': max_Z,
                       'cbg_treatment': cbg_treatment,
                       'poi_treatment': poi_treatment,
                       'train_ratio': train_ratio,
                       'neg_sample_rate': neg_sample_rate,
                       'distance_based': distance_based,
                       'dist_scalar': dist_scalar,
                       'visit_type': visit_type,
                       'only_keep_adjacent': only_keep_adjacent}
    with open(os.path.join(save_dir, 'kwargs.pkl'), 'wb') as f:
        pickle.dump((dset_kwargs, sampling_kwargs), f)
    
    # map county to list of CBGs / POIs
    assert dset.has_county_data
    assert (treatment_tier - control_tier) == 1  # must be adjacent
    county2cbg = {c:[] for c in dset.indices['counties']}
    county2poi = {c:[] for c in dset.indices['counties']}
    for i, cbg in enumerate(dset.indices['cbgs']):
        county = helper.extract_county_code_fr_fips(cbg)
        county2cbg[county].append(i)
    for j, poi_cbg in enumerate(dset.indices['poi_cbgs']):
        county = helper.extract_county_code_fr_fips(poi_cbg)
        county2poi[county].append(j) 
        
    adj_dict = helper.load_county_adjacency_dict()
    adj_mat = np.zeros((dset.num_counties(), dset.num_counties()))  # California county adjacency matrix
    for ct in adj_dict:
        if ct in dset.county2idx:
            neighbors = adj_dict[ct]
            for n in neighbors:
                if n in dset.county2idx:
                    adj_mat[dset.county2idx[ct], dset.county2idx[n]] = 1

    for w in dset.week_indices:  
        ts = time.time()
        print('=== WEEK %d (%s) ===' % (w, dset.indices['weeks'][w]))
        tier_dt = dset.county_tier_dates[w]
        weekday = tier_dt.weekday()
        # if past Wednesday, we can't use this for mobility that was measured Monday-Sunday
        if weekday > 2:
            print('Skipping data since tier date (%s) is past Wednesday' % tier_dt.strftime('%Y-%m-%d'))
            for v in range(num_versions):
                with open(os.path.join(save_dir, 'w%d_train_nnz.pkl' % w), 'wb') as f:
                    pickle.dump([], f)
                with open(os.path.join(save_dir, 'w%d_v%d_train_zero.pkl' % (w, v)), 'wb') as f:
                    pickle.dump(([], {}), f)
        else:
            # filter for data points that meet requirements: # 1) complier wrt assignment variable, 
            # 2) in control or treatment tier, 3) within bandwidth
            expected_assignment = (dset.county_assignment_vars[w] < 0).astype(int)
            actual_assignment = (dset.county_tiers[w] > control_tier).astype(int)
            complier = expected_assignment == actual_assignment
            in_control = dset.county_tiers[w] == control_tier
            in_treatment = dset.county_tiers[w] == treatment_tier
            within_bandwidth = (dset.county_assignment_vars[w] >= min_Z) & (dset.county_assignment_vars[w] <= max_Z)
            counties_to_keep = complier & (in_control | in_treatment) & within_bandwidth
            valid_counties = np.array(dset.indices['counties'])[counties_to_keep]
            treatment_vec = in_treatment.astype(int)[counties_to_keep]  # 1 if treatment, 0 if control
            
            cbgs_to_keep = []
            pois_to_keep = []
            for ct, trt in zip(valid_counties, treatment_vec):
                if cbg_treatment is None or cbg_treatment == trt:
                    cbgs_to_keep.extend(county2cbg[ct])
                if poi_treatment is None or poi_treatment == trt:                    
                    pois_to_keep.extend(county2poi[ct])
            print('Keeping %d counties (%d in control, %d in treatment), %d CBGs, %d POIs' % 
                  (len(valid_counties), np.sum(1-treatment_vec), np.sum(treatment_vec), 
                   len(cbgs_to_keep), len(pois_to_keep)))

            c_vec = np.repeat(cbgs_to_keep, len(pois_to_keep))  # repeat each element
            p_vec = np.tile(pois_to_keep, len(cbgs_to_keep))  # repeat entire array
            if visit_type != 'both':
                cbg_counties = dset.cbg_county_indices[c_vec]
                poi_counties = dset.poi_county_indices[p_vec]
                if visit_type == 'within':
                    to_keep = cbg_counties == poi_counties
                else:
                    assert visit_type == 'cross'
                    to_keep = cbg_counties != poi_counties
                c_vec = c_vec[to_keep]
                p_vec = p_vec[to_keep]
                print('Keeping %d/%d data points for %s county' % (len(c_vec), len(to_keep), visit_type))
            if only_keep_adjacent:
                assert visit_type != 'within'  # no overlap between within-county and adjacent visits
                cbg_counties = dset.cbg_county_indices[c_vec]
                poi_counties = dset.poi_county_indices[p_vec]
                adj_county = adj_mat[cbg_counties, poi_counties].astype(bool)
                c_vec = c_vec[adj_county]
                p_vec = p_vec[adj_county]
                print('Keeping %d/%d data points for adjacent counties' % (len(c_vec), len(adj_county)))
            all_idx = dset.wcp_to_index(np.ones(len(p_vec)) * w, c_vec, p_vec)
            
            num_train = int(len(all_idx) * train_ratio)
            if existing_train_test_idx is None:
                np.random.shuffle(all_idx)
                train_idx = all_idx[:num_train]
                test_idx = all_idx[num_train:]  # put val and test in test_idx for now, to simplify code logic
                print('Shuffled and split all valid data for week')
            else:
                # want to keep the same val+test as before, just resample train
                test_idx = np.concatenate([existing_train_test_idx[w][1], existing_train_test_idx[w][2]])
                in_train = ~np.isin(all_idx, test_idx)
                train_idx = all_idx[in_train]
                assert len(train_idx) == num_train
                print('Split into existing train/test for week')
            
            nnz_idx = dset.get_nonzero_indices_for_week(w, as_wcp=False)
            train_is_nnz = np.isin(train_idx, nnz_idx, assume_unique=True)  # nonzero data points in train
            train_nnz = train_idx[train_is_nnz]
            train_zero = train_idx[~train_is_nnz]
            print('Separated zero and nonzero data points in train')
            labels = ['train_nnz', 'val', 'test']
            num_val = int(len(test_idx) / 2)
            idx_groups = [train_nnz, test_idx[:num_val], test_idx[num_val:]]
            for s, idx in zip(labels, idx_groups):  # save train nonzero, val, and test for week
                if len(idx) > 0:  # don't save val or test if train ratio = 1
                    with open(os.path.join(save_dir, 'w%d_%s.pkl' % (w, s)), 'wb') as f:
                        pickle.dump(idx, f)

            # compute sampling probabilities for zero data points in train based on CBG-POI distances
            if distance_based:
                _, zc, zp = dset.index_to_wcp(train_zero)
                zdists = dset.get_cbg_poi_dists(zc, zp)  # distances for zero data points in train
                probs = 1 / (1+(zdists * dist_scalar))  # probability of sampling is inversely proportional to scaled distance 
                expected_total = len(train_nnz) if neg_sample_rate == 'nonzero' else int(neg_sample_rate * len(train_zero))
                probs = (expected_total * probs) / np.sum(probs)  # normalize to sum to expected_total
                to_clip = np.sum(probs > 1)
                if to_clip > 0:
                    print('Warning: clipping probabilities for %d data points to 1' % to_clip)
                    probs = np.clip(probs, None, 1)
            else:
                probs = np.ones(len(train_zero)) * neg_sample_rate  # uniform probability of sampling
            corrections = 1 / probs
            
            num_zero_sampled = []
            for v in range(num_versions):
                to_keep = np.random.binomial(1, probs).astype(bool) 
                train_zero_v = train_zero[to_keep]
                num_zero_sampled.append(len(train_zero_v))
                # update corrections dictionary to contain corrections *only for sampled zero data points*
                sample2correction = dict(zip(train_zero_v, corrections[to_keep]))
                with open(os.path.join(save_dir, 'w%d_v%d_train_zero.pkl' % (w, v)), 'wb') as f:
                    pickle.dump((train_zero_v, sample2correction), f)
            
            print('Finished sampling! Added %d nonzero, %s zero and %d test data points' % 
                  (len(train_nnz), num_zero_sampled, len(test_idx)))
        print('Finished week [time = %.3fs]\n' % (time.time() - ts))

def sample_from_nnz_with_replacement(dset, save_dir, num_versions, num_weeks):
    """
    Sample from nonzero data points with replacement.
    """
    all_train_nnz = []
    for w in range(num_weeks):
        with open(os.path.join(save_dir, 'w%d_train_nnz.pkl' % w), 'rb') as f:
            train_nnz = pickle.load(f)
        all_train_nnz.append(train_nnz)
    all_train_nnz = np.concatenate(all_train_nnz)
    print('Found %d train nnz points' % len(all_train_nnz))
    
    for v in range(num_versions):
        sample = np.random.choice(all_train_nnz, size=len(all_train_nnz), replace=True)
        w_vec, _, _ = dset.index_to_wcp(sample)
        lengths = []
        for w in range(num_weeks):
            week_sample = sample[w_vec == w]
            lengths.append(len(week_sample))
            with open(os.path.join(save_dir, 'w%d_v%d_train_nnz.pkl' % (w, v)), 'wb') as f:
                pickle.dump(week_sample, f)
        print('Finished v%d: lengths =' % v, lengths)
    
def save_evaluation_data_points(dset, control_tier, treatment_tier, save_dir, epsilon):
    """
    Get all data points within epsilon of 0.
    """
    # map county to list of CBGs / POIs
    assert dset.has_county_data
    assert (treatment_tier - control_tier) == 1  # must be adjacent
    county2cbg = {c:[] for c in dset.indices['counties']}
    county2poi = {c:[] for c in dset.indices['counties']}
    for i, cbg in enumerate(dset.indices['cbgs']):
        county = helper.extract_county_code_fr_fips(cbg)
        county2cbg[county].append(i)
    for j, poi_cbg in enumerate(dset.indices['poi_cbgs']):
        county = helper.extract_county_code_fr_fips(poi_cbg)
        county2poi[county].append(j) 

    all_eval = []
    for w in dset.week_indices:  
        ts = time.time()
        print('=== WEEK %d (%s) ===' % (w, dset.indices['weeks'][w]))
        tier_dt = dset.county_tier_dates[w]
        weekday = tier_dt.weekday()
        # if past Wednesday, we can't use this for mobility that was measured Monday-Sunday
        if weekday > 2:
            print('Skipping data since tier date (%s) is past Wednesday' % tier_dt.strftime('%Y-%m-%d'))
            all_eval.append([])
        else:
            # filter for data points that meet requirements: # 1) complier wrt assignment variable, 
            # 2) in control or treatment tier, 3) within epsilon
            expected_assignment = (dset.county_assignment_vars[w] < 0).astype(int)
            actual_assignment = (dset.county_tiers[w] > control_tier).astype(int)
            complier = expected_assignment == actual_assignment
            in_control = dset.county_tiers[w] == control_tier
            in_treatment = dset.county_tiers[w] == treatment_tier
            within_bandwidth = (dset.county_assignment_vars[w] >= -epsilon) & (dset.county_assignment_vars[w] <= epsilon)
            counties_to_keep = complier & (in_control | in_treatment) & within_bandwidth
            valid_counties = np.array(dset.indices['counties'])[counties_to_keep]
            print('Valid counties')
            print(valid_counties)
            treatment_vec = in_treatment.astype(int)[counties_to_keep]  # 1 if treatment, 0 if control
            
            cbgs_to_keep = []
            pois_to_keep = []
            for ct in valid_counties:
                cbgs_to_keep.extend(county2cbg[ct])
                pois_to_keep.extend(county2poi[ct])
            print('Keeping %d counties (%d in control, %d in treatment), %d CBGs, %d POIs, %d data points' % 
                  (len(valid_counties), np.sum(1-treatment_vec), np.sum(treatment_vec), 
                   len(cbgs_to_keep), len(pois_to_keep), len(cbgs_to_keep) * len(pois_to_keep)))
            c_vec = np.repeat(cbgs_to_keep, len(pois_to_keep))  # repeat each element
            p_vec = np.tile(pois_to_keep, len(cbgs_to_keep))  # repeat entire array
            all_idx = dset.wcp_to_index(np.ones(len(p_vec)) * w, c_vec, p_vec)
            all_eval.append(all_idx)
    with open(os.path.join(save_dir, 'eval_ep%s.pkl' % epsilon), 'wb') as f:
        pickle.dump(all_eval, f)
        
    
def load_train_test_indices_from_cfg(cfg):
    """
    Load saved train/test indices from config. Checks if saved parameters match config parameters.
    """
    directory = os.path.join(cu.PATH_TO_CBG_POI_DATA, cfg.data.name, 'sampled_data', cfg.data.train_test_dir)
    version = cfg.data.neg_sample_version
    test_set = cfg.test.test_set if cfg.test.test_set != 'none' else None
    dset_kwargs, sampling_kwargs, train_test_idx, correction_terms = load_train_test_indices(directory, version, test_set=test_set, use_sampled_nnz=cfg.data.use_sampled_nnz)
    assert dset_kwargs['directory'].endswith(cfg.data.name)  # indices are only valid if they apply to the same dataset
    assert dset_kwargs['start_date'] == cfg.data.start_date  # the filtered weeks must be the same as well
    assert dset_kwargs['end_date'] == cfg.data.end_date
    return dset_kwargs, sampling_kwargs, train_test_idx, correction_terms

def load_train_test_indices(directory, version, test_set=None, weeks=None, use_sampled_nnz=False):
    """
    Load saved train/test indices from directory for given version of negative samples.
    """
    print('Loading sampled data with negative sample version=%d (using sampled nnz=%s) and test set=%s' % 
          (version, use_sampled_nnz, test_set))
    with open(os.path.join(directory, 'kwargs.pkl'), 'rb') as f:
        dset_kwargs, sampling_kwargs = pickle.load(f)
    train_test_idx = []
    correction_terms = {}
    if weeks is None:
        weeks = sorted(list(set([int(fn[1]) for fn in os.listdir(directory) if fn.startswith('w')])))
    for w in weeks:
        if use_sampled_nnz:
            with open(os.path.join(directory, 'w%d_v%d_train_nnz.pkl' % (w, version)), 'rb') as f:
                train_nnz = pickle.load(f)
        else:
            with open(os.path.join(directory, 'w%d_train_nnz.pkl' % w), 'rb') as f:
                train_nnz = pickle.load(f)
        with open(os.path.join(directory, 'w%d_v%d_train_zero.pkl' % (w, version)), 'rb') as f:
            train_zero, corrections = pickle.load(f)
            assert len(train_zero) == len(corrections)
        if test_set is None:
            test = []
        else:
            assert test_set in {'val', 'test'}
            with open(os.path.join(directory, 'w%d_%s.pkl' % (w, test_set)), 'rb') as f:
                test = pickle.load(f)
        train_test_idx.append((np.concatenate([train_nnz, train_zero]), test))
        correction_terms = {**correction_terms, **corrections}
        print('Loaded sample for week %d' % w)
    return dset_kwargs, sampling_kwargs, train_test_idx, correction_terms

def plot_cbg_frequencies_in_train_test(train_test_idx, w, dset):
    """
    Plots a histogram of CBG frequencies in train, train nonzero, train zero,
    and test, test nonzero, and test zero.
    """
    def _plot_cbg_frequencies_from_indices(idx, ax, title):
        _, c_vec, _ = dset.index_to_wcp(idx)
        counter = Counter(c_vec)
        counts = [counter[c] if c in counter else 0 for c in np.arange(dset.num_cbgs())]
        ax.hist(counts, bins=30)
        ax.set_xlabel(title)
    
    train_idx, test_idx = train_test_idx[w]
    nnz = dset.get_nonzero_indices_for_week(w, as_wcp=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    _plot_cbg_frequencies_from_indices(train_idx, axes[0,0], 'CBG counts in train')
    nnz_train = np.array(list(set(train_idx).intersection(set(nnz))))
    _plot_cbg_frequencies_from_indices(nnz_train, axes[0,1], 'CBG nonzero counts in train')
    zero_train = np.array(list(set(train_idx) - set(nnz)))
    _plot_cbg_frequencies_from_indices(zero_train, axes[0,2], 'CBG zero counts in train')
    
    if len(test_idx) > 0:
        _plot_cbg_frequencies_from_indices(test_idx, axes[1,0], 'CBG counts in test')
        nnz_test = np.array(list(set(test_idx).intersection(set(nnz))))
        _plot_cbg_frequencies_from_indices(nnz_test, axes[1,1], 'CBG nonzero counts in test')
        zero_test = np.array(list(set(test_idx) - set(nnz)))
        _plot_cbg_frequencies_from_indices(zero_test, axes[1,2], 'CBG zero counts in test')

    plt.show()

def plot_distances_from_indices(idx, dset, ax, title):
    _, c_vec, p_vec = dset.index_to_wcp(idx)
    distances = dset.get_cbg_poi_dists(c_vec, p_vec)
    ax.hist(distances, bins=30)
    ax.set_xlabel(title)
    return distances
        
def plot_distances_in_train(directory, week, dset, version=0):
    """
    Plots a histogram of distances in train nonzero and zero.
    """    
    with open(os.path.join(directory, 'w%d_train_nnz.pkl' % week), 'rb') as f:
        train_nnz = pickle.load(f)
    with open(os.path.join(directory, 'w%d_v%d_train_zero.pkl' % (week, version)), 'rb') as f:
        train_zero, corrections = pickle.load(f)
        assert len(train_zero) == len(corrections)
    print('Num nonzero: %d. Num zero: %d.' % (len(train_nnz), len(train_zero)))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dist_nnz = plot_distances_from_indices(train_nnz, dset, axes[0], 'Distances in nonzero train')
    dist_zero = plot_distances_from_indices(train_zero, dset, axes[1], 'Distances in zero train')
    plt.show()
    return train_nnz, dist_nnz, train_zero, dist_zero

    
def plot_distances_in_train_test(train_test_idx, w, dset):
    """
    Plots a histogram of distances in train, train nonzero, train zero,
    and test, test nonzero, and test zero.
    """    
    train_idx, test_idx = train_test_idx[w]
    nnz_idx = dset.get_nonzero_indices_for_week(w, as_wcp=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    train_is_nnz = np.isin(train_idx, nnz_idx, assume_unique=True)  # nonzero data points in train
    train_nnz = train_idx[train_is_nnz]
    train_zero = train_idx[~train_is_nnz]
    plot_distances_from_indices(train_nnz, dset, axes[0,0], 'Distances in nonzero train')
    plot_distances_from_indices(train_zero, dset, axes[0,1], 'Distances in zero train')
    
    if len(test_idx) > 0:
        test_is_nnz = np.isin(test_idx, nnz_idx, assume_unique=True)  # nonzero data points in test
        test_nnz = test_idx[test_is_nnz]
        test_zero = test_idx[~test_is_nnz]
        plot_distances_from_indices(test_nnz, dset, axes[1,0], 'Distances in nonzero test')
        plot_distances_from_indices(test_zero, dset, axes[1,1], 'Distances in zero test')

    plt.show()
    
def get_county_pair_counts_per_tier_pair_type(train_idx, dset, control_tier, treatment_tier):
    counters = {}
    for source_tier in [control_tier, treatment_tier]:
        for dest_tier in [control_tier, treatment_tier]:
            for s in ['zero', 'nonzero']:
                counters[(source_tier, dest_tier, s)] = Counter([])         
                
    for w, idx in enumerate(train_idx):
        print('Num data points in train in week %s:' % dset.indices['weeks'][w], len(idx))
        if len(idx) == 0:
            print('No data -> skipping!')
        else:
            _, c_vec, p_vec = dset.index_to_wcp(idx)
            cbg_counties = dset.cbg_county_indices[c_vec]
            poi_counties = dset.poi_county_indices[p_vec]
            diff_county = (cbg_counties != poi_counties)  # only want to keep data from different counties
            c_vec, p_vec = c_vec[diff_county], p_vec[diff_county]
            cbg_counties = cbg_counties[diff_county]
            poi_counties = poi_counties[diff_county]
            visits = np.array([dset.visits[w][c,p] for c, p in zip(c_vec, p_vec)])
            nonzero = visits > 0
            print('%d cross-county data points -> %d nonzero, %d zero' % (len(c_vec), np.sum(nonzero), np.sum(~nonzero)))   

            tiers = dset.county_tiers[w]
            cbg_tiers = tiers[cbg_counties]
            poi_tiers = tiers[poi_counties]
            total_matches = 0
            # we want list of county pairs that fit this tier pair type in this week
            for source_tier in [control_tier, treatment_tier]:
                for dest_tier in [control_tier, treatment_tier]:
                    tier_match = (cbg_tiers == source_tier) & (poi_tiers == dest_tier)
                    total_matches += np.sum(tier_match)
                    tier_match_nnz = tier_match & nonzero
                    tier_match_zero = tier_match & (~nonzero)
                    print('(%s, %s): %d data points matching tiers -> %d nonzero, %d zero' % 
                          (source_tier, dest_tier, np.sum(tier_match), np.sum(tier_match_nnz), np.sum(tier_match_zero)))
                    tier_pairs_nnz = list(zip(cbg_counties[tier_match_nnz], poi_counties[tier_match_nnz]))
                    counters[(source_tier, dest_tier, 'nonzero')] = counters[(source_tier, dest_tier, 'nonzero')] + Counter(tier_pairs_nnz)
                    tier_pairs_zero = list(zip(cbg_counties[tier_match_zero], poi_counties[tier_match_zero]))
                    counters[(source_tier, dest_tier, 'zero')] = counters[(source_tier, dest_tier, 'zero')] + Counter(tier_pairs_zero)
            assert total_matches == len(c_vec)  # all data points should be in one of these 4 tier pair types 
        print()
    return counters

def main_inner(args):
    """
    Executes sampling process based on command line args.
    """
    ca_dir = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'CA')
    county_data_fn = os.path.join(ca_dir, 'county_dynamic_attrs_2021_t1t2.pkl')
    data_kwargs = {'directory': ca_dir,
                   'start_date': '2021-02-01',
                   'end_date': '2021-03-29',
                   'path_to_county_data': county_data_fn,
                   'cbg_count_min': 50,
                   'poi_count_min': 30,
                   'load_distances_dynamically': True}
    dset = CBGPOIDataset(**data_kwargs)
    
    save_dir = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'CA', 'sampled_data', args.dir)    
    assert args.bandwidth > 0
    sample_train_test_indices(dset, data_kwargs, control_tier=1, treatment_tier=2, 
                              save_dir=save_dir, min_Z=-args.bandwidth, max_Z=args.bandwidth, 
                              cbg_treatment=args.cbg_treatment, poi_treatment=args.poi_treatment,
                              train_ratio=1, neg_sample_rate=args.rate, 
                              distance_based=args.distance_based, dist_scalar=1,
                              num_versions=args.num_versions, visit_type=args.visit_type, 
                              only_keep_adjacent=args.only_adj)


def main_outer(args):
    save_dir = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'CA', 'sampled_data', args.dir)    
    assert os.path.isdir(save_dir)
    cmd = 'nohup python -u sampling.py --mode inner'
    for arg in vars(args):
        if arg != 'mode' and getattr(args, arg) is not None:
            cmd += ' --%s %s' % (arg, getattr(args, arg))  # inherit arguments from command line
    out_str = os.path.join(save_dir, 'sampling_log')
    cmd += ' > %s.out 2>&1 & ' % out_str
    print('Command:', cmd)
    os.system(cmd)
    
def do_bandwidth_sampling(args):
    """
    Given a bandwidth, separately sample data in the 4 pairwise treatment regions.
    """
    for cbg_treatment, poi_treatment in [(0,0), (0,1), (1,0), (1,1)]:
        region_dir = '%s_cbg%d_poi%d' % (args.dir, cbg_treatment, poi_treatment)
        full_dir = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'CA', 'sampled_data', region_dir)
        assert os.path.isdir(full_dir)
        out_str = os.path.join(full_dir, 'sampling_log')
        cmd = f'nohup python -u sampling.py --dir {region_dir} --mode inner --bandwidth {args.bandwidth} ' \
                f'--cbg_treatment {cbg_treatment} --poi_treatment {poi_treatment} --num_versions {args.num_versions} ' \
                f'--rate {args.rate} > {out_str}.out 2>&1 &'
        print('Command:', cmd)
        os.system(cmd)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--mode', choices=['inner', 'outer'], default='outer')
    parser.add_argument('--bandwidth', type=float, default=5)
    parser.add_argument('--cbg_treatment', type=int, choices=[0, 1])
    parser.add_argument('--poi_treatment', type=int, choices=[0, 1])
    parser.add_argument('--rate', type=float, default=0.01)
    parser.add_argument('--distance_based', type=int, choices=[0, 1], default=1)
    parser.add_argument('--num_versions', type=int, default=1)
    parser.add_argument('--visit_type', type=str, choices=['within', 'cross', 'both'], default='both')
    parser.add_argument('--only_adj', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    save_dir = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'CA', args.dir)
    
    if args.mode == 'outer':
        # do_bandwidth_sampling(args)
        main_outer(args)
    else:
        main_inner(args)