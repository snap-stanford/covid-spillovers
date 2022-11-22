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
                              num_versions=1, visit_type='all'):
    """
    Filters and samples data.
    1) Filters data points to only keep counties that are "valid": compliers based on Z, in the current 
    control/treatment tiers, and within the bandwidth (min_Z and max_Z). 
    2) Filters based on desired visit type: only adjacent counties, only cross-county, only within county, or all.
    3) Optional: splits filtered data points into train/val/test (if train_ratio < 1). Otherwise all datapoints 
    are used for training.
    4) Samples zero data points in train. If distance_based, zero data points in train are sampled with 
    probability proportional to 1 / (1 + cd), where d is distance between POI and CBG and c is dist_scalar. 
    Otherwise, data points are sampled uniformly.
    Returns a list of (train_idx, test_idx) per week in the dataset, each list of indices is a numpy array. 
    The indices are in single-index (not wcp) form; see dataset.py for details. Also returns correction terms 
    as a dict mapping each negative sample in train to its correction term (inverse of sampling probability).
    """
    assert dset.has_county_data
    assert (treatment_tier - control_tier) == 1  # must be consecutive
    assert visit_type in {'all', 'within', 'cross', 'adj'}
    assert train_ratio > 0 and train_ratio <= 1 
    
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
                       'visit_type': visit_type}
    with open(os.path.join(save_dir, 'kwargs.pkl'), 'wb') as f:
        pickle.dump((dset_kwargs, sampling_kwargs), f)
        
    # construct adjacency matrix for California counties
    adj_dict = helper.load_county_adjacency_dict()  # leaves out self-loop
    adj_mat = np.zeros((dset.num_counties(), dset.num_counties()))
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
            # filter for data points that meet requirements: 1) complier wrt assignment variable, 
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
                if cbg_treatment is None or cbg_treatment == trt:  # can require CBGs to be in treatment/control
                    cbgs_to_keep.extend(dset.county2cbgs[ct])
                if poi_treatment is None or poi_treatment == trt:  # can require POIs to be in treatment/control                 
                    pois_to_keep.extend(dset.county2pois[ct])
            print('Keeping %d counties (%d in control, %d in treatment), %d CBGs, %d POIs' % 
                  (len(valid_counties), np.sum(1-treatment_vec), np.sum(treatment_vec), 
                   len(cbgs_to_keep), len(pois_to_keep)))
            c_vec = np.repeat(cbgs_to_keep, len(pois_to_keep))  # repeat each element
            p_vec = np.tile(pois_to_keep, len(cbgs_to_keep))  # repeat entire array
            
            # filter on visit type
            if visit_type != 'all':
                cbg_counties = dset.cbg_county_indices[c_vec]
                poi_counties = dset.poi_county_indices[p_vec]
                if visit_type == 'within':
                    to_keep = cbg_counties == poi_counties
                elif visit_type == 'cross':
                    to_keep = cbg_counties != poi_counties
                else:
                    assert visit_type == 'adj'
                    to_keep = adj_mat[cbg_counties, poi_counties].astype(bool)
                c_vec = c_vec[to_keep]
                p_vec = p_vec[to_keep]
                print('Keeping %d/%d data points for %s county' % (len(c_vec), len(to_keep), visit_type))
            all_idx = dset.wcp_to_index(np.ones(len(p_vec)) * w, c_vec, p_vec)
            
            # split into train, validation, and test, if necessary
            if train_ratio < 1:
                num_train = int(len(all_idx) * train_ratio)
                num_heldout = len(all_idx) - num_train
                num_val = int(num_heldout / 2)
                num_test = num_heldout - num_val
                if existing_train_test_idx is None:
                    # make new train/val/test split, save val and test
                    np.random.shuffle(all_idx)
                    train_idx = all_idx[:num_train]
                    val_idx = all_idx[num_train:num_train+num_val]
                    test_idx = all_idx[num_train+num_val:]
                    with open(os.path.join(save_dir, 'w%d_val.pkl' % w), 'wb') as f:
                        pickle.dump(val_idx, f)
                    with open(os.path.join(save_dir, 'w%d_test.pkl' % w), 'wb') as f:
                        pickle.dump(test_idx, f)
                    print('Shuffled and split all valid data for week')
                else:
                    # want to keep the same val+test as before, just resample train
                    val_idx, test_idx = existing_train_test_idx[w][1], existing_train_test_idx[w][2]
                    in_train = ~np.isin(all_idx, test_idx)  # existing train test idx has sampled train, not full train, saved
                    train_idx = all_idx[in_train]
                    assert len(train_idx) == num_train
                    assert len(val_idx) == num_val
                    assert len(test_idx) == num_test
                    print('Split into existing train/val/test for week')
            else:
                assert existing_train_test_idx is None
                train_idx = all_idx
                num_heldout = 0
            
            # compute sampling probabilities for zero data points in train based on CBG-POI distances
            nnz_idx = dset.get_nonzero_indices_for_week(w, as_wcp=False)
            train_is_nnz = np.isin(train_idx, nnz_idx, assume_unique=True)  # nonzero data points in train
            train_nnz = train_idx[train_is_nnz]
            train_zero = train_idx[~train_is_nnz]
            print('Separated zero and nonzero data points in train')                    
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
            
            # repeat negative sampling for different random seeds
            num_zero_sampled = []
            for v in range(num_versions):
                np.random.seed(v)
                to_keep = np.random.binomial(1, probs).astype(bool) 
                train_zero_v = train_zero[to_keep]
                num_zero_sampled.append(len(train_zero_v))
                # update corrections dictionary to contain corrections *only for sampled zero data points*
                sample2correction = dict(zip(train_zero_v, corrections[to_keep]))
                with open(os.path.join(save_dir, 'w%d_v%d_train_zero.pkl' % (w, v)), 'wb') as f:
                    pickle.dump((train_zero_v, sample2correction), f)
            
            print('Finished sampling! Added %d nonzero, %s zero train points and %d held-out points' % 
                  (len(train_nnz), num_zero_sampled, num_heldout))
        print('Finished week [time = %.3fs]\n' % (time.time() - ts))

        
def sample_from_nnz_with_replacement(dset, save_dir, num_versions):
    """
    Sample from nonzero data points with replacement.
    """
    all_train_nnz = []
    for w in dset.week_indices:  
        with open(os.path.join(save_dir, 'w%d_train_nnz.pkl' % w), 'rb') as f:
            train_nnz = pickle.load(f)
        all_train_nnz.append(train_nnz)
    all_train_nnz = np.concatenate(all_train_nnz)
    print('Found %d train nnz points' % len(all_train_nnz))
    
    for v in range(num_versions):
        np.random.seed(v)
        sample = np.random.choice(all_train_nnz, size=len(all_train_nnz), replace=True)
        w_vec, _, _ = dset.index_to_wcp(sample)
        lengths = []
        for w in dset.week_indices:
            week_sample = sample[w_vec == w]
            lengths.append(len(week_sample))
            with open(os.path.join(save_dir, 'w%d_v%d_train_nnz.pkl' % (w, v)), 'wb') as f:
                pickle.dump(week_sample, f)
        print('Finished v%d: lengths =' % v, lengths)
        
    
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
    if weeks is None:  # use all available weeks in directory
        weeks = sorted(set([int(fn[1]) for fn in os.listdir(directory) if fn.startswith('w')]))
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
     

def get_county_pair_counts_per_tier_pair_type(dset, directory, top_n=5):
    """
    Check representation across county pairs in nonzero train data per tier pair type.
    """
    with open(os.path.join(directory, 'kwargs.pkl'), 'rb') as f:
        dset_kwargs, sampling_kwargs = pickle.load(f)
    control_tier, treatment_tier = sampling_kwargs['control_tier'], sampling_kwargs['treatment_tier']
    all_nnz = []
    for week in np.arange(dset.num_weeks()):
        with open(os.path.join(directory, 'w%d_train_nnz.pkl' % week), 'rb') as f:
            all_nnz.append(pickle.load(f))
    all_nnz = np.concatenate(all_nnz)
    print('Total num nonzero:', len(all_nnz))
    
    w_vec, c_vec, p_vec = dset.index_to_wcp(all_nnz)
    cbg_counties = dset.cbg_county_indices[c_vec]
    cbg_tiers = dset.county_tiers[w_vec, cbg_counties]
    poi_counties = dset.poi_county_indices[p_vec]
    poi_tiers = dset.county_tiers[w_vec, poi_counties]
    for cbg_tier in [control_tier, treatment_tier]:
        for poi_tier in [control_tier, treatment_tier]:
            in_tier_pair = (cbg_tiers == cbg_tier) & (poi_tiers == poi_tier)
            # each time this directed county pair appears for this tier pair
            county_pairs = zip(cbg_counties[in_tier_pair], poi_counties[in_tier_pair])
            most_common = Counter(county_pairs).most_common()
            print('CBG tier: %d. POI tier: %d. Total nonzero data points: %d. Num unique county pairs: %d.' % (
                cbg_tier, poi_tier, np.sum(in_tier_pair), len(most_common)))
            counts = np.array([t[1] for t in most_common])
            for p, ct in most_common[:top_n]:
                print(dset.indices['counties'][p[0]], dset.indices['counties'][p[1]],
                      ct, 100. * ct / np.sum(in_tier_pair))
            print()

            
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
                              num_versions=args.num_versions, visit_type=args.visit_type)

def main_outer(args):
    """
    Outer call that kicks off inner nohup command.
    """
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
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--mode', choices=['inner', 'outer'], default='outer')
    parser.add_argument('--bandwidth', type=float, default=5)
    parser.add_argument('--cbg_treatment', type=int, choices=[0, 1])
    parser.add_argument('--poi_treatment', type=int, choices=[0, 1])
    parser.add_argument('--rate', type=float, default=0.02)
    parser.add_argument('--distance_based', type=int, choices=[0, 1], default=1)
    parser.add_argument('--num_versions', type=int, default=1)
    parser.add_argument('--visit_type', type=str, choices=['within', 'cross', 'all', 'adj'], default='all')
    args = parser.parse_args()
    save_dir = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'CA', args.dir)
    
    if args.mode == 'outer':
        main_outer(args)
    else:
        main_inner(args)