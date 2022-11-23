import argparse
import copy
from dateutil import tz
import datetime
import math
import numpy as np
import nvgpu
from omegaconf import OmegaConf
import os
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import time
import warnings

import covid_constants_and_util as cu
import helper_methods_for_aggregate_data_analysis as helper
from dataset import *
from sampling import *

    
class PoissonRegModel(t.nn.Module):
    """
    Poisson regression model.
    """
    def __init__(self, features_dict, control_tier, treatment_tier, zero_inflated=True, 
                 use_poi_cat_groups=True, separate_same_and_cross_county=False):
        """
        Args:
            features_dict (dict): records structure of data provided in batch, from CBGPOIDataset.FEATURE_DICT
            control_tier (int): tier that we are using as control
            treatment_tier (int): tier that we are using as treatment
            zero_inflated (bool): whether to use zero inflation in model
            use_poi_cat_groups (bool): whether to learn heterogeneous treatment effects per POI group
            separate_same_and_cross_county (bool): whether to learn separate model parameters for within-county and 
                cross-county visits; not relevant when input data has already been filtered to be one visit type
        """
        assert (treatment_tier - control_tier) == 1  # must be consecutive
        super().__init__()
        self.features_dict = features_dict  # from CBGPOIDataset object
        self.control_tier = control_tier
        self.treatment_tier = treatment_tier
        self.zero_inflated = zero_inflated
        self.use_poi_cat_groups = use_poi_cat_groups
        self.separate_same_and_cross_county = separate_same_and_cross_county        
        
        if self.zero_inflated:
            # start with ones for scaling factors
            # scaling on distance in lambda, scaling on population, scaling on distance in pi (2 params)
            self.scaling_params = nn.Parameter(t.ones(size=(4,), requires_grad=True))
        else:  # no pi scaling param
            self.scaling_params = nn.Parameter(t.ones(size=(2,), requires_grad=True))
        num_cbg_feats = self.features_dict['cbg_static_attrs'][1] - features_dict['cbg_static_attrs'][0]
        self.cbg_weights = nn.Linear(num_cbg_feats, 1, bias=False)  # weights on static CBG attrs
        num_poi_feats = self.features_dict['poi_static_attrs'][1] - features_dict['poi_static_attrs'][0]
        self.poi_weights = nn.Linear(num_poi_feats, 1, bias=False)  # weights on static POI attrs
        t.nn.init.normal_(self.cbg_weights.weight)  # init these weights with N(0, 1)
        t.nn.init.normal_(self.poi_weights.weight)
                
        if self.use_poi_cat_groups:  # learn POI group-specific weights
            self.tier_pair_weights = nn.Parameter(t.randn(
                size=(self.features_dict['num_poi_groups'], 2, 2), requires_grad=True))
            self.tier_self_weights = nn.Parameter(t.randn(size=(self.features_dict['num_poi_groups'], 2,), requires_grad=True))
        else:
            self.tier_pair_weights = nn.Parameter(t.randn(size=(2, 2), requires_grad=True))
            self.tier_self_weights = nn.Parameter(t.randn(size=(2,), requires_grad=True))
            
        # large vs small county, pre vs post vaccine equity goal
        self.z_source_weights = nn.Parameter(t.randn(size=(2, 2), requires_grad=True))  # source (CBG's county) in external visit
        self.z_dest_weights = nn.Parameter(t.randn(size=(2, 2), requires_grad=True))  # destination (POI's county) in external visit
        self.z_self_weights = nn.Parameter(t.randn(size=(2, 2), requires_grad=True))  # internal visit
        
    def get_lambdas(self, batch):
        """
        Returns the lambda terms (ie, Poisson rate) per data point.
        """
        idx, y, cbg_attrs, poi_attrs, edge_attrs = batch  
        start_idx, end_idx = self.features_dict['cbg_static_attrs']
        cbg_static_attrs = cbg_attrs[:, start_idx:end_idx].float()
        cbg_biases = self.cbg_weights(cbg_static_attrs).reshape(-1)
        start_idx, end_idx = self.features_dict['poi_static_attrs']
        poi_static_attrs = poi_attrs[:, start_idx:end_idx].float()
        poi_biases = self.poi_weights(poi_static_attrs).reshape(-1)
        
        dists = edge_attrs[:, self.features_dict['cbg_poi_dist']]
        dist_scaling = -(nn.ReLU()(self.scaling_params[0]))  # util must be non-increasing in distance
        dist_terms = dist_scaling * t.log(dists)
        
        cbg_device_cts = cbg_attrs[:, self.features_dict['cbg_device_ct']]
        pop_scaling = nn.ReLU()(self.scaling_params[1])  # util must be non-decreasing in device count
        pop_terms = pop_scaling * t.log(cbg_device_cts)
        
        same_county = edge_attrs[:, self.features_dict['same_county']]
        cbg_tiers = cbg_attrs[:, self.features_dict['cbg_tier']]
        cbg_tiers = (cbg_tiers == self.treatment_tier).long()  # 1 if treatment, 0 if control
        poi_tiers = poi_attrs[:, self.features_dict['poi_tier']]
        poi_tiers = (poi_tiers == self.treatment_tier).long()
        if self.use_poi_cat_groups:
            poi_groups = poi_attrs[:, self.features_dict['poi_group']].long()
            tier_pair_terms = self.tier_pair_weights[poi_groups, cbg_tiers, poi_tiers]
            tier_self_terms = self.tier_self_weights[poi_groups, cbg_tiers]
        else:
            tier_pair_terms = self.tier_pair_weights[cbg_tiers, poi_tiers]
            tier_self_terms = self.tier_self_weights[cbg_tiers]
        if self.separate_same_and_cross_county:
            tier_terms = (same_county * tier_self_terms) + ((1 - same_county) * tier_pair_terms)
        else:
            tier_terms = tier_pair_terms  # all datapoints use the tier pair weights
        
        blueprint_stage = edge_attrs[:, self.features_dict['blueprint_stage']].long()  
        cbg_small_county = cbg_attrs[:, self.features_dict['cbg_small_county']].long()
        cbg_z = cbg_attrs[:, self.features_dict['cbg_assignment_var']]
        z_source_terms = self.z_source_weights[cbg_small_county, blueprint_stage] * cbg_z
        poi_small_county = poi_attrs[:, self.features_dict['poi_small_county']].long()
        poi_z = poi_attrs[:, self.features_dict['poi_assignment_var']]
        z_dest_terms = self.z_dest_weights[poi_small_county, blueprint_stage] * poi_z
        z_pair_terms = z_source_terms + z_dest_terms
        z_self_terms = self.z_self_weights[cbg_small_county, blueprint_stage] * cbg_z
        if self.separate_same_and_cross_county:
            z_terms = (same_county * z_self_terms) + ((1 - same_county) * z_pair_terms)
        else:
            z_terms = z_pair_terms

        utils = cbg_biases + poi_biases + dist_terms + pop_terms + tier_terms + z_terms
        lambdas = t.exp(utils)
        return lambdas
        
    def get_pis(self, dists):
        """
        Returns the pi terms (ie, mixing parameter for zero inflation) per data point.
        """
        dists_pow = dists ** self.scaling_params[3]
        pis = 1 / (1 + (self.scaling_params[2] * dists_pow))
        return pis
        
    def predict_and_compute_loss_on_data(self, batch, correction_terms=None, return_pred=False):
        """
        Predict visits and compute loss on real visits.
        """
        lambdas = self.get_lambdas(batch)
        y = batch[1]
        is_zero = y == 0
        poisson_nll = t.nn.PoissonNLLLoss(log_input=False, reduction='none')
        
        # neg_losses and pos_losses are vectors, represeting negative log likelihood per data point
        if self.zero_inflated:
            dists = batch[-1][:, self.features_dict['cbg_poi_dist']]
            pis = self.get_pis(dists)
            neg_losses = -t.log(1 - pis[is_zero] + (pis[is_zero] * t.exp(-lambdas[is_zero])))
            pos_losses = poisson_nll(lambdas[~is_zero], y[~is_zero])
            pos_losses = pos_losses - t.log(pis[~is_zero])
            pred = lambdas * pis
        else:
            neg_losses = poisson_nll(lambdas[is_zero], y[is_zero])
            pos_losses = poisson_nll(lambdas[~is_zero], y[~is_zero])
            pred = lambdas
        
        # sum over loss per data point, apply corrections if necessary
        if correction_terms is not None:
            if len(correction_terms) == 2:
                neg_term, pos_term = correction_terms
                neg_loss = t.sum(neg_losses) * neg_term
                pos_loss = t.sum(pos_losses) * pos_term
            else:
                assert len(correction_terms) == len(y), '%d correction terms, %d y' % (len(correction_terms), len(y))
                neg_loss = t.sum(neg_losses * correction_terms[is_zero])
                pos_loss = t.sum(pos_losses * correction_terms[~is_zero])
            loss = neg_loss + pos_loss
        else:  # no correction
            loss = t.sum(neg_losses) + t.sum(pos_losses)  
        
        if return_pred:
            return loss, pred
        else:
            return loss
    
    def get_lambdas_without_tier_or_z(self, batch):
        """
        Get lambda terms without using tier-related inputs. This is used in compute_county_county_weights.
        """
        idx, y, cbg_attrs, poi_attrs, edge_attrs = batch  
        start_idx, end_idx = self.features_dict['cbg_static_attrs']
        cbg_static_attrs = cbg_attrs[:, start_idx:end_idx].float()
        cbg_biases = self.cbg_weights(cbg_static_attrs).reshape(-1)
        start_idx, end_idx = self.features_dict['poi_static_attrs']
        poi_static_attrs = poi_attrs[:, start_idx:end_idx].float()
        poi_biases = self.poi_weights(poi_static_attrs).reshape(-1)
        
        dists = edge_attrs[:, self.features_dict['cbg_poi_dist']]
        dist_scaling = -(nn.ReLU()(self.scaling_params[0]))  # util must be non-increasing in distance
        dist_terms = dist_scaling * t.log(dists)
        
        cbg_device_cts = cbg_attrs[:, self.features_dict['cbg_device_ct']]
        pop_scaling = nn.ReLU()(self.scaling_params[1])  # util must be non-decreasing in device count
        pop_terms = pop_scaling * t.log(cbg_device_cts)
        
        utils = cbg_biases + poi_biases + dist_terms + pop_terms
        lambdas = t.exp(utils)
        return lambdas

    
#####################################################################
# Functions to interact with computer system and manage model 
# experiments, eg, parse configs, partition jobs
#####################################################################
def call_inner_jobs_from_outer(cfg, args):
    """
    Invokes inner call(s) and partitions job(s) to GPUs, when applicable.
    """
    assert cfg.num_trials >= 1
    for i in range(cfg.num_trials):
        # so that we can run multiple trials of the same experient in parallel
        if t.cuda.is_available():
            best_gpu, best_mem_avail = find_best_gpu()
            while best_gpu is None:
                print('No GPUs available right now...')
                time.sleep(10)
                best_gpu, best_mem_avail = find_best_gpu()
            print('Sending job to GPU %d with %d memory available' % (best_gpu, best_mem_avail))
        else:
            best_gpu = 0  # GPU argument is ignored if we are not on a GPU server
        out_str = cfg.experiment_name + (f'_{i}' if cfg.num_trials>1 else '') \
            + f'_{cfg.extension}'
        out_str = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'outputs', out_str)
        if args.weights is not None:
            cmd = f'nohup python -u poisson_reg_model.py --config {args.config} ' \
                f'--mode inner --gpu {best_gpu} --trial {i} --extension {cfg.extension} ' \
                f'--weights {args.weights} > {out_str}.out 2>&1 &'
        else:
            cmd = f'nohup python -u poisson_reg_model.py --config {args.config} ' \
                f'--mode inner --gpu {best_gpu} --trial {i} --extension {cfg.extension} ' \
                f'> {out_str}.out 2>&1 &'
        print('Command:', cmd)
        os.system(cmd)
        if i < (cfg.num_trials-1):  # if this is not the last trial
            time.sleep(5)

def run_experiment_across_negative_samples(cfg, args):
    """
    Constructs new configs per negative sample version and kicks off experiment for each one.
    """
    print('Running experiment across negative samples (use_sampled_nnz = %s)' % cfg.data.use_sampled_nnz)
    base_config_name = (args.config).split('.')[0]
    base_experiment_name = cfg.experiment_name
    assert (args.num_versions * cfg.train.num_workers) <= 100  # can't have too many workers if we are running in parallel
    for v in range(args.first_version, args.first_version+args.num_versions):
        # create config
        new_config_name = '%s_v%d.yml' % (base_config_name, v)
        cfg.experiment_name = '%s_v%d' % (base_experiment_name, v)
        cfg.data.neg_sample_version = v  
        OmegaConf.save(cfg, os.path.join(cu.PATH_TO_CBG_POI_DATA, 'configs', new_config_name))
        # note: this config will already have all arguments filled in, including extension
        
        # run experiment with config
        if t.cuda.is_available():
            best_gpu, best_mem_avail = find_best_gpu()
            while best_gpu is None:
                print('No GPUs available right now...')
                time.sleep(10)
                best_gpu, best_mem_avail = find_best_gpu()
            print('Sending job to GPU %d with %d memory available' % (best_gpu, best_mem_avail))
        else:
            best_gpu = 0  # GPU argument is ignored if we are not on a GPU server
        cmd = 'python poisson_reg_model.py --config %s' % new_config_name  # outer call
        print('Command:', cmd)
        os.system(cmd)
        time.sleep(5)
        
def find_best_gpu(min_mem=10000):
    """
    Finds the GPU with the most available memory, with at least min_mem available.
    If no GPU meets the minimum threshold, then None is returned.
    """
    gpu_info = nvgpu.gpu_info()
    best_option = None
    best_mem_avail = 0
    for info in gpu_info:
        mem_avail = info['mem_total'] - info['mem_used']
        if mem_avail >= min_mem and mem_avail > best_mem_avail:
            best_option = int(info['index'])
            best_mem_avail = mem_avail
    return best_option, best_mem_avail
    
def set_up_cfg(args):
    """
    Set up config object based on .yml file. Fill in missing fields with defaults.
    """
    cfg = OmegaConf.load(os.path.join(cu.PATH_TO_CBG_POI_DATA, 'configs', args.config))
    assert helper.check_cfg_contains_structure_of_reference(cfg, cu.cfg_field_reqs), 'Config file is missing fields. See required fields.'
    cfg.experiment_name = cfg.experiment_name + (f'_{args.trial}' if (args.trial is not None) and (cfg.num_trials > 1) else '')

    now = datetime.datetime.now(tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    cfg.extension = cfg.extension if 'extension' in cfg else (args.extension if args.extension is not None else timestamp)
    cfg = helper.fill_in_cfg_with_defaults(cfg, cu.cfg_field_reqs)
    return cfg


#####################################################################
# Functions to run model experiments
#####################################################################
def run_experiment(cfg, args):
    """
    Runs one model experiment. Calls several helper functions.
    """
    print('CONFIG')
    print(cfg)
    # 1. set up compute device
    if t.cuda.is_available():
        assert args.gpu is not None
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = t.device('cuda:0')
        print('Setting torch device to %s (gpu=%d)' % (device, args.gpu))
    else:
        device = t.device('cpu')
        print('Setting torch device to', device)
    
    # 2. load data
    dset_kwargs, sampling_kwargs, train_test_idx, correction_terms = load_train_test_indices_from_cfg(cfg)
    dset = CBGPOIDataset(**dset_kwargs)                    
    assert len(train_test_idx) <= dset.num_weeks()
    if len(train_test_idx) < dset.num_weeks():
        print('Warning: train_test_idx has fewer weeks (%d) than dataset (%d); experiment assumes that we start at week 0' % 
              (len(train_test_idx), dset.num_weeks()))
    if cfg.train.apply_corrections:
        if isinstance(correction_terms, dict):
            print('Received correction terms as dictionaries')
        else:
            assert len(correction_terms) == 2
            print('Received correction terms: neg correction = %.2f, pos correction = %.2f' % 
              (correction_terms[0], correction_terms[1]))
        dset.set_correction_terms(correction_terms)
    
    all_train_idx = np.concatenate([t[0] for t in train_test_idx])
    all_test_idx = np.concatenate([t[1] for t in train_test_idx])  # will be empty list if cfg.test.test_set = 'none'
    print('Total number of data points: train = %d, test (%s) = %d' % (len(all_train_idx), cfg.test.test_set, len(all_test_idx)))
    cfg.train.reg_lambda = float(cfg.train.reg_lambda / len(all_train_idx))  # do this so later we can just scale by length of batch
    
    # 3. construct model
    mdl = PoissonRegModel(dset.FEATURE_DICT, control_tier=sampling_kwargs['control_tier'],
                          treatment_tier=sampling_kwargs['treatment_tier'],
                          zero_inflated=cfg.model.zero_inflated,
                          use_poi_cat_groups=cfg.data.use_poi_cat_groups)
    print('Model parameters')
    for param in mdl.state_dict():
        print(param, mdl.state_dict()[param].size())
    mdl = mdl.to(device)
    opt = t.optim.Adam(mdl.parameters(), lr=cfg.train.lr) 
    scheduler = t.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    
    if args.weights is not None:
        print('Loading model and optimizer state dicts from', args.weights)
        full_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', args.weights)
        mdl.load_state_dict(t.load(full_fn, map_location=device))
        full_fn = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', '%s_opt' % args.weights)
        opt.load_state_dict(t.load(full_fn, map_location=device))
    
    # 4. fit model
    fit_start = time.time()
    results_per_epoch = {}
    for ep in range(cfg.train.epochs+1):
        if ep < cfg.train.epochs:
            print('===== EPOCH %d =====' % ep)
            eval_test = ((ep+1) % cfg.test.eval_freq) == 0  # don't evaluate test on every epoch
            if eval_test and cfg.test.test_set != 'none':
                results = run_epoch(dset, all_train_idx, mdl, opt, cfg, device, test_idx=all_test_idx, update_params=True)
            else:
                results = run_epoch(dset, all_train_idx, mdl, opt, cfg, device, test_idx=None, update_params=True)  
            save_model_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', '%s_%s_ep%d' % 
                                              (cfg.experiment_name, cfg.extension, ep))
            t.save(mdl.state_dict(), save_model_path)  # save weights after this epoch
        else:
            print('===== EXTRA EPOCH (not updating parameters) =====')
            if cfg.test.test_set != 'none':
                results = run_epoch(dset, all_train_idx, mdl, None, cfg, device, test_idx=all_test_idx, update_params=False)
            else:
                results = run_epoch(dset, all_train_idx, mdl, None, cfg, device, test_idx=None, update_params=False)
      
        results_per_epoch['epoch_%d' % ep] = results  # save results after each epoch
        save_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_results', f'{cfg.experiment_name}_{cfg.extension}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(results_per_epoch, f)
        scheduler.step()  # update learning rate
    
    dur = time.time() - fit_start
    print('Finished experiment! Total fitting time = %.2fs -> %.2fs per epoch' % (dur, dur / (cfg.train.epochs+1)))    
    
    
def run_epoch(dset, train_idx, mdl, opt, cfg, device, test_idx=None, update_params=True):  
    """
    Runs one epoch of training and, optionallly, one epoch of testing. Computes gradient and updates model 
    parameters if update_params is True.
    """
    ep_start = time.time()
    save_model_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', f'{cfg.experiment_name}_{cfg.extension}')
    t.save(mdl.state_dict(), save_model_path)  # save initial state    
    
    # go through train samples
    mdl.train()
    np.random.shuffle(train_idx) 
    train_loss, train_batch_loss = calculate_losses_over_data(dset, train_idx, mdl, cfg, device,
                                       apply_corrections=cfg.train.apply_corrections, opt=opt, 
                                       update_per_batch=update_params)
    # go through test samples
    if test_idx is not None:
        mdl.eval()
        with t.no_grad():
            if len(test_idx) > cfg.test.sample_size: 
                sample_test_idx = np.random.choice(test_idx, cfg.test.sample_size, replace=False)
                test_loss, _, test_batch_loss = calculate_losses_over_data(dset, sample_test_idx, mdl, cfg, device, 
                                                          apply_corrections=False, update_per_batch=False)
            else:
                test_loss, _, test_batch_loss = calculate_losses_over_data(dset, test_idx, mdl, cfg, device, 
                                                          apply_corrections=False, update_per_batch=False)
        results = {'train_loss':float(train_loss), 'train_loss_per_batch': train_batch_loss, 
                   'test_loss':float(test_loss), 'test_loss_per_batch': test_batch_loss}
    else:
        results = {'train_loss':float(train_loss), 'train_loss_per_batch': train_batch_loss, 
                   'test_loss':np.nan, 'test_loss_per_batch': []}
    print('Total train loss = %.3f, test loss = %.3f [time=%.3f]' % 
          (results['train_loss'], results['test_loss'], time.time() - ep_start))
    return results
    
    
def calculate_losses_over_data(dset, idx, mdl, cfg, device, apply_corrections=False,
                               opt=None, update_per_batch=True, verbosity=10):
    """
    Iterate through batches and calculate model's loss on each batch. Returns total loss (as a PyTorch loss)
    over all batches.
    """    
    if update_per_batch:
        batch_size = cfg.train.batch_size
        assert opt is not None
    else:
        batch_size = max(cfg.train.batch_size, 100000)  # can use larger batch size when we aren't updating per batch
    
    # prepare batches
    sampler = []  # fetch datapoints in batches; this is faster than getting datapoints individually then collating
    num_batches = math.ceil(len(idx) / batch_size)
    for i in range(num_batches):
        start_batch = i * batch_size
        end_batch = min(start_batch + batch_size, len(idx))
        sampler.append(idx[start_batch:end_batch])
    dl = DataLoader(dset, shuffle=False, sampler=sampler, collate_fn=collate_batch,
                    num_workers=cfg.train.num_workers, pin_memory=True)        
    
    total_loss = 0    
    avg_loss_per_batch = []
    for b, batch in enumerate(dl, start=0):
        batch_start = time.time()
        batch = [d.to(device) for d in batch]
        if apply_corrections:
            if dl.dataset.has_individual_corrections:  # we have correction terms per data point
                batch_idx = sampler[b]
                # only sampled zero data points are in correction terms; nnz data points are always included so correction is 1
                correction_terms = t.tensor([dl.dataset.correction_terms[i] if i in dl.dataset.correction_terms else 1 for i in batch_idx]).to(device)  
            else:
                correction_terms = dl.dataset.correction_terms  # 2-tuple of neg_term, pos_term
        else:
            correction_terms = None
        loss = mdl.predict_and_compute_loss_on_data(batch, correction_terms=correction_terms, return_pred=False)
        total_loss += loss
        avg_loss_per_batch.append(float(loss) / len(batch[1]))
        
        if update_per_batch:
            update_params_and_save_state_dicts(mdl, opt, loss, cfg, device)
        if verbosity > 0 and ((b+1) % verbosity) == 0:  # verbosity is frequency of printing
            print('Batch %d: loss = %.3f [time=%.3f]' % 
                  (b, float(loss), time.time()-batch_start))
    return total_loss, avg_loss_per_batch


def update_params_and_save_state_dicts(mdl, opt, loss, cfg, device, print_delta=False):
    """
    Backpropagate on loss, update model parameters, and save states for model and optimizer.
    """
    opt.zero_grad()
    loss.backward()
    opt.step()
    save_model_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', f'{cfg.experiment_name}_{cfg.extension}')
    save_opt_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'model_weights', f'{cfg.experiment_name}_{cfg.extension}_opt')
    # save new model weights, check how much weights changed
    if print_delta:
        saved_weights = t.load(save_model_path, map_location=device)
        new_weights = mdl.state_dict()
        delta = 0
        for k in saved_weights.keys():
            delta += t.norm(saved_weights[k] - new_weights[k])
        print('Change in model parameters = %.6f' % delta)
    t.save(mdl.state_dict(), save_model_path) 
    t.save(opt.state_dict(), save_opt_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to experiment config file.')
    parser.add_argument('--mode', choices=['inner', 'outer', 'across_negative_samples'], default='outer')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--trial', type=int, default=None)
    parser.add_argument('--extension', type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--num_versions', type=int, default=1)
    parser.add_argument('--first_version', type=int, default=0)
    args = parser.parse_args()
    cfg = set_up_cfg(args)
    
    if args.mode == 'outer':
        # save the filled-in config and args so we can recreate experiment if necessary
        save_path = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'outputs', '%s_%s_cfg_args.pkl' % (cfg.experiment_name, cfg.extension))
        with open(save_path, 'wb') as f:  
            pickle.dump((cfg, args), f)
        call_inner_jobs_from_outer(cfg, args)
    elif args.mode == 'across_negative_samples':
        run_experiment_across_negative_samples(cfg, args)
    else:
        run_experiment(cfg, args)
            
