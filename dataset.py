import os
from haversine import haversine_vector, Unit
import datetime
import time
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import pickle
import itertools
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix

import covid_constants_and_util as cu
import helper_methods_for_aggregate_data_analysis as helper


class CBGPOIDataset(Dataset):
    def __init__(self, data=None, directory=None, start_date=None, end_date=None, path_to_county_data=None, 
                 cbg_count_min=cu.CBG_NONZERO_MIN_COUNT, poi_count_min=cu.POI_NONZERO_MIN_COUNT, 
                 acs_coverage_min=0.9, use_key_demographics=True, poi_cat_count_min=20, 
                 scale_attributes=True, correction_terms=None, load_distances_dynamically=True):
        """
        Args:
            data (tuple): used when we want to provide toy data to test, must be the 7 necessary data elements
            directory (string): path to directory with data (see load_data)
            start_date (string): start date of data to include
            end_date (string): end date of data to include
            cbg_count_min (int): min number of nonzero visits the CBG must have to be included
            poi_count_min (int): min number of nonzero visits the POI must have to be included
            acs_coverage_min (proportion): min proportion of CBGs that this ACS field must be non-nan for 
                                           to be included in CBG attributes
            use_key_demographics (bool): whether to only use a set of pre-selected key demographics (eg, age,
                                         race), instead of all ACS fields
            poi_cat_count_min (int): min number of kept POIs that the POI sub/topcategory must have to be included 
                                    in POI attributes
            load_dynamic (bool): whether to load dynamic attributes for CBGs and POIs
            scale_attributes (bool): whether to apply min-max pooling to non-one-hot features (which are 
                                     tier and POI category)
            correction_terms (dict/tuple): correction terms for data points, see set_correction_terms()
                                           for details; defaults to (1,1) when correction_terms is None
        """
        self.use_key_demographics = use_key_demographics
        self.scale_attributes = scale_attributes
        self.load_distances_dynamically = load_distances_dynamically
        self.has_county_data = False
        
        if correction_terms is None:
            self.set_correction_terms((1, 1))
        else:
            self.set_correction_terms(correction_terms)
        if data is None:
            assert all([directory is not None, start_date is not None, end_date is not None])
            self._init_cbg_poi_data(directory, start_date, end_date, cbg_count_min, poi_count_min, acs_coverage_min)
        else:
            self.indices, self.visits, self.distances, self.cbg_attrs, self.poi_attrs, self.cbg_device_counts = data
            self.week_indices = np.arange(self.num_weeks())
            self.cbg_indices = np.arange(self.num_cbgs())
            self.poi_indices = np.arange(self.num_pois())
        
        self.county2idx = dict(zip(self.indices['counties'], range(len(self.indices['counties']))))
        self.cbg_county_indices = np.array([self.county2idx[helper.extract_county_code_fr_fips(cbg)] for cbg in self.indices['cbgs']])  # county idx corresponding to each CBG
        self.poi_county_indices = np.array([self.county2idx[helper.extract_county_code_fr_fips(poi_cbg)] for poi_cbg in self.indices['poi_cbgs']])  # county idx corresponding to each POI
        
        if path_to_county_data is not None:
            self._init_county_data(path_to_county_data)
        self.run_data_checks()
        self.check_values_are_valid()
                                                
        # Map POI subcategories to indices, will use later for one-hot encoding
        self.subcat2idx = {}
        self.num_subcat_classes = 1  # we will at least have Other
        self.poi_subcat_indices = np.zeros(self.num_pois(), dtype=int)  # the subcat index for each POI
        for subcat, poi_indices in self.poi_attrs.groupby('sub_category').indices.items():
            if (len(poi_indices) < poi_cat_count_min) or (subcat == ''):
                subcat_idx = 0
            else:
                subcat_idx = self.num_subcat_classes
                self.num_subcat_classes += 1
            self.subcat2idx[subcat] = subcat_idx
            self.poi_subcat_indices[poi_indices] = subcat_idx
        
        # Map common POI topcategories to indices, will use for heterogeneous treatment effects 
        self.subcat2group = {}
        self.poi_group_labels = ['Other']
        self.poi_group_indices = np.zeros(self.num_pois(), dtype=int)  # the group index for each POI
        for topcat, poi_indices in self.poi_attrs.groupby('top_category').indices.items():
            if (len(poi_indices) < 1000) or (topcat in {'', 'Restaurants and Other Eating Places'}):
                group_idx = 0
            else:
                group_idx = len(self.poi_group_labels)
                self.poi_group_labels.append(topcat)
            subcats = self.poi_attrs.iloc[poi_indices].sub_category.unique()
            for s in subcats:
                self.subcat2group[s] = group_idx
            self.poi_group_indices[poi_indices] = group_idx
            
        # Allow these subcategories to be their own group
        for subcat in ['Full-Service Restaurants', 'Snack and Nonalcoholic Beverage Bars', 
                       'Limited-Service Restaurants', 'Fitness and Recreational Sports Centers']:
            group_idx = self.num_poi_groups()
            self.poi_group_labels.append(subcat)
            self.subcat2group[subcat] = group_idx
            poi_indices = self.poi_indices[self.poi_attrs.sub_category == subcat]
            assert len(poi_indices) >= 1000
            self.poi_group_indices[poi_indices] = group_idx
        print('Modeling %d subcategories and %d POI groups' % (self.num_subcat_classes, self.num_poi_groups()))
        
        # global variables that describe structure of data in batches/single sample
        self.BATCH_KEYS = ['indices', 'num_visits', 'cbg_attrs', 'poi_attrs', 'edge_attrs']
        cbg_d = self.cbg_attrs.values.shape[1]
        poi_d = self.num_subcat_classes + self.poi_attrs.values.shape[1] - 2  # first 2 cols of poi_attrs are sub/top category
        self.FEATURE_DICT = {
            'num_weeks':self.num_weeks(),
            'num_cbgs':self.num_cbgs(),
            'num_pois':self.num_pois(),
            'num_counties':self.num_counties(),
            'num_poi_groups':self.num_poi_groups(),
            
            'cbg_num_attrs':cbg_d+4,
            'cbg_static_attrs':(0, cbg_d),
            'cbg_device_ct':cbg_d,
            'cbg_tier':cbg_d+1,
            'cbg_assignment_var':cbg_d+2,
            'cbg_small_county':cbg_d+3,
            
            'poi_num_attrs':poi_d+4,
            'poi_static_attrs':(0, poi_d),
            'poi_group':poi_d,
            'poi_tier':poi_d+1,
            'poi_assignment_var':poi_d+2,
            'poi_small_county':poi_d+3,
            
            'edge_num_attrs':3,
            'cbg_poi_dist':0,
            'same_county':1,
            'blueprint_stage':2
        }
                
    def _init_cbg_poi_data(self, directory, start_date, end_date, cbg_count_min, poi_count_min, acs_coverage_min):
        """
        Load CBG and POI data. Filter CBGs and POIs based on different checks. 
        Run this if processed data is not provided.
        """
        data = load_cbg_poi_data(directory, use_key_demographics=self.use_key_demographics,
                                 load_distances=not(self.load_distances_dynamically))
        self.indices, self.visits, self.distances, self.cbg_attrs, self.poi_attrs, self.cbg_device_counts = data
        print('Loaded data from %s' % directory)
        if self.load_distances_dynamically:  # load lat/lons so we can generate distances on the spot
            cbg_df = pd.read_csv(os.path.join(cu.PATH_TO_CBG_POI_DATA, 'cbg_centroid_lat_lon.csv')).set_index('census_block_group')
            self.cbg_lat_lon = cbg_df.loc[self.indices['cbgs']]            
            poi_df = pd.read_csv(os.path.join(directory, 'poi_attrs.csv')).set_index('safegraph_place_id')
            poi_df = poi_df[['latitude', 'longitude']]
            self.poi_lat_lon = poi_df.loc[self.indices['pois']]     
        self.run_data_checks()
        self.cbg_indices = np.arange(self.num_cbgs())
        self.poi_indices = np.arange(self.num_pois())
        
        # filter based on desired weeks
        weeks_to_keep = [(start_date <= w <= end_date) for w in self.indices['weeks']]
        self.indices['weeks'] = [w for i, w in enumerate(self.indices['weeks']) if weeks_to_keep[i]]
        print('Keeping data for %d weeks, from %s to %s' % (len(self.indices['weeks']), self.indices['weeks'][0], self.indices['weeks'][-1]))
        self.visits = [m for i, m in enumerate(self.visits) if weeks_to_keep[i]]
        self.cbg_device_counts = self.cbg_device_counts[self.indices['weeks']]
        self.week_indices = np.arange(self.num_weeks())
        
        # filter out CBGs and POIs without enough nonzero visits (from the desired time period)
        _, c_idx, p_idx = self.get_nonzero_indices(as_wcp=True)
        cbg_counter = Counter(c_idx)
        cbg_counts = np.array([cbg_counter[c] for c in self.cbg_indices])
        cbgs_to_keep = cbg_counts >= cbg_count_min
        print('Keeping %d/%d CBGs with at least %d nonzero visits' % 
              (np.sum(cbgs_to_keep), self.num_cbgs(), cbg_count_min))
        poi_counter = Counter(p_idx)
        poi_counts = np.array([poi_counter[p] for p in self.poi_indices])
        pois_to_keep = poi_counts >= poi_count_min
        print('Keeping %d/%d POIs with at least %d nonzero visits' % 
              (np.sum(pois_to_keep), self.num_pois(), poi_count_min))
            
        # drop CBGs and POIs that we aren't keeping
        self.indices['cbgs'] = self.indices['cbgs'][cbgs_to_keep]
        self.indices['pois'] = self.indices['pois'][pois_to_keep]
        self.indices['poi_cbgs'] = self.indices['poi_cbgs'][pois_to_keep]
        self.visits = [m[cbgs_to_keep, :][:, pois_to_keep] for m in self.visits]
        self.cbg_attrs = self.cbg_attrs.loc[self.indices['cbgs']]
        self.poi_attrs = self.poi_attrs.loc[self.indices['pois']]
        self.cbg_device_counts = self.cbg_device_counts.loc[self.indices['cbgs']]
        self.cbg_indices = np.arange(self.num_cbgs())
        self.poi_indices = np.arange(self.num_pois())
        if self.load_distances_dynamically:
            self.cbg_lat_lon = cbg_df.loc[self.indices['cbgs']]
            self.poi_lat_lon = poi_df.loc[self.indices['pois']]
        else:
            self.distances = self.distances[cbgs_to_keep, :][:, pois_to_keep]
                
        # Deal with NaNs in CBG attributes
        # 1. Only keep ACS fields with enough coverage (see acs_coverage_min)
        cbg_attrs_nan = self.cbg_attrs.isna().values
        acs_field_coverage = np.sum(~cbg_attrs_nan, axis=0)
        fields_to_keep = acs_field_coverage >= (self.num_cbgs() * acs_coverage_min)
        # only want estimate fields, not margin of error
        kept_fields = [c for c, k in zip(self.cbg_attrs.columns, fields_to_keep) if k and 'e' in c]
        print('Keeping %d/%d ACS fields that are estimate, not margin of error, and cover at least %d%% of CBGs.' % 
              (len(kept_fields), len(self.cbg_attrs.columns), acs_coverage_min * 100))
        self.cbg_attrs = self.cbg_attrs[kept_fields]
        cbg_attrs_nan = cbg_attrs_nan[:, fields_to_keep]
        
        # 2. Fill remaining NaNs in ACS data with medians
        print('Filling remaining NaN entries (%.3f%% of all entries) with medians.' % 
              (100 * np.sum(cbg_attrs_nan) / (self.cbg_attrs.shape[0] * self.cbg_attrs.shape[1])))
        medians = np.nanmedian(self.cbg_attrs.values, axis=0)  # get median per field, ignoring NaNs
        median_dict = {c:m for c, m in zip(self.cbg_attrs.columns, medians)}
        self.cbg_attrs = self.cbg_attrs.fillna(value=median_dict)
        
        # Deal with NaNs in POI attributes
        # sub_category
        poi_sub_cats = self.poi_attrs.sub_category.isna().values
        print('%d POIs (%.2f%%) are missing sub_category' % 
              (np.sum(poi_sub_cats), 100 * np.sum(poi_sub_cats) / self.num_pois()))
        self.poi_attrs['sub_category'] = self.poi_attrs.sub_category.fillna('')
        # top_category
        poi_top_cats = self.poi_attrs.top_category.isna().values
        print('%d POIs (%.2f%%) are missing top_category' % 
              (np.sum(poi_top_cats), 100 * np.sum(poi_top_cats) / self.num_pois()))
        self.poi_attrs['top_category'] = self.poi_attrs.top_category.fillna('')     
        
        # Deal with zeros in CBG device counts
        assert (~self.cbg_device_counts.isna()).all(axis=None)
        invalid = self.cbg_device_counts <= 0
        print('Found %d values <= 0 in CBG device counts -> clipping to 2' % invalid.sum().sum())
        self.cbg_device_counts = self.cbg_device_counts.clip(lower=2, upper=None, axis=None)
        
        # Apply min-max scaling to attributes, so that losses aren't extremely big at first
        if self.scale_attributes:
            print('Applying min-max scaling to CBG and POI attributes (besides POI categories).')
            self.cbg_attrs.iloc[:, :] = MinMaxScaler().fit_transform(self.cbg_attrs.values)
            assert self.poi_attrs.columns[0] == 'sub_category'
            assert self.poi_attrs.columns[1] == 'top_category'
            self.poi_attrs.iloc[:, 2:] = MinMaxScaler().fit_transform(self.poi_attrs.iloc[:, 2:].values)
                

    def _init_county_data(self, path_to_data):
        """
        Load county-level data based on (filtered) CBG and POI data.
        """
        with open(path_to_data, 'rb') as f:
            fips, population, tier_dates, blueprint_stages, tiers, assignment_vars = pickle.load(f)
        fips2idx = {f:i for i,f in enumerate(fips)}
        counties_to_keep = [fips2idx[c] for c in self.indices['counties']] 
        self.county_populations = population[counties_to_keep]
        dates_to_keep = []
        for dt in tier_dates:
            weekday = dt.weekday()
            # map tier date (usually a Tuesday) to the prior Monday
            prev_monday = dt + datetime.timedelta(days=-weekday)  
            ds = datetime.datetime.strftime(prev_monday, '%Y-%m-%d')
            if ds in self.indices['weeks']:
                dates_to_keep.append(True)
            else:
                dates_to_keep.append(False)
        assert self.num_weeks() == np.sum(dates_to_keep)  # should find a tier date corresponding to each mobility date
        self.weekly_blueprint_stages = blueprint_stages[dates_to_keep]
        self.county_tier_dates = np.array(tier_dates)[dates_to_keep]
        self.county_tiers = tiers[dates_to_keep, :][:, counties_to_keep]
        self.county_assignment_vars = assignment_vars[dates_to_keep, :][:, counties_to_keep]
        
        self.county2cbgs = {c:[] for c in self.indices['counties']}  # county to CBG indices
        self.county2pois = {c:[] for c in self.indices['counties']}  # county to POI indices
        for i, cbg in enumerate(self.indices['cbgs']):
            county = helper.extract_county_code_fr_fips(cbg)
            self.county2cbgs[county].append(i)
        for j, poi_cbg in enumerate(self.indices['poi_cbgs']):
            county = helper.extract_county_code_fr_fips(poi_cbg)
            self.county2pois[county].append(j) 
        print('Loaded dynamic county data from', path_to_data)
        self.has_county_data = True
                   
            
    def __len__(self):
        """
        Returns the total number of data points.
        """
        return self.num_weeks() * self.num_cbgs() * self.num_pois()

    def __getitem__(self, idx):
        """
        Returns either a single data point or a batch of data points, depending on 
        the type of idx. 
        """
        if type(idx) in {list, np.ndarray}:
            return self.get_batch(np.array(idx))  # received list of indices
        return self.get_single_sample(idx)
    
    def get_single_sample(self, idx):
        """
        Returns a single data point.
        """
        assert idx < self.__len__() and idx >= 0
        batch = self.get_batch(np.array([idx]))
        sample = {k:b[0] for k, b in zip(self.BATCH_KEYS, batch)}
        return sample
        
    def get_batch(self, idxs):
        """
        Custom function to create a batch of data, based on indices in idxs. Returns the same
        thing as collate_fn would on a list of dataset[i], but it's much faster because we don't 
        need to individually get each datapoint.
        """
        assert all((idxs < self.__len__()) & (idxs >= 0))
        w_vec, c_vec, p_vec = self.index_to_wcp(idxs)
        wcp_indices = t.tensor([w_vec, c_vec, p_vec]).T  # indices
        num_visits = t.tensor([self.visits[w][c,p] for w, c, p in zip(w_vec, c_vec, p_vec)])
        
        cbg_attrs = np.zeros((len(idxs), self.FEATURE_DICT['cbg_num_attrs']))
        start_idx, end_idx = self.FEATURE_DICT['cbg_static_attrs']
        cbg_attrs[:, start_idx:end_idx] = self.cbg_attrs.values[c_vec, :]
        cbg_attrs[:, self.FEATURE_DICT['cbg_device_ct']] = self.cbg_device_counts.values[c_vec, w_vec]
        
        poi_attrs = np.zeros((len(idxs), self.FEATURE_DICT['poi_num_attrs']))
        subcats = self.poi_subcat_indices[p_vec]
        poi_attrs[np.arange(len(idxs)), subcats] = 1  # one-hot
        start_idx, end_idx = self.num_subcat_classes, self.FEATURE_DICT['poi_static_attrs'][1]
        poi_attrs[:, start_idx:end_idx] = self.poi_attrs.values[p_vec, 2:].astype(float)
        poi_attrs[:, self.FEATURE_DICT['poi_group']] = self.poi_group_indices[p_vec]
        
        cbg_counties = self.cbg_county_indices[c_vec]
        poi_counties = self.poi_county_indices[p_vec]
        if self.has_county_data:
            cbg_attrs[:, self.FEATURE_DICT['cbg_tier']] = self.county_tiers[w_vec, cbg_counties]
            cbg_attrs[:, self.FEATURE_DICT['cbg_assignment_var']] = self.county_assignment_vars[w_vec, cbg_counties]
            cbg_attrs[:, self.FEATURE_DICT['cbg_small_county']] = (self.county_populations[cbg_counties] < 106000).astype(int)
            poi_attrs[:, self.FEATURE_DICT['poi_tier']] = self.county_tiers[w_vec, poi_counties]
            poi_attrs[:, self.FEATURE_DICT['poi_assignment_var']] = self.county_assignment_vars[w_vec, poi_counties]
            poi_attrs[:, self.FEATURE_DICT['poi_small_county']] = (self.county_populations[poi_counties] < 106000).astype(int)
        
        edge_attrs = np.zeros((len(idxs), self.FEATURE_DICT['edge_num_attrs']))
        edge_attrs[:, self.FEATURE_DICT['cbg_poi_dist']] = self.get_cbg_poi_dists(c_vec, p_vec)
        edge_attrs[:, self.FEATURE_DICT['same_county']] = (cbg_counties == poi_counties).astype(int)
        edge_attrs[:, self.FEATURE_DICT['blueprint_stage']] = self.weekly_blueprint_stages[w_vec]
        
        return wcp_indices, num_visits, t.tensor(cbg_attrs), t.tensor(poi_attrs), t.tensor(edge_attrs)
    
    def get_batch_for_county_weights(self, idxs):
        """
        Only retrieve the inputs needed for compute_county_county_weights
        """
        assert all((idxs < self.__len__()) & (idxs >= 0))
        w_vec, c_vec, p_vec = self.index_to_wcp(idxs)
        wcp_indices = t.tensor([w_vec, c_vec, p_vec]).T  # indices
        num_visits = None
        
        cbg_attrs = np.zeros((len(idxs), self.FEATURE_DICT['cbg_num_attrs']))
        start_idx, end_idx = self.FEATURE_DICT['cbg_static_attrs']
        cbg_attrs[:, start_idx:end_idx] = self.cbg_attrs.values[c_vec, :]
        cbg_attrs[:, self.FEATURE_DICT['cbg_device_ct']] = self.cbg_device_counts.values[c_vec, w_vec]
        
        poi_attrs = np.zeros((len(idxs), self.FEATURE_DICT['poi_num_attrs']))
        subcats = [self.subcat2idx[cat] for cat in self.poi_attrs.sub_category.values[p_vec]]
        poi_attrs[np.arange(len(idxs)), subcats] = 1  # one-hot
        start_idx, end_idx = self.num_subcat_classes, self.FEATURE_DICT['poi_static_attrs'][1]
        poi_attrs[:, start_idx:end_idx] = self.poi_attrs.values[p_vec, 2:].astype(float)
        
        edge_attrs = np.zeros((len(idxs), self.FEATURE_DICT['edge_num_attrs']))
        edge_attrs[:, self.FEATURE_DICT['cbg_poi_dist']] = self.get_cbg_poi_dists(c_vec, p_vec)
        
        return wcp_indices, num_visits, t.tensor(cbg_attrs), t.tensor(poi_attrs), t.tensor(edge_attrs)
        
    
    def get_visits(self, idxs): 
        """
        Returns visits corresponding to the indices in idxs.
        """
        assert all((idxs < self.__len__()) & (idxs >= 0))
        w_vec, c_vec, p_vec = self.index_to_wcp(idxs)
        return np.array([self.visits[w][c,p] for w, c, p in zip(w_vec, c_vec, p_vec)])
        
    def num_weeks(self):
        """
        Returns number of weeks.
        """
        return len(self.indices['weeks'])
    
    def num_cbgs(self):
        """
        Returns number of CBGs.
        """
        return len(self.indices['cbgs'])
    
    def num_pois(self):
        """
        Returns number of POIs.
        """
        return len(self.indices['pois'])
    
    def num_counties(self):
        """
        Returns number of counties.
        """
        return len(self.indices['counties'])
    
    def num_poi_groups(self):
        """
        Returns number of POI groups.
        """
        return len(self.poi_group_labels)
    
    def set_correction_terms(self, correction_terms):
        """
        Sets correction terms, which will be used when computing model losses. Correction terms
        are either a dict mapping individual data index to correction or a 2-tuple of
        (negative_correction_term, positive_correction_term), where the former applies to all
        data points with 0 visits and the latter applies to all data points with > 0 visits.
        """
        self.correction_terms = correction_terms
        if isinstance(correction_terms, dict):
            corrections = np.array(list(self.correction_terms.values()))
            assert all(corrections >= 1)
            self.has_individual_corrections = True
        else:
            assert len(correction_terms) == 2
            assert correction_terms[0] >= 1
            assert correction_terms[1] >= 1
            if correction_terms[0] < correction_terms[1]:
                print('WARNING: negative correction term (%s) is smaller than positive correction term (%s); did you mean to flip the order?' % (correction_terms[0], correction_terms[1]))
            self.has_individual_corrections = False
    
    def run_data_checks(self):
        """
        Check that data matches requirements.
        """
        assert len(self.indices['poi_cbgs']) == self.num_pois()
        assert len(self.visits) == self.num_weeks()
        assert all([m.shape == (self.num_cbgs(), self.num_pois()) for m in self.visits])
        assert all(self.indices['cbgs'] == self.cbg_attrs.index.values)
        assert all(self.indices['pois'] == self.poi_attrs.index.values)
        assert self.poi_attrs.columns[0] == 'sub_category'
        assert self.poi_attrs.columns[1] == 'top_category'
        assert all(self.indices['cbgs'] == self.cbg_device_counts.index.values)
        assert all(self.indices['weeks'] == self.cbg_device_counts.columns.values)
        if self.load_distances_dynamically:
            assert all(self.indices['cbgs'] == self.cbg_lat_lon.index.values)
            assert self.cbg_lat_lon.columns[0] == 'latitude'
            assert self.cbg_lat_lon.columns[1] == 'longitude'
            assert all(self.indices['pois'] == self.poi_lat_lon.index.values)
            assert self.poi_lat_lon.columns[0] == 'latitude'
            assert self.poi_lat_lon.columns[1] == 'longitude'
        else:
            assert self.distances.shape == (self.num_cbgs(), self.num_pois())
        
        if self.has_county_data:
            assert self.county_populations.shape == (self.num_counties(),)
            assert self.county_tier_dates.shape == (self.num_weeks(),)
            assert self.weekly_blueprint_stages.shape == (self.num_weeks(),)
            assert self.county_tiers.shape == (self.num_weeks(), self.num_counties())
            assert self.county_assignment_vars.shape == (self.num_weeks(), self.num_counties())
        
    def check_values_are_valid(self):
        """
        Check for nans or invalid values any of the data.
        """
        assert (~self.cbg_attrs.isna()).all(axis=None)
        assert (~self.poi_attrs.isna()).all(axis=None)
        assert (~self.cbg_device_counts.isna()).all(axis=None)
        assert (self.cbg_device_counts > 0).all(axis=None)  # must all be positive
        if self.load_distances_dynamically:
            assert (~self.cbg_lat_lon.isna()).all(axis=None)
            assert (~self.poi_lat_lon.isna()).all(axis=None)
        else:
            assert np.sum(np.isnan(self.distances)) == 0
            assert np.all(self.distances > 0)  # must all be positive
        if self.has_county_data:
            assert np.sum(np.isnan(self.county_populations)) == 0
            assert np.all(self.county_populations > 0)
            assert np.sum(np.isnan(self.weekly_blueprint_stages)) == 0
            assert np.sum(np.isnan(self.county_tiers)) == 0
            assert np.sum(np.isnan(self.county_assignment_vars)) == 0
        assert self.__len__() < np.iinfo(np.int64).max  # make sure we won't have overflow problems
        
    def index_to_wcp(self, idx):
        """
        Translates from single index to week, CBG, and POI index.
        """
        assert all((idx < self.__len__()) & (idx >= 0))
        idx = idx.astype(np.int64)  # will also return np.int64
        w = idx // (self.num_cbgs() * self.num_pois())
        c = (idx % (self.num_cbgs() * self.num_pois())) // self.num_pois()
        p = idx % self.num_pois()
        return w, c, p
    
    def wcp_to_index(self, w, c, p):
        """
        Translates from week, CBG, and POI index to single index.
        """
        assert all((w < self.num_weeks()) & (w >= 0))
        assert all((c < self.num_cbgs()) & (c >= 0))
        assert all((p < self.num_pois()) & (p >= 0))
        idx = w.astype(np.int64) * (self.num_cbgs() * self.num_pois())
        idx += c.astype(np.int64) * self.num_pois()
        idx += p.astype(np.int64)
        return idx  # will also return np.int64
    
    def get_nonzero_indices(self, as_wcp=True):
        """
        Returns the indices corresponding to the data points with nonzero visits.
        """
        all_w = []
        all_c = []
        all_p = []
        for w in range(self.num_weeks()):
            w_vec, c_vec, p_vec = self.get_nonzero_indices_for_week(w)
            all_w.append(w_vec)
            all_c.append(c_vec)
            all_p.append(p_vec)
        w = np.concatenate(all_w)
        c = np.concatenate(all_c)
        p = np.concatenate(all_p)
        if as_wcp:
            return w, c, p
        return self.wcp_to_index(w, c, p)   
    
    def get_nonzero_indices_for_week(self, week, as_wcp=True):
        """
        Returns the indices corresponding to the data points with nonzero visits for this week.
        """
        assert week < self.num_weeks() and week >= 0
        c, p = self.visits[week].nonzero()
        c, p = c.astype(np.int64), p.astype(np.int64)  # nonzero() returns np.int32
        w = np.ones(len(c), dtype=np.int64) * week
        if as_wcp:
            return w, c, p
        return self.wcp_to_index(w, c, p)
    
    def get_cbg_poi_dists(self, c_vec, p_vec):
        """
        Returns distances between CBGs and POIs.
        """
        assert all((c_vec < self.num_cbgs()) & (c_vec >= 0))
        assert all((p_vec < self.num_pois()) & (p_vec >= 0))
        if self.load_distances_dynamically:
            cbg_mat = self.cbg_lat_lon.values[c_vec]
            poi_mat = self.poi_lat_lon.values[p_vec]
            return haversine_vector(cbg_mat, poi_mat, Unit.KILOMETERS)
        return self.distances[c_vec, p_vec]
    
    def get_normalized_adj_mat_for_week(self, week, train_idx=None, predict_binary=True):
        """
        Get normalized adjacency matrix for week, where A_norm[i,j] = A[i,j] / sqrt(deg_i)sqrt(deg_j).
        If predict_binary, then A[i,j] and degrees are unweighted (i.e., A[i, j] is 0 or 1).
        This is used in LightGCN model.
        """
        m = self.visits[week]
        if predict_binary:  # keep only 1's instead of number of visits
            m = csr_matrix((m > 0).astype(int))  
        if train_idx is not None:  # only keep visits in train
            _, c, p = self.index_to_wcp(train_idx)  
            mask = csr_matrix((np.ones(len(c), dtype=int), (c, p)), shape=m.shape)
            m = m.multiply(mask)  # elementwise multiplication

        c, p = m.nonzero()
        cbg_degrees = m @ np.ones(self.num_pois())
        poi_degrees = m.transpose() @ np.ones(self.num_cbgs())
        norm1 = np.sqrt(cbg_degrees[c])
        norm2 = np.sqrt(poi_degrees[p])
        visits = np.asarray(m[c, p]).reshape(-1)  # nonzero entries of m
        normed_visits = t.tensor(visits / (norm1 * norm2))

        row_idx = np.concatenate([c, p + self.num_cbgs()])
        col_idx = np.concatenate([p + self.num_cbgs(), c])
        indices = t.tensor([row_idx, col_idx])
        data = t.cat([normed_visits, normed_visits])  # adj mat is symmetric, so repeat the normed visits
        s = self.num_cbgs() + self.num_pois()
        A_norm = t.sparse_coo_tensor(indices, data, size=[s,s], dtype=t.float32)
        return A_norm
        

def collate_individual_datapoints(data_list):
    """
    Combines a list of dicts indexed from a CBGPOIDataset object.
    data_list (list): List of dicts containing the data (see CBGPOIDataset.__getitem__)
    """
    indices = t.stack([data['indices'] for data in data_list], dim=0)
    num_visits = t.stack([data['num_visits'] for data in data_list], dim=0)
    cbg_attrs = t.stack([data['cbg_attrs'] for data in data_list], dim=0)
    poi_attrs = t.stack([data['poi_attrs'] for data in data_list], dim=0)
    edge_attrs = t.stack([data['edge_attrs'] for data in data_list], dim=0)
    return (
        indices,
        num_visits,
        cbg_attrs,
        poi_attrs,
        edge_attrs
    )

def collate_batch(batch):
    """
    batch is in the form of [batch].
    """
    assert len(batch) == 1
    return batch[0]


def load_cbg_poi_data(directory, use_key_demographics=False, load_distances=True):
    """
    Every data directory should have:
    - index.pkl (a dictionary with keys 'cbgs', 'pois', 'weeks')
    - visits.pkl (a list of sparse matrices, each of size n_cbgs x n_pois)
    - distances.pkl (a numpy array of size n_cbgs x n_pois)
    - cbg_attrs.csv (a pandas dataframe with n_cbgs rows and a column per CBG attribute)
    - poi_attrs.csv (a pandas dataframe with n_pois rows and a column per POI attribute)
    """
    with open(os.path.join(directory, 'index.pkl'), 'rb') as f:
        indices = pickle.load(f)
    
    with open(os.path.join(directory, 'visits.pkl'), 'rb') as f:
        visits = pickle.load(f)
    
    # this can be a very big file (dense matrix of num_cbgs x num_pois) so we don't always load it
    if load_distances:  
        with open(os.path.join(directory, 'distances.pkl'), 'rb') as f:
            distances = pickle.load(f)
    else:
        distances = None
    
    if use_key_demographics:  # only load 10 key demographics (age, race, income)
        cbg_attrs = helper.load_key_cbg_demographics()
        cbg_attrs = cbg_attrs.loc[indices['cbgs']]
    else:
        cbg_attrs = pd.read_csv(os.path.join(directory, 'cbg_attrs.csv')).set_index('census_block_group')
    
    poi_attrs = pd.read_csv(os.path.join(directory, 'poi_attrs.csv')).set_index('safegraph_place_id')
    cols_to_keep = [c for c in poi_attrs if c not in ['latitude', 'longitude']]  # drop lat,lon as features
    poi_attrs = poi_attrs[cols_to_keep]
    
    cbg_device_counts = pd.read_csv(os.path.join(cu.PATH_TO_CBG_POI_DATA, 'cbg_device_counts.csv')).set_index('census_block_group')
    cbg_device_counts = cbg_device_counts.loc[indices['cbgs']][indices['weeks']]
    
    return indices, visits, distances, cbg_attrs, poi_attrs, cbg_device_counts
        
    
if __name__ == '__main__':
    directory = os.path.join(cu.PATH_TO_CBG_POI_DATA, 'bay_area')
    start_date = '2020-03-02'
    end_date = '2020-03-30'
    dset = CBGPOIDataset(directory=directory, start_date=start_date, end_date=end_date, only_use_tiers=False)
    print(dset[4])
