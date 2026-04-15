# experiment 1.2: Gold-Only Baseline, Human-Bias Selected

import h5py
import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal

# random seed
RANDOM_SEED = 1

# path for dv file
static_dv_file = 'InferenceRuns/expC1/static_datavectors_seed'+str(RANDOM_SEED)+'.json'

# locations of samples from joint fermat/csqrt(J) posteriors
lsst_models = {
    'quads_20perc_h5_file':'DataVectors/lsst_catalog/quad_posteriors_20percfpd.h5', # LSST
    'quads_05perc_h5_file':'DataVectors/lsst_catalog/quad_posteriors_05percfpd.h5', # ROMAN
    'quads_03perc_h5_file':'DataVectors/lsst_catalog/quad_posteriors_03percfpd.h5', # ADDITIONAL

    # 
    'dbls_20perc_h5_file':'DataVectors/lsst_catalog/dbl_posteriors_20percfpd.h5', # LSST
    'dbls_05perc_h5_file':'DataVectors/lsst_catalog/dbl_posteriors_05percfpd.h5' # ROMAN
    #dbls_03perc_h5_file = 'DataVectors/lsst_catalog/from_sherlock/dbl_posteriors_03percfpd.h5' # ADDITIONAL
}

roman_models = {
    'quads_05perc_h5_file':'DataVectors/roman_catalog/quad_posteriors_05percfpd.h5', # ROMAN
    #'quads_03perc_h5_file':'DataVectors/lsst_catalog/from_sherlock/quad_posteriors_03percfpd.h5', # ADDITIONAL

    # 
    'dbls_05perc_h5_file':'DataVectors/roman_catalog/dbl_posteriors_05percfpd.h5' # ROMAN
    #dbls_03perc_h5_file = 'DataVectors/lsst_catalog/from_sherlock/dbl_posteriors_03percfpd.h5' # ADDITIONAL
}


lsst_metadata_file = 'DataVectors/lsst_catalog/truth_metadata_popsigma05.csv'
roman_metadata_file = 'DataVectors/roman_catalog/truth_metadata_popsigma05.csv'

NUM_FPD_SAMPS = 5000
NUM_MCMC_EPOCHS = 10
NUM_MCMC_WALKERS = 50
COSMO_MODEL = 'LCDM_lambda_int_beta_ani'
HI_REWEIGHTING = False
OMEGA_M_PRIOR = True # !!!! INFORMATIVE Omega_M prior when True !!!!!
# this beta_ani prior does affect the inference.
BETA_ANI_PRIOR = norm(loc=0.,scale=0.2).logpdf
# where to store the chain...
BACKEND_PATH = 'InferenceRuns/expC1/LCDM_seed'+str(RANDOM_SEED)+'_backend.h5'
RESET_BACKEND=True

# assumed modeling prior
mu_lp_gold = np.asarray([0.85,0.,0.,2.09,0.,0.,0.,0.,0.,0.]) # hst_norms.csv
stddev_lp_gold = np.asarray([0.28,0.06,0.06,0.16,0.20,0.20,0.06,0.06,0.34,0.34])

# truth information for those indices
lsst_df = pd.read_csv(lsst_metadata_file)
# NOTE: excluding bad lenses (bad paltas model, NaN kinematics computation, etc.)
lsst_bad_idx = [ 'l00676' , # <- this one is a bad paltas model 
    'l00113', 'l00322', 'l00389', 'l00525', 'l00586', 'l00659', 'l00823', 'l00902', # <- these are all 
    'l01242' ,'l01297', 'l01312', 'l01385', 'l01410' ,'l01413', 'l01434', 'l01559', # Nan kinematics
    'l01580', 'l01834', 'l01893', 'l01915', 'l02010', 'l02040', 'l02100', 'l02101',
    'l02141', 'l02310', 'l02573', 'l02575' ,'l02586', 'l02587']

lsst_df = lsst_df[~lsst_df['catalog_idx'].isin(lsst_bad_idx)].reset_index(drop=True)
# track remaining catalog_idxs
lsst_df_catalog_idxs = lsst_df.loc[:,'catalog_idx'].to_numpy()

# truth information for those indices
roman_df = pd.read_csv(roman_metadata_file)
# NOTE: excluding bad lenses (bad paltas model, NaN kinematics computation, etc.)
roman_bad_idx = [ 'r01414', 'r01422', 'r02878', 'r03451', 'r04294', 'r04965'] # <- bad paltas models 
roman_df = roman_df[~roman_df['catalog_idx'].isin(lsst_bad_idx)].reset_index(drop=True)
# track remaining catalog_idxs
roman_df_catalog_idxs = roman_df.loc[:,'catalog_idx'].to_numpy()

#########################
# Human selection cuts!!
#########################

# use the random seed
np.random.seed(RANDOM_SEED)

################
# 8 TDCOSMO LENSES (all quads)
################
num_tdcosmo_quads = 8
nirspec_quads_avail = np.where(
    (lsst_df['point_source_parameters_num_images'].to_numpy() == 4) &
    ((np.abs(lsst_df['td01'].to_numpy()) > 10.) | 
     (np.abs(lsst_df['td02'].to_numpy()) > 10.) | 
     (np.abs(lsst_df['td03'].to_numpy()) > 10.)) &
    (lsst_df['lens_light_parameters_mag_app'].to_numpy() < 24.) &
    (lsst_df['source_parameters_mag_app'].to_numpy() < 24.)
)[0]
# take the catalog idxs you want
#print('num tdcosmo quads available: ', nirspec_quads_avail)
catalog_idx_avail = lsst_df.loc[nirspec_quads_avail,'catalog_idx'].to_numpy()
tdcosmo_quads_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_tdcosmo_quads,replace=False)
# then remove them from the dataframe
lsst_df = lsst_df[~lsst_df['catalog_idx'].isin(tdcosmo_quads_catalog_idxs)].reset_index(drop=True)


# 800 LSST lenses. 80 are quads, 720 are doubles (10% quad fraction starting from 800)
num_lsst_aper_quads = 40
num_lsst_only_quads = 40
num_lsst_aper_dbls = 360 # is this too high?
num_lsst_only_dbls = 360

################
# LSST Quads with aperture-kin (40)
################
lsst_aper_quads_avail = np.where(
    (lsst_df['point_source_parameters_num_images'].to_numpy() == 4) &
    ((np.abs(lsst_df['td01'].to_numpy()) > 10.) |  # 10-day LSST time-delay threshold
     (np.abs(lsst_df['td02'].to_numpy()) > 10.) | 
     (np.abs(lsst_df['td03'].to_numpy()) > 10.)) &
    (lsst_df['lens_light_parameters_mag_app'].to_numpy() < 22.) # Bright enough for kinematics
)[0]
# edge case: not enough quads that pass this criteria, add to lsst-only bucket
if len(lsst_aper_quads_avail)<num_lsst_aper_quads:
    num_lsst_only_quads += (num_lsst_aper_quads - len(lsst_aper_quads_avail))
    num_lsst_aper_quads = len(lsst_aper_quads_avail)
# now sample the catalog idxs
catalog_idx_avail = lsst_df.loc[lsst_aper_quads_avail,'catalog_idx'].to_numpy()
lsst_aper_quads_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_lsst_aper_quads,replace=False)
print('Including %d Lsst+Aper Quads'%(len(lsst_aper_quads_catalog_idxs)))
# then remove them from the dataframe
lsst_df = lsst_df[~lsst_df['catalog_idx'].isin(lsst_aper_quads_catalog_idxs)].reset_index(drop=True)

################
# LSST Quads w/out aperture-kin (40)
################
lsst_only_quads_avail = np.where(
    (lsst_df['point_source_parameters_num_images'].to_numpy() == 4) &
    ((np.abs(lsst_df['td01'].to_numpy()) > 10.) |  # 10-day LSST time-delay threshold
     (np.abs(lsst_df['td02'].to_numpy()) > 10.) | 
     (np.abs(lsst_df['td03'].to_numpy()) > 10.))
)[0]
# edge case: not enough quads that pass this criteria, add to lsst-only-doubles bucket
if len(lsst_only_quads_avail)<num_lsst_only_quads:
    num_lsst_only_dbls += (num_lsst_only_quads - len(lsst_only_quads_avail))
    num_lsst_only_quads = len(lsst_only_quads_avail)
# now sample the catalog idxs
catalog_idx_avail = lsst_df.loc[lsst_only_quads_avail,'catalog_idx'].to_numpy()
lsst_only_quads_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_lsst_only_quads,replace=False)
print('Including %d Lsst-Only Quads'%(len(lsst_only_quads_catalog_idxs)))
# then remove them from the dataframe
lsst_df = lsst_df[~lsst_df['catalog_idx'].isin(lsst_only_quads_catalog_idxs)].reset_index(drop=True)

################
# LSST Doubles with aperture-kin (360)
################
lsst_aper_dbls_avail = np.where(
    (lsst_df['point_source_parameters_num_images'].to_numpy() == 2) &
    (np.abs(lsst_df['td01'].to_numpy()) > 10.) & # 10-day LSST time-delay threshold)
    (lsst_df['lens_light_parameters_mag_app'].to_numpy() < 22.) # Bright enough for kinematics
)[0]
# edge case: not enough dbls that pass this criteria, add to lsst-only-doubles bucket
if len(lsst_aper_dbls_avail)<num_lsst_aper_dbls:
    num_lsst_only_dbls += (num_lsst_aper_dbls - len(lsst_aper_dbls_avail))
    num_lsst_aper_dbls = len(lsst_aper_dbls_avail)
# now sample the catalog idxs
catalog_idx_avail = lsst_df.loc[lsst_aper_dbls_avail,'catalog_idx'].to_numpy()
lsst_aper_dbls_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_lsst_aper_dbls,replace=False)
print('Including %d Lsst+Aper Dbls'%(len(lsst_aper_dbls_catalog_idxs)))
# then remove them from the dataframe
lsst_df = lsst_df[~lsst_df['catalog_idx'].isin(lsst_aper_dbls_catalog_idxs)].reset_index(drop=True)

################
# LSST Doubles w/out aperture-kin (360)
################
lsst_only_dbls_avail = np.where(
    (lsst_df['point_source_parameters_num_images'].to_numpy() == 2) &
    (np.abs(lsst_df['td01'].to_numpy()) > 10.)  # 10-day LSST time-delay threshold)
)[0]
# edge case: not enough dbls that pass this criteria, total sample will be smaller
if len(lsst_only_dbls_avail)<num_lsst_aper_dbls:
    num_lsst_only_dbls = len(lsst_only_dbls_avail)
# now sample the catalog idxs
catalog_idx_avail = lsst_df.loc[lsst_only_dbls_avail,'catalog_idx'].to_numpy()
lsst_only_dbls_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_lsst_only_dbls,replace=False)
print('Including %d Lsst-Only Dbls'%(len(lsst_only_dbls_catalog_idxs)))
# then remove them from the dataframe
lsst_df = lsst_df[~lsst_df['catalog_idx'].isin(lsst_only_dbls_catalog_idxs)].reset_index(drop=True)


# 400 total that are faint / smaller sep. (5% quad fraction in this regime)
num_roman_faint_quads = 20
num_roman_faint_dbls = 380

#########################
# Roman Small/Faint Quads
#########################

roman_faint_quads_avail = np.where(
    (roman_df['point_source_parameters_num_images'].to_numpy() == 4) &
    ((np.abs(roman_df['td01'].to_numpy()) > 10.) | 
     (np.abs(roman_df['td02'].to_numpy()) > 10.) | 
     (np.abs(roman_df['td03'].to_numpy()) > 10.)) &
    # look for lenses too faint for LSST time-delays
    (roman_df['second_brightest_image_ps_mag_i'].to_numpy() > 23.3) & # require 2nd_im_mag_cut < 23.3 (in LSST i-band)
    (roman_df['second_brightest_image_ps_mag_i'].to_numpy() < 25)
)[0]
# now sample the catalog idxs
catalog_idx_avail = roman_df.loc[roman_faint_quads_avail,'catalog_idx'].to_numpy()
roman_faint_quads_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_roman_faint_quads,replace=False)
print('Including %d Roman-Faint Quads'%(len(roman_faint_quads_catalog_idxs)))
# then remove them from the dataframe
roman_df = roman_df[~roman_df['catalog_idx'].isin(roman_faint_quads_catalog_idxs)].reset_index(drop=True)

#########################
# Roman Small/Faint Dbls
#########################
roman_faint_dbls_avail = np.where(
    (roman_df['point_source_parameters_num_images'].to_numpy() == 2) &
    (np.abs(roman_df['td01'].to_numpy()) > 10.) &
    # look for lenses too faint for LSST time-delays
    (roman_df['second_brightest_image_ps_mag_i'].to_numpy() > 23.3) & # require 2nd_im_mag_cut < 23.3 (in LSST i-band)
    (roman_df['second_brightest_image_ps_mag_i'].to_numpy() < 25)
)[0]
# now sample the catalog idxs
catalog_idx_avail = roman_df.loc[roman_faint_dbls_avail,'catalog_idx'].to_numpy()
roman_faint_dbls_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_roman_faint_dbls,replace=False)
print('Including %d Roman-Faint Dbls'%(len(roman_faint_dbls_catalog_idxs)))
# then remove them from the dataframe
roman_df = roman_df[~roman_df['catalog_idx'].isin(roman_faint_dbls_catalog_idxs)].reset_index(drop=True)

########################################################
# Upgrade 1/4 of these lenses with Roman imaging (5% fpd)
########################################################

# integer division for indexing
num_rom_aper_quads = len(lsst_aper_quads_catalog_idxs) // 4
num_rom_only_quads = len(lsst_only_quads_catalog_idxs) // 4
num_rom_aper_dbls = len(lsst_aper_dbls_catalog_idxs) // 4
num_rom_only_dbls = len(lsst_only_dbls_catalog_idxs) // 4

##############################
# Set-up inference configs
##############################
likelihood_configs = {

    # TDCOSMO counterpart (8 IFU lenses)
    'tdcosmo_quads':{
        'posteriors_h5_file':lsst_models['quads_03perc_h5_file'],
        'metadata_file':lsst_metadata_file,
        'catalog_idxs':tdcosmo_quads_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':1., # CHANGED TO 1-DAY MEAS. ERROR ALWAYS
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'NIRSPEC',
        'kin_meas_error_percent':0.03, # CHANGED TO 3% KINEMATICS
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # lsst quads with aperture kin, UPGRADED with Roman imaging (~10 lenses)
    'roman_aper_quads':{
        'posteriors_h5_file':lsst_models['quads_05perc_h5_file'], # UPGRADE TO 5% fpd
        'metadata_file':lsst_metadata_file,
        'catalog_idxs':lsst_aper_quads_catalog_idxs[:num_rom_aper_quads],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':3., # 3-Day LSST Meas. Precision
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.03, # 3% single-aperture
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # lsst quads with aperture kin (~30 lenses)
    'lsst_aper_quads':{
        'posteriors_h5_file':lsst_models['quads_20perc_h5_file'],
        'metadata_file':lsst_metadata_file,
        'catalog_idxs':lsst_aper_quads_catalog_idxs[num_rom_aper_quads:],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':3., # 5-Day LSST Meas. Precision
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.03, # 3% single-aperture
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # quads without kinematics
    'roman_only_quads':{
        'posteriors_h5_file':lsst_models['quads_05perc_h5_file'], # UPGRADE TO 5% fpd
        'metadata_file':lsst_metadata_file,
        'catalog_idxs':lsst_only_quads_catalog_idxs[:num_rom_only_quads],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':3., # 3-Day LSST Meas. Precision
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    'lsst_only_quads':{
        'posteriors_h5_file':lsst_models['quads_20perc_h5_file'],
        'metadata_file':lsst_metadata_file,
        'catalog_idxs':lsst_only_quads_catalog_idxs[num_rom_only_quads:],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':3., # 3-Day LSST Meas. Precision
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # lsst dbls with aperture kin (40 lenses)
    'roman_aper_dbls':{
        'posteriors_h5_file':lsst_models['dbls_05perc_h5_file'], # UPGRADE TO 5% fpd
        'metadata_file':lsst_metadata_file,
        'catalog_idxs':lsst_aper_dbls_catalog_idxs[:num_rom_aper_dbls],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':3., # 3-Day LSST Meas. Precision
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.03, # 3% single-aperture
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    'lsst_aper_dbls':{
        'posteriors_h5_file':lsst_models['dbls_20perc_h5_file'],
        'metadata_file':lsst_metadata_file,
        'catalog_idxs':lsst_aper_dbls_catalog_idxs[num_rom_aper_dbls:],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':3., # 3-Day LSST Meas. Precision
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.03, # 3% single-aperture
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # doubles with no kinematics
    'roman_only_dbls':{
        'posteriors_h5_file':lsst_models['dbls_05perc_h5_file'], # UPGRADE TO 5% fpd
        'metadata_file':lsst_metadata_file,
        'catalog_idxs':lsst_only_dbls_catalog_idxs[:num_rom_only_dbls],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':3., # 3-Day LSST Meas. Precision
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    'lsst_only_dbls':{
        'posteriors_h5_file':lsst_models['dbls_20perc_h5_file'],
        'metadata_file':lsst_metadata_file,
        'catalog_idxs':lsst_only_dbls_catalog_idxs[num_rom_only_dbls:],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':3., # 3-Day LSST Meas. Precision
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    }, 

    # faint roman quads
    'roman_faint_quads':{
        'posteriors_h5_file':roman_models['quads_05perc_h5_file'], # UPGRADE TO 5% fpd
        'metadata_file':roman_metadata_file,
        'catalog_idxs':roman_faint_quads_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':3., # 3-Day LSST Meas. Precision
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # faint roman dbls
    'roman_faint_dbls':{
        'posteriors_h5_file':roman_models['dbls_05perc_h5_file'], # UPGRADE TO 5% fpd
        'metadata_file':roman_metadata_file,
        'catalog_idxs':roman_faint_dbls_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':3., # 3-Day LSST Meas. Precision
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },
}