# This helper code finishes the emulation and prepares data vectors for input to tdc_sampler
import numpy as np
import pandas as pd
import h5py
from scipy.stats import norm, multivariate_normal, uniform
import sys
import os
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, '../../..'))
from Utils.data_vector_utils import emulate_measurements


# NOTE: hard-coded modeling priors (can change later)
GAMMA_LENS_PRIOR = norm(loc=2.,scale=0.2).logpdf
BETA_ANI_PRIOR = norm(loc=0.,scale=0.2).logpdf
COSMO_MODEL = 'LCDM_lambda_int_beta_ani'


def retrieve_truth_fpd(metadata_df,num_td):
    if num_td == 1: 
        fpd_truth = metadata_df.loc[:,['fpd01']].to_numpy()
    elif num_td == 3:
        fpd_truth = metadata_df.loc[:,['fpd01','fpd02','fpd03']].to_numpy()

    return fpd_truth

def retrieve_truth_lp(metadata_df):

    return metadata_df.loc[:,['main_deflector_parameters_theta_E',
    'main_deflector_parameters_gamma1',
    'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
    'main_deflector_parameters_e1','main_deflector_parameters_e2',
    'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
    'source_parameters_center_x','source_parameters_center_y']].to_numpy()


def retrieve_truth_td(metadata_df,num_td):
    if num_td == 1: 
        # NOTE: will this keep the last dim=(..,..,1) ? 
        td_truth = metadata_df.loc[:,['td01']].to_numpy()
    elif num_td == 3:
        td_truth = metadata_df.loc[:,['td01','td02','td03']].to_numpy()

    return td_truth

def retrieve_truth_kin(metadata_df,kinematic_type):
    """
    Args:
        metadata_df
        kinematic_type (string): Options are: '4MOST', 'MUSE', or 'NIRSPEC'
    """

    # 4MOST
    if kinematic_type == '4MOST':
        return metadata_df.loc[:,['sigma_v_4MOST_kmpersec']].to_numpy()

    # MUSE
    if kinematic_type == 'MUSE':
        muse_keys = ['sigma_v_MUSE_bin'+str(j)+'_kmpersec' for j in range(0,3)]
        return metadata_df.loc[:,muse_keys].to_numpy()

    # NIRSPEC
    if kinematic_type == 'NIRSPEC':
        nirspec_keys = ['sigma_v_NIRSPEC_bin'+str(j)+'_kmpersec' for j in range(0,10)]
        return metadata_df.loc[:,nirspec_keys].to_numpy()


    raise ValueError("kinematic_type not supported")


def gaussianize(input_samps,gt_for_debiasing=None):
    """takes in input samples, fits a Gaussian, and returns Mu,Cov of that 
        Gaussian
    Args:
        input_samps (n_input_samps,n_params)
        gt_for_debiasing (n_params): Default is None, no de-biasing. We must 
            have access to ground truth to de-bias.
    Returns:
        Mu,Cov
    """
    Mu = np.mean(input_samps,axis=0)
    Cov = np.cov(input_samps,rowvar=False)

    # de-bias if requested
    if gt_for_debiasing is not None:
        Mu = multivariate_normal.rvs(mean=gt_for_debiasing,cov=Cov)

    return Mu,Cov


def gaussianize_samples(input_samps,num_gaussian_samps=5000,
    gt_for_debiasing=None):
    """takes in input samples, fits a Gaussian, and returns new samples
        from that Gaussian
    Args:
        input_samps (n_input_samps,n_params)
        gt_for_debiasing (n_params): Default is None, no de-biasing. We must 
            have access to ground truth to de-bias.
    Returns:
        output_samps (n_gaussian_samps,n_params)
    """

    Mu,Cov = gaussianize(input_samps,gt_for_debiasing)

    gaussianized_samps = multivariate_normal.rvs(mean=Mu,cov=Cov,
        size=num_gaussian_samps)

    return gaussianized_samps

def gaussianize_scale_and_debias(input_samps,
    desired_param_prec,desired_param_idx,gt_for_debiasing,
    num_gaussian_samps=5000):
    """
    Args:
        input_samps (n_input_samps,n_params)
        desired_param_prec: fractional percent error on a chosen parameter
        desired_param_idx: index of this parameter in input_samps and 
            gt_for_debiasing
        gt_for_debiasing: must have access to ground truth to de-bias
        num_gaussian_samps (int)
    """


    Mu = np.mean(input_samps,axis=0)
    Cov = np.cov(input_samps,rowvar=False)

    # re-scale based on desired precision of a chosen parameter
    current_std = np.sqrt(Cov[desired_param_idx,desired_param_idx])
    desired_std_gamma = gt_for_debiasing[desired_param_idx]*desired_param_prec
    Cov *= (desired_std_gamma/current_std)**2

    # debiasing is required to avoid over/under confident posteriors
    Mu = multivariate_normal.rvs(mean=gt_for_debiasing,cov=Cov)

    # now, take new samples!
    gaussianized_samps = multivariate_normal.rvs(mean=Mu,cov=Cov,
        size=num_gaussian_samps)

    return gaussianized_samps


#############################
# Construct likelihood object
#############################

def create_static_data_vectors(
    posteriors_h5_file,metadata_file,catalog_idxs,
    cosmo_model,
    td_meas_error_percent=None,td_meas_error_days=None,
    kappa_ext_meas_error_value=0.05,
    kinematic_type=None,
    kin_meas_error_percent=None,kin_meas_error_kmpersec=None,
    num_gaussianized_samps=None,
    lens_params_nu_int_means=None,
    lens_params_nu_int_stddevs=None,
    log_prob_beta_ani_nu_int=None,
    debias_models=False):
    """
    Args:
        posteriors_h5_file ()
        metadata_file ()
        catalog_idxs (np.array[int]): catalog indices of the subset of lenses 
            used from these files
        cosmo_model (str): Options are 'LCDM', etc...
        kinematic_type (string): Default=None. Options are: '4MOST', 'MUSE', or 'NIRSPEC'
        num_gaussianized_samps (int): Default=None (use samples as is). If specified,
            a Gaussian will be fit to the provided samples, and a new batch of 
            |num_gaussianized_samps| samples will be drawn from that distribution 
        lens_params_nu_int_means ([float]): Means of the training prior, default=None.
            Must have dim (n_lens_params)
        lens_params_nu_int_stddevs ([float]): Std. Devs. of the training prior,default=None
            Must have dim (n_lens_params)
        log_prob_beta_ani_nu_int (callable): default=None
    
    Returns: 
        (dict) data_vector_dict = 
            {   'td_measured':,
                'td_likelihood_prec':,
                'sigma_v_measured':,
                'sigma_v_likelihood_prec':,
                'fpd_samples':,
                'beta_ani_samples':,
                'kin_pred_samples':,
                'kappa_ext_samples':,
                'z_lens':,
                'z_src':,
            }
        
    """

    if debias_models and kinematic_type is not None:
        raise ValueError('de-biasing only works in TD-only case (for now)')

    # load in from posteriors file
    with h5py.File(posteriors_h5_file, "r") as h5:

        # set-up indexing
        h5_catalog_idxs_byte = h5['catalog_idxs'][:]
        h5_catalog_idxs = [ b_string.decode('utf-8') for b_string in h5_catalog_idxs_byte ]
        my_idxs = np.isin(h5_catalog_idxs,catalog_idxs)

        #print('requested catalog idxs: ', catalog_idxs)
        #print('total matched indices: ', np.sum(my_idxs))

        fpd_samps = h5['fpd_samps'][my_idxs]
        lens_param_samps = h5['lens_param_samps'][my_idxs]
        beta_ani_samps = h5['beta_ani_samps'][my_idxs]
        h5_catalog_idxs = h5['catalog_idxs'][my_idxs]

        # pull c_sqrtJ_samps based on kinematic type
        if kinematic_type is not None:
            if kinematic_type == '4MOST':
                c_sqrtJ_samps = h5['c_sqrtJ_samps'][my_idxs]
            elif kinematic_type == 'MUSE':
                c_sqrtJ_samps = h5['MUSE_c_sqrtJ_samps'][my_idxs]
            elif kinematic_type == 'NIRSPEC':
                c_sqrtJ_samps = h5['NIRSPEC_c_sqrtJ_samps'][my_idxs]
            else:
                raise ValueError("kinematic_type not supported")
            
            num_kin_bins = np.shape(c_sqrtJ_samps)[-1]
            
    # set up some sizes
    num_lenses = np.shape(fpd_samps)[0]
    num_td = np.shape(fpd_samps)[-1]
    num_lp = np.shape(lens_param_samps)[-1]
            
    # load in from metadata file
    all_metadata_df = pd.read_csv(metadata_file)
    # set up indexing
    metadata_catalog_idxs = all_metadata_df.loc[:,'catalog_idx']
    metadata_idx = np.isin(metadata_catalog_idxs,catalog_idxs)
    metadata_df = all_metadata_df.loc[metadata_idx]

    # emulate time-delay measurement
    td_truth = retrieve_truth_td(metadata_df, num_td)
    td_meas, td_meas_prec = emulate_measurements(td_truth, 
        td_meas_error_percent,td_meas_error_days)
    
    # emulate kappa_ext
    if num_gaussianized_samps is not None:
        num_fpd_samps = num_gaussianized_samps
    else:
        num_fpd_samps = np.shape[fpd_samps][1]

    kappa_ext_samps = norm.rvs(loc=0.,
        scale=kappa_ext_meas_error_value,
        size=(num_lenses,num_fpd_samps))
    
    # emulate kinematics
    if kinematic_type is not None:
        kin_truth = retrieve_truth_kin(metadata_df,kinematic_type)
        sigma_v_meas,sigma_v_meas_prec = emulate_measurements(kin_truth,
            kin_meas_error_percent,kin_meas_error_kmpersec)

    # gaussianize samples if requested
    if num_gaussianized_samps is not None:
        to_gaussianize_input = []
        # fpds
        fpd_truth = retrieve_truth_fpd(metadata_df,num_td)
        for i in range(0,num_td):
            to_gaussianize_input.append(fpd_samps[:,:,i])
        # TODO: all lens params
        lp_truth = retrieve_truth_lp(metadata_df)
        for lp in range(0,num_lp):
            to_gaussianize_input.append(lens_param_samps[:,:,lp])

        # kinematics
        if kinematic_type is not None:
            # beta_ani
            to_gaussianize_input.append(beta_ani_samps)
            beta_idx = num_td+num_lp
            # sigma_v bins
            for j in range(0,num_kin_bins):
                to_gaussianize_input.append(c_sqrtJ_samps[:,:,j])

        to_gaussianize_input = np.asarray(to_gaussianize_input)
        # switch 1st dim to last dim (parameters dim)
        input_samps = np.transpose(to_gaussianize_input,axes=(1,2,0))
        # now gaussianize
        gaussian_samps = np.empty((num_lenses,
            num_gaussianized_samps,np.shape(input_samps)[-1]))
        for l_idx in range(0,num_lenses):
            tdonly_truth = None
            if debias_models:
                tdonly_truth = np.concatenate((fpd_truth[l_idx],lp_truth[l_idx]))
            gaussian_samps[l_idx] = gaussianize_samples(
                input_samps[l_idx],num_gaussianized_samps,
                gt_for_debiasing=tdonly_truth)


    # TODO: get this into format for likelihood ... (add repeated axes, etc.)
    data_vector_dict ={}

    # catalog idxs
    data_vector_dict['catalog_idxs'] = np.asarray(catalog_idxs)

    # redshifts
    data_vector_dict['z_lens'] = metadata_df.loc[:,'main_deflector_parameters_z_lens'].to_numpy()
    data_vector_dict['z_src'] = metadata_df.loc[:,'source_parameters_z_source'].to_numpy()

    # samples already with fpd dimension
    if num_gaussianized_samps is not None:
        data_vector_dict['fpd_samples'] = gaussian_samps[:,:,0:num_td]
        data_vector_dict['lens_param_samples'] = gaussian_samps[:,:,num_td:(num_td+num_lp)]
        data_vector_dict['kappa_ext_samples'] = kappa_ext_samps

        if kinematic_type is not None:
            data_vector_dict['beta_ani_samples'] = gaussian_samps[:,:,beta_idx]
            data_vector_dict['kin_pred_samples'] = gaussian_samps[:,:,-num_kin_bins:]
    else:
        raise ValueError("not implemented")
    
    # time-delays
    # pad with a 2nd batch dim for # of fpd samples
    data_vector_dict['td_measured'] = np.repeat(td_meas[:, np.newaxis, :],
        num_fpd_samps, axis=1)
    data_vector_dict['td_likelihood_prec'] = np.repeat(td_meas_prec[:, np.newaxis, :, :],
        num_fpd_samps, axis=1)
    data_vector_dict['td_likelihood_prefactors'] = np.log( (1/(2*np.pi)**(num_td/2)) / 
        np.sqrt(np.linalg.det(np.linalg.inv(data_vector_dict['td_likelihood_prec']))) )

    # save info to data_vector_dict for later use
    data_vector_dict['lens_params_nu_int_means'] = lens_params_nu_int_means
    data_vector_dict['lens_params_nu_int_stddevs'] = lens_params_nu_int_stddevs

    # setting these to None = a uniform modeling prior
    if lens_params_nu_int_means is None and lens_params_nu_int_stddevs is None:
        # set log prob. to zero, equivalent to a uniform prior...
        data_vector_dict['log_prob_lens_param_samps_nu_int'] = np.zeros(
            (data_vector_dict['lens_param_samples'].shape[0:2]))
        
    # diagonal Gaussian modeling prior...
    else:
        # construct multivariate normal
        log_prob_lens_params_nu_int = multivariate_normal(
            mean=lens_params_nu_int_means,
            cov=np.diag(lens_params_nu_int_stddevs**2)).logpdf
        # log prob condenses over lens params dimension
        data_vector_dict['log_prob_lens_param_samps_nu_int'] = np.empty(
            (data_vector_dict['lens_param_samples'].shape[0:2]))
        for i in range(0,num_lenses):
            # TODO: will this work with a multivariate log_prob function? this still doesn't have 
            # a condensed sample dimension?
            data_vector_dict['log_prob_lens_param_samps_nu_int'][i] = log_prob_lens_params_nu_int(
                data_vector_dict['lens_param_samples'][i])
            

    # kinematics if requested
    if kinematic_type is not None:

        # measured sigma_v
        # pad with a 2nd batch dim for # of fpd samples
        data_vector_dict['sigma_v_measured'] = np.repeat(sigma_v_meas[:, np.newaxis, :],
                num_fpd_samps, axis=1)
        data_vector_dict['sigma_v_likelihood_prec'] = np.repeat(
            sigma_v_meas_prec[:, np.newaxis, :, :],
            num_fpd_samps, axis=1)
        
        data_vector_dict['sigma_v_likelihood_prefactors'] = np.log( (1/(2*np.pi)**(num_kin_bins/2)) / 
            np.sqrt(np.linalg.det(np.linalg.inv(data_vector_dict['sigma_v_likelihood_prec']))))
        
        # beta_ani
        if log_prob_beta_ani_nu_int is None:
            # default: assume un-informative prior
            data_vector_dict['log_prob_beta_ani_samps_nu_int'] = uniform.logpdf(
                data_vector_dict['beta_ani_samples'],loc=-0.5,scale=1.)
        else:
            # user-provided modeling prior
            data_vector_dict['log_prob_beta_ani_samps_nu_int'] = np.empty(
                (data_vector_dict['beta_ani_samples'].shape))
            for i in range(0,num_lenses):
                data_vector_dict['log_prob_beta_ani_samps_nu_int'][i] = log_prob_beta_ani_nu_int(
                    data_vector_dict['beta_ani_samples'][i])


    return data_vector_dict

