import numpy as np
from scipy.stats import multivariate_normal

def emulate_measurements(sigma_v_truth, 
        measurement_error_percent=None,measurement_error_kmpersec=None):
    """NOTE: this is written in terms of sigma_v, but this can be used for
        time-delays as well :) 

    Args:
        sigma_v_truth (n_lenses,n_kin_bins)
        measurement_error_percent (float): a fractional value
            (i.e. use 0.01 to represent 1% error)
        measurement_error_kmpersec
    Returns: 
        sigma_v_measured (n_lenses,n_kin_bins), 
        sigma_v_measurement_prec (n_lenses,n_kin_bins, n_kin_bins)
    """

    # must define either measurement_error_percent or measurement_error_kmpersec
    if measurement_error_percent is not None and measurement_error_kmpersec is not None:
        raise ValueError('Must specify kin. meas. error in either percent OR kmpersec (not both)')
    elif measurement_error_percent is None and measurement_error_kmpersec is None:
        raise ValueError('Must specify kin. meas. error in either percent OR kmpersec')
    
    # grab number of kin bins
    num_kin_bins = np.shape(sigma_v_truth)[-1]
    num_lenses = np.shape(sigma_v_truth)[0]

    # construct array of measurement errors
    # meas_sigma has shape (n_lenses,n_kin_bins)
    if measurement_error_percent is not None:
        meas_sigma = measurement_error_percent*np.abs(sigma_v_truth) # must be a positive number!!
    elif measurement_error_kmpersec is not None:
        meas_sigma = measurement_error_kmpersec*np.ones(np.shape(sigma_v_truth))  


    # construct covariance / precision matrices (NOTE: diagonal for now!)
    cov_sigma_v = np.repeat(np.eye(num_kin_bins)[np.newaxis,:,:],repeats=num_lenses,axis=0)
    for bin in range(0,num_kin_bins):
        cov_sigma_v[:,bin,bin] = meas_sigma[:,bin]**2
    prec_sigma_v = np.linalg.inv(cov_sigma_v)

    # emulate mean measurement off of the truth
    sigma_v_measured = np.empty(np.shape(sigma_v_truth))
    for lens_idx in range(0,num_lenses):
        sigma_v_measured[lens_idx] = multivariate_normal.rvs(
            mean=sigma_v_truth[lens_idx],cov=cov_sigma_v[lens_idx])

    # save to data vectors dict
    return sigma_v_measured, prec_sigma_v