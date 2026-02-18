import numpy as np
import pandas as pd
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
# my own utils
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import Utils.tdc_utils as tdc_utils
import Modeling.Kinematics.galkin_utils as galkin_utils
import batched_fermatpot

"""
Modifies a paltas-formatted metadata_df (pandas.DataFrame)
- Assumes image positions are already pre-computed.
"""

def populate_fermat_differences(metadata_df):
    """Populates ground truth fermat potential differences at image positions
    Args:
    Returns:
        modifies metadata_df in place to add fpd_01 (& fpd02,fpd03 for quads)
    """

    num_images = metadata_df.loc[:,'point_source_parameters_num_images'].to_numpy()
    dbls_idxs = np.where(num_images == 2.)[0]
    quads_idxs = np.where(num_images == 4.)[0]

    #fill in fpd01 for doubles
    x_im_dbls = metadata_df.loc[dbls_idxs,
        ['point_source_parameters_x_image_0',
         'point_source_parameters_x_image_1']].to_numpy().astype(float)
    y_im_dbls = metadata_df.loc[dbls_idxs,
        ['point_source_parameters_y_image_0',
         'point_source_parameters_y_image_1']].to_numpy().astype(float)
    lens_params_dbls = metadata_df.loc[dbls_idxs,
        ['main_deflector_parameters_theta_E',
        'main_deflector_parameters_gamma1',
        'main_deflector_parameters_gamma2', 
        'main_deflector_parameters_gamma', 
        'main_deflector_parameters_e1', 
        'main_deflector_parameters_e2',
        'main_deflector_parameters_center_x', 
        'main_deflector_parameters_center_y']].to_numpy().astype(float)
    x_src_dbls = metadata_df.loc[dbls_idxs,
        ['source_parameters_center_x']].to_numpy().astype(float)
    y_src_dbls = metadata_df.loc[dbls_idxs,
        ['source_parameters_center_y']].to_numpy().astype(float)
    for i in range(0,len(dbls_idxs)):
        dbls_fermatpot = batched_fermatpot.eplshear_fp_samples(x_im_dbls[i],
            y_im_dbls[i],[lens_params_dbls[i]],x_src_dbls[i],y_src_dbls[i])
        # write in new values!
        metadata_df.loc[dbls_idxs[i],'fpd01'] = dbls_fermatpot[0][0] - dbls_fermatpot[0][1]

    #fill in fpd01,fpd02,fpd03 for quads
    x_im_quads = metadata_df.loc[quads_idxs,
        ['point_source_parameters_x_image_0',
         'point_source_parameters_x_image_1',
         'point_source_parameters_x_image_2',
         'point_source_parameters_x_image_3']].to_numpy().astype(float)
    y_im_quads = metadata_df.loc[quads_idxs,
        ['point_source_parameters_y_image_0',
         'point_source_parameters_y_image_1',
         'point_source_parameters_y_image_2',
         'point_source_parameters_y_image_3']].to_numpy().astype(float)
    lens_params_quads = metadata_df.loc[quads_idxs,
        ['main_deflector_parameters_theta_E',
        'main_deflector_parameters_gamma1',
        'main_deflector_parameters_gamma2', 
        'main_deflector_parameters_gamma', 
        'main_deflector_parameters_e1', 
        'main_deflector_parameters_e2',
        'main_deflector_parameters_center_x', 
        'main_deflector_parameters_center_y']].to_numpy().astype(float)
    x_src_quads = metadata_df.loc[quads_idxs,
        ['source_parameters_center_x']].to_numpy().astype(float)
    y_src_quads = metadata_df.loc[quads_idxs,
        ['source_parameters_center_y']].to_numpy().astype(float)
    
    for i in range(0,len(quads_idxs)):
        quads_fermatpot = batched_fermatpot.eplshear_fp_samples(x_im_quads[i],
            y_im_quads[i],[lens_params_quads[i]],x_src_quads[i],y_src_quads[i])
        # write in new values here!!
        for j in range(1,4):
            column_name = 'fpd0'+str(j)
            metadata_df.loc[quads_idxs[i], column_name] = (
                quads_fermatpot[0][0] - quads_fermatpot[0][j])
            

def populate_truth_Ddt_timedelays(metadata_df,gt_cosmo_astropy):
    """Populate truth time delay distances (Ddt) using ground truth 
        redshifts & cosmology. Then, use fpds and Ddts to fill in time-delays.

    Returns: 
        modifies metadata_df in place!
    """

    truth_Ddt = np.empty((len(metadata_df)))
    for r in range(0,len(metadata_df)):
        Ddt = tdc_utils.ddt_from_redshifts(gt_cosmo_astropy,
            metadata_df.loc[r,'main_deflector_parameters_z_lens'],
            metadata_df.loc[r,'source_parameters_z_source'])
        truth_Ddt[r] = Ddt.value
        
    metadata_df['Ddt_Mpc'] = truth_Ddt

    # for doubles, will just write in a nan...
    for j in range(0,3):
        td_truth = tdc_utils.td_from_ddt_fpd(
            metadata_df['Ddt_Mpc'],
            metadata_df['fpd0'+str(j+1)])
        # lambda_int scaling!!
        metadata_df['td0'+str(j+1)] = metadata_df['lambda_int']*td_truth

def populate_truth_sigma_v_4MOST(metadata_df,gt_cosmo_astropy):
    """Using ground truth lens properties + a ground truth cosmology, computes
        the velocity dispersion in the 4MOST R=0.725" aperture

    Returns:
        modifies metadata_df in place
    """

    for r in range(0,len(metadata_df)):
        sigma_v = galkin_utils.ground_truth_veldisp(
            metadata_df.loc[r,'main_deflector_parameters_theta_E'],
            metadata_df.loc[r,'main_deflector_parameters_gamma'],
            metadata_df.loc[r,'lens_light_parameters_R_sersic'],
            metadata_df.loc[r,'lens_light_parameters_n_sersic'],
            metadata_df.loc[r,'main_deflector_parameters_z_lens'],
            metadata_df.loc[r,'source_parameters_z_source'],
            gt_cosmo_astropy,
            beta_ani=metadata_df.loc[r,'beta_ani'])
        # write in the value!

        sigma_v *= np.sqrt(metadata_df.loc[r,'lambda_int'])

        metadata_df.loc[r,'sigma_v_4MOST_kmpersec'] = sigma_v


def populate_truth_sigma_v_IFU(metadata_df,gt_cosmo_astropy):
    """Using ground truth lens properties + a ground truth cosmology, computes
        the velocity dispersion in bins of MUSE and JWST NIRSPEC

    Returns:
        modifies metadata_df in place
    """

    for r in range(0,len(metadata_df)):

        # compute MUSE kin
        sigma_v_muse = galkin_utils.ground_truth_ifu_vdisp(
            galkin_utils.kinematicsAPI_MUSE,
            metadata_df.loc[r,'main_deflector_parameters_theta_E'],
            metadata_df.loc[r,'main_deflector_parameters_gamma'],
            metadata_df.loc[r,'lens_light_parameters_R_sersic'],
            metadata_df.loc[r,'lens_light_parameters_n_sersic'],
            metadata_df.loc[r,'main_deflector_parameters_z_lens'],
            metadata_df.loc[r,'source_parameters_z_source'],
            gt_cosmo_astropy,
            beta_ani=metadata_df.loc[r,'beta_ani']
        )
        # lambda_int scaling
        sigma_v_muse *= np.sqrt(metadata_df.loc[r,'lambda_int'])

        # write in the value!
        for b in range(0,len(sigma_v_muse)):
            metadata_df.loc[r,'sigma_v_MUSE_bin%d_kmpersec'%(b)] = sigma_v_muse[b]

        # compute NIRSPEC kin
        sigma_v_nirspec = galkin_utils.ground_truth_ifu_vdisp(
            galkin_utils.kinematicsAPI_NIRSPEC,
            metadata_df.loc[r,'main_deflector_parameters_theta_E'],
            metadata_df.loc[r,'main_deflector_parameters_gamma'],
            metadata_df.loc[r,'lens_light_parameters_R_sersic'],
            metadata_df.loc[r,'lens_light_parameters_n_sersic'],
            metadata_df.loc[r,'main_deflector_parameters_z_lens'],
            metadata_df.loc[r,'source_parameters_z_source'],
            gt_cosmo_astropy,
            beta_ani=metadata_df.loc[r,'beta_ani']
        )
        # lambda_int scaling
        sigma_v_nirspec *= np.sqrt(metadata_df.loc[r,'lambda_int'])

        # write in the value!
        for b in range(0,len(sigma_v_nirspec)):
            metadata_df.loc[r,'sigma_v_NIRSPEC_bin%d_kmpersec'%(b)] = sigma_v_nirspec[b]


def slsim_catalog_to_fasttdc_catalog(slsim_catalog_path):
    """
    Converts an slsim catalog to a pandas dataframe with keys formatted for
        fasttdc convention
    
    Args:
        slsim_catalog_path (string): path to a .fits file containing
            the slsim catalog

    Returns:
        pandas.DataFrame()

    NOTE: .byteswap().newbyteorder() is required to switch byte ordering convention
        from .fits -> pd.DataFrame()
    """

    # hard-coded column map from slsim convention to fasttdc convention
    column_map = {
        # MAIN DEFLECTOR
        'x_deflector_mass_position_arcsec': 'main_deflector_parameters_center_x',
        'y_deflector_mass_position_arcsec': 'main_deflector_parameters_center_y',
        'deflector_mass_e1': 'main_deflector_parameters_e1',
        'deflector_mass_e2': 'main_deflector_parameters_e2',
        'deflector_pl_slope': 'main_deflector_parameters_gamma',
        'external_shear_gamma1': 'main_deflector_parameters_gamma1',
        'external_shear_gamma2': 'main_deflector_parameters_gamma2',
        'theta_E_arcsec': 'main_deflector_parameters_theta_E',
        # SOURCE LIGHT
        'host_light_R_eff_arcsec': 'source_parameters_R_sersic',
        'x_host_position_arcsec': 'source_parameters_center_x',
        'y_host_position_arcsec': 'source_parameters_center_y',
        'host_light_e1': 'source_parameters_e1',
        'host_light_e2': 'source_parameters_e2',
        'unlensed_host_mag_i': 'source_parameters_mag_app', # TODO: decide what to use
        'host_light_n_sersic': 'source_parameters_n_sersic',
        # 'source_parameters_output_ab_zeropoint', do we need this?
        # LENS LIGHT
        'deflector_light_R_eff_arcsec': 'lens_light_parameters_R_sersic',
        'x_deflector_light_position_arcsec': 'lens_light_parameters_center_x',
        'y_deflector_light_position_arcsec': 'lens_light_parameters_center_y',
        'deflector_light_e1': 'lens_light_parameters_e1',
        'deflector_light_e2': 'lens_light_parameters_e2',
        'deflector_mag_i': 'lens_light_parameters_mag_app', # TODO: decide what to use
        'deflector_light_n_sersic': 'lens_light_parameters_n_sersic',
        # 'lens_light_parameters_output_ab_zeropoint': 'lens_light_parameters_output_ab_zeropoint',
        # POINT SOURCE
        'unlensed_ps_mag_i': 'point_source_parameters_mag_app',
        #point_source_parameters_output_ab_zeropoint
        'x_ps_position_arcsec': 'point_source_parameters_x_point_source',
        'y_ps_position_arcsec': 'point_source_parameters_y_point_source',
        'num_images': 'point_source_parameters_num_images',
        'x_ps_image_positions_arcsec_1': 'point_source_parameters_x_image_0',
        'y_ps_image_positions_arcsec_1': 'point_source_parameters_y_image_0',
        'x_ps_image_positions_arcsec_2': 'point_source_parameters_x_image_1',
        'y_ps_image_positions_arcsec_2': 'point_source_parameters_y_image_1',
        'x_ps_image_positions_arcsec_3': 'point_source_parameters_x_image_2',
        'y_ps_image_positions_arcsec_3': 'point_source_parameters_y_image_2',
        'x_ps_image_positions_arcsec_4': 'point_source_parameters_x_image_3',
        'y_ps_image_positions_arcsec_4': 'point_source_parameters_y_image_3'
    }


    # load in slsim catalog
    with fits.open(slsim_catalog_path) as hdul:
        slsim_catalog = hdul[1].data  # assuming the catalog is in the first extension

    # first put data into a dict, then into dataframe (to avoid fragmentation)
    fasttdc_catalog_dict = {}

    # loop thru all keys in slsim catalog
    for col in slsim_catalog.dtype.names:
        # if in column map, use the new column name
        if col in column_map.keys():
            fasttdc_catalog_dict[column_map[col]] = slsim_catalog[col].byteswap().newbyteorder()
        # else, save the existing column
        else:
            fasttdc_catalog_dict[col] = slsim_catalog[col].byteswap().newbyteorder()

    # account for edge cases 
    # handle columns that duplicate information (but are necessary for paltas usage)
    fasttdc_catalog_dict['main_deflector_parameters_z_lens'] = slsim_catalog['z_D'].byteswap().newbyteorder()
    fasttdc_catalog_dict['lens_light_parameters_z_source'] = slsim_catalog['z_D'].byteswap().newbyteorder()
    fasttdc_catalog_dict['source_parameters_z_source'] = slsim_catalog['z_S'].byteswap().newbyteorder()
    fasttdc_catalog_dict['point_source_parameters_z_point_source'] = slsim_catalog['z_S'].byteswap().newbyteorder()

    # columns that need to be manually added
    fasttdc_catalog_dict['main_deflector_parameters_dec_0'] = np.zeros(len(slsim_catalog['z_S']))
    fasttdc_catalog_dict['main_deflector_parameters_ra_0'] = np.zeros(len(slsim_catalog['z_S']))

    # now instantiate dataframe from the dict
    fasttdc_catalog_df = pd.DataFrame(fasttdc_catalog_dict)

    return fasttdc_catalog_df