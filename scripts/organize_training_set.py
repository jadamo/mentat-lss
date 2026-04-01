# This script generates a training set for the spherex emulator based on Yosuke's EFT power spectrum model
# Utilizing an embarrasingly parallel code structure

import time, math, yaml
import numpy as np

from multiprocessing import Pool
from itertools import repeat

import ps_theory_calculator
from mentat_lss.utils import *

#-------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------


k = np.array([0.00694, 0.01482, 0.0227, 0.03058, 0.03846, 0.04634, 0.05422, 0.0621, 0.06998,
              0.07786, 0.08574, 0.09362, 0.1015,  0.10938, 0.11726, 0.12514, 0.13302, 0.1409,
              0.14878, 0.15666, 0.16454, 0.17242, 0.1803,  0.18818, 0.19606])

# fraction of dataset to be partitioned to the training | validation | test sets
train_frac = 0.8
valid_frac = 0.1
test_frac  = 0.1

net_config_file = filepaths.network_pars_dir+"network_pars_3_sample_1_redshift.yaml"
cosmo_config_file = filepaths.cosmo_pars_dir+"eft_3_sample_1_redshift.yaml"
survey_config_file = filepaths.survey_pars_dir+'survey_pars_3_sample_1_redshift.yaml'
#Joes directory: save_dir = "/home/joeadamo/Research/Data/SPHEREx-Data/Training-Set-EFT-2s-2z/"

#Same filepath to save as tns training set
#save_dir = filepaths.data_dir
save_dir = '/xdisk/timeifler/jadamo/Training-Set-EFT-3s-1z/'

#-------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------
def main():
    
    assert os.path.exists(save_dir)

    cosmo_dict  = load_config_file(cosmo_config_file)
    survey_dict = load_config_file(survey_config_file)
    config_dict = load_config_file(net_config_file)

    ndens_table = np.array([[float(survey_dict['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_dict['nz'])] for i in range(survey_dict['nsample'])])
    z_eff = (np.array(survey_dict["zbin_lo"]) + np.array(survey_dict["zbin_hi"])) / 2.
    num_spectra = ndens_table.shape[0] + math.comb(ndens_table.shape[0], 2)
    num_ells = 5

    # table of number densities of tracer samples. 
    # (i, j) component is the number density of the i-th sample at the j-th redshift.
    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = z_eff # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)

    param_names, param_ranges = get_parameter_ranges(cosmo_dict)

    organize_training_set(save_dir, train_frac, valid_frac, test_frac, "ps", len(param_names),
                          num_zbins=len(z_eff), num_spectra=num_spectra, num_ells=num_ells,
                          k_dim=len(k), remove_old_files=False)

if __name__ == "__main__":
    main()
