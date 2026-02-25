# This script generates a training set for the spherex emulator based on Yosuke's EFT power spectrum model
# Utilizing an embarrasingly parallel code structure

import time, math, yaml, sys, os
import numpy as np
from mpi4py import MPI

import ps_theory_calculator_camb as ps_theory_calculator
from mentat_lss.utils import *

#-------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------

# ells to generate
# TODO: Move this to a better spot
ells = [0, 2]

# fraction of dataset to be partitioned to the training | validation | test sets
train_frac = 0.8
valid_frac = 0.1
test_frac  = 0.1

#-------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------

def is_fiducial_cosmology_in_hyperspere(cosmo_dict, param_ranges):
    cosmo_fid = []
    for param in cosmo_dict["cosmo_params"].keys():
        if "prior" in cosmo_dict["cosmo_params"][param]:
            cosmo_fid.append(cosmo_dict["cosmo_params"][param]["value"])
    for param in cosmo_dict["nuisance_params"].keys():
        if "prior" in cosmo_dict["nuisance_params"][param]:
            cosmo_fid.append(cosmo_dict["nuisance_params"][param]["value"])

    return is_in_hypersphere(param_ranges, np.array(cosmo_fid))

def prepare_header_info(param_names, fiducial_cosmology, n_samples):

    header_info = {}
    header_info["total_samples"] = n_samples
    cosmo_params = []
    bias_params = []
    for key in fiducial_cosmology["cosmo_params"].keys():
        if key in param_names:
            cosmo_params.append(key)
    for key in fiducial_cosmology["nuisance_params"].keys():
        if key in param_names:
            bias_params.append(key)

    header_info["cosmo_params"] = cosmo_params
    header_info["nuisance_params"] = bias_params
    return header_info

def get_power_spectrum(samples, k, param_names, cosmo_dict, ps_config, theory):

    num_tracers = ps_config['number_density_table'].shape[0]
    num_spectra = num_tracers + math.comb(num_tracers, 2)
    num_zbins = len(ps_config["redshift_list"])

    num_to_calculate = len(samples)
    galaxy_ps = np.zeros((num_to_calculate, num_spectra, num_zbins, len(k), len(ells)))
    result = np.zeros(num_to_calculate)

    for idx in range(num_to_calculate):
        sample_dict = dict(zip(param_names, samples[idx]))
        param_vector = prepare_ps_inputs(sample_dict, cosmo_dict, num_tracers, num_zbins)
        try:
            ps = theory(k, ells, param_vector) / cosmo_dict["cosmo_params"]["h"]["value"]**3
            ps = np.transpose(ps, (1, 0, 3, 2))
            if not np.any(np.isnan(ps)) and \
                not np.any(np.isinf(ps)): 
                galaxy_ps[idx] = ps
            else: 
                print("NaN or inf fouund in power spectrum!")
                result[idx] = -1
        except:
            print("Power spectrum calculation failed!")
            result[idx] = -1

    return galaxy_ps, result
#-------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------
def main():

    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    if len(sys.argv) < 7:
        if rank == 0: print("USAGE: python make_training_set_eft.py <cosmo_config_file> <survey_config_file> <save_dir> <k array file> <N> <mode>")
        return -1

    cosmo_config_file  = sys.argv[1]
    survey_config_file = sys.argv[2]
    save_dir           = sys.argv[3]
    k_array_file       = sys.argv[4]
    N                  = int(sys.argv[5])
    mode               = sys.argv[6]

    if not os.path.exists(save_dir):
        print("Attempting to create save directory...")
        os.mkdir(save_dir)

    cosmo_dict  = load_config_file(cosmo_config_file)
    survey_dict = load_config_file(survey_config_file)

    # TODO: Upgrade to handle different k-bins for different redshifts
    k_data = np.load(k_array_file)
    k = k_data["k"]

    ndens_table = np.array([[float(survey_dict['number_density_in_hinvMpc_%s' % (i+1)][j]) for j in range(survey_dict['nz'])] for i in range(survey_dict['nsample'])])
    z_eff = (np.array(survey_dict["zbin_lo"]) + np.array(survey_dict["zbin_hi"])) / 2.
    num_spectra = ndens_table.shape[0] + math.comb(ndens_table.shape[0], 2)

    # table of number densities of tracer samples. 
    # (i, j) component is the number density of the i-th sample at the j-th redshift.
    ps_config = {}
    ps_config['number_density_table'] = ndens_table
    ps_config['redshift_list'] = z_eff # redshift bins
    ps_config['Omega_m_ref'] = 0.3 # Omega_m value of the reference cosmology (assuming a flat LambdaCDM)

    param_names, param_ranges = get_parameter_ranges(cosmo_dict)
    num_params = len(param_names)

    # Split up samples to multiple MPI ranks
    if rank == 0: 
        if   mode == "hypercube":   
            all_samples = make_latin_hypercube(param_ranges, N)
        elif mode == "hypersphere": 
            all_samples = make_hypersphere(param_ranges, num_params, N)
            is_valid_cosmo, r = is_fiducial_cosmology_in_hyperspere(cosmo_dict, param_ranges)
            if not is_valid_cosmo:
                raise ValueError(f"Fiducial cosmology is outside of hypersphere! r = {r}")
            else:
                print("Verified fiducial cosmology lies within hypersphere")
    else:         
        all_samples = np.zeros((N, num_params))
    comm.Barrier()
    comm.Bcast(all_samples, root=0)

    assert N % size == 0
    offset = int((N / size) * rank)
    data_len = int(N / size)
    rank_samples = all_samples[offset:offset+data_len,:]
    assert rank_samples.shape[0] == data_len

    theory = ps_theory_calculator.PowerSpectrumMultipole1Loop(ps_config)

    if rank == 0:
        print("Number of samples:", ps_config['number_density_table'].shape[0])
        print("Number of redshift bins:", len(ps_config["redshift_list"]))
        print("Number of varied parameters:", len(param_names),':', param_names)
        print("samping mode:", mode)
        print("Saving to", save_dir)

        # first, generate the power spectrum at the fiducial cosmology
        print("Generating fiducial power spectrum...")
        galaxy_ps, result = get_power_spectrum([{}], k, param_names, cosmo_dict, ps_config, theory)
        if result == 0:
            np.save(os.path.join(save_dir,"ps_fid.npy"), galaxy_ps)
            np.savez(os.path.join(save_dir,"ps_properties.npz"), k=k, z_eff=z_eff, ells=np.array(ells), ndens=ndens_table)
        else:
            print("ERROR! failed to calculate fiducial power spectrum! Exiting...")
            return -1

    t1 = time.time()
    if rank == 0: print("Generating", str(int(N)), "power spectra across", str(size), "processors ("+str(int(N / size))+" per processor)...")

    galaxy_ps, result = get_power_spectrum(rank_samples, k, param_names, cosmo_dict, ps_config, theory)

    # aggregate data
    galaxy_ps = np.array(galaxy_ps)
    result = np.array(result)

    idx_pass = np.where(result == 0)[0]
    fail_compute = len(np.where(result == -1)[0])

    galaxy_ps = galaxy_ps[idx_pass]
    rank_samples = rank_samples[idx_pass]

    dataset_info = prepare_header_info(param_names, cosmo_dict, N - fail_compute)

    if galaxy_ps.shape[0] > 1:
        np.savez(os.path.join(save_dir,"pk-raw_"+str(rank)+"_.npz"), params=rank_samples, galaxy_ps=galaxy_ps)
    
    if rank == 0:
        with open(os.path.join(save_dir,'info.yaml'), 'w') as outfile:
            yaml.dump(dataset_info, outfile, sort_keys=False, default_flow_style=False)

    del galaxy_ps

    t2 = time.time()
    print("Rank {:d} Done! Took {:0.0f} hours {:0.0f} minutes".format(rank, math.floor((t2 - t1)/3600), math.floor((t2 - t1)/60%60)))
    print("Made {:0.0f} / {:0.0f} galaxy power spectra".format((N/size) - fail_compute, N/size))
    #print("{:0.0f} ({:0.2f}%) power spectra failed to compute".format(fail_compute, 100.*fail_compute / (N/size)))

    comm.Barrier()
    if rank == 0:
        print("\nRe-organizing data to training / validation / test sets...")
        organize_training_set(save_dir, train_frac, valid_frac, test_frac,
                              rank_samples.shape[1], len(z_eff), num_spectra, len(ells), len(k), True)

if __name__ == "__main__":
    main()
