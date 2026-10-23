import torch
import pytest
import numpy as np
from mentat_lss.emulator import ps_emulator
from mentat_lss.utils import *

def test_normalize_cosmo_params():
    test_tensor = (torch.rand(10*10) * 100) - 50
    normalizations = torch.tensor([-10, 10])
    manual_norm_tensor = (test_tensor + 10) / (20)

    norm_tensor = normalize_cosmo_params(test_tensor, normalizations)
    unnorm_tensor = norm_tensor * (normalizations[1] - normalizations[0]) + normalizations[0]

    assert torch.allclose(manual_norm_tensor, norm_tensor)
    assert torch.allclose(test_tensor, unnorm_tensor)

# TODO: This test is wildly inconsistent
def test_normalize_power_spectra():
    
    test_fid_tensor = (torch.ones((3, 2, 2*25)) * 100) - 25

    test_cov = torch.rand(3, 2, 2*25, 2*25)
    test_eigvals = torch.zeros((3, 2, 2*25))
    test_Q = torch.zeros_like(test_cov)
    test_Q_inv = torch.zeros_like(test_cov)
    
    for i in range(test_cov.shape[0]):
        for j in range(test_cov.shape[1]):
            cov = test_cov[i,j] @ test_cov[i,j].T
            eig, q = torch.linalg.eigh(cov)
            test_eigvals[i,j] = eig.real
            test_Q[i,j] = q.real
            test_Q_inv[i,j] = torch.linalg.inv(q).real

    test_tensor = (torch.ones((1, 3, 2, 2*25)) * 100) - 15

    norm_tensor = normalize_power_spectrum(test_tensor, test_fid_tensor, test_eigvals, test_Q)
    unnorm_tensor = un_normalize_power_spectrum(norm_tensor, test_fid_tensor, test_eigvals, test_Q, test_Q_inv)

    # numerical errors from normalization process expected to be higher than floating point percision
    assert torch.all(torch.isnan(norm_tensor)) == False
    assert torch.amax((test_tensor - unnorm_tensor) / test_tensor) < 0.075


def test_get_parameter_ranges():

    input_dict = {"cosmo_params" : {"fnl" : {"value" : 0., "prior" : {"min" : -50, "max": 50}},
                                    "As" : {"value" : 2.1e-9}},
                  "nuisance_params" : {"b1_0_0" : {"value" : 1.5, "prior" : {"min" : 1, "max": 3}},
                                  "b2" : {"value" : -0.4, "prior" : {"min" : -1, "max": 0}}} }
    params, priors = get_parameter_ranges(input_dict)

    expected_params = ["fnl", "b1_0_0", "b2"]
    expected_priors = np.array([[-50, 50],
                                [1, 3],
                                [-1, 0]])
    
    assert expected_params == params
    assert np.all(np.equal(expected_priors, priors))

@pytest.mark.parametrize("batch_size", [1, 2, 8, 200])
def test_hyperbolic_chi2(batch_size):
    """The hyperbolic transform must be applied per sample, then averaged."""
    torch.manual_seed(0)
    invcov = torch.eye(10)
    predict = torch.randn(batch_size, 10) * 0.1
    target = torch.zeros(batch_size, 10)

    per_sample_chi2 = ((predict - target) ** 2).sum(dim=1)
    expected = torch.mean(torch.sqrt(1 + 2 * per_sample_chi2)) - 1

    assert torch.allclose(hyperbolic_chi2_loss(predict, target, invcov, True), expected)

@pytest.mark.parametrize("input, output, invcov, normalized, expected", [
    (torch.ones(1,10), torch.ones(1,10), torch.eye(10), True, 0),
    (torch.ones(10), torch.ones(10), torch.eye(10), True, 0),
    (torch.ones(10), torch.ones(1, 10), torch.eye(10), True, ValueError),
    (torch.ones(1,10), torch.ones(1,10), torch.eye(10), False, 0),
])
def test_delta_chi2(input, output, invcov, normalized, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            chi2 = delta_chi_squared(input, output, invcov, normalized)
    else:
        chi2 = delta_chi_squared(input, output, invcov, normalized)
        assert torch.allclose(chi2, torch.Tensor(expected))

@pytest.mark.parametrize("num_spectra, num_zbins, num_kbins, num_ells", [
    (1, 1, 20, 2)
])
def test_get_invcov_blocks(num_spectra, num_zbins, num_kbins, num_ells):

    test_cov = torch.rand(num_zbins, num_spectra*num_kbins*num_ells, num_spectra*num_kbins*num_ells)

    invcov_blocks = get_invcov_blocks(test_cov, num_spectra, num_zbins, num_kbins, num_ells)
    assert invcov_blocks.shape == (num_spectra, num_zbins, num_ells*num_kbins, num_ells*num_kbins)
    
@pytest.mark.parametrize("factor, type, expected",[
    (0.5, "np", True),
    (0.6, "torch", True),
    (1.0, "torch", False)
])
def test_is_in_hypersphere(factor, type, expected):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    test_cosmo_dir = current_dir+"/../configs/cosmo_pars/cosmo_pars_example.yaml"

    cosmo_dict = load_config_file(test_cosmo_dir)
    param_names, param_bounds = get_parameter_ranges(cosmo_dict)
    
    if type == "np":
        test_params = np.ones(len(param_names)) * factor
    elif type == "torch":
        test_params = torch.ones(len(param_names)) * factor
        param_bounds = torch.from_numpy(param_bounds)
    test_params = test_params * (param_bounds[:,1] - param_bounds[:,0]) + param_bounds[:,0]

    result, r = is_in_hypersphere(param_bounds, test_params)

    assert result == expected
    if result == False:
        assert r > 1.
